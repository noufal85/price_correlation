"""Flask web server for stock clustering pipeline."""

import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, Response

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# Pipeline run state with progress tracking
pipeline_state = {
    "running": False,
    "last_run": None,
    "last_result": None,
    "error": None,
    "current_step": 0,
    "total_steps": 8,
    "step_name": "",
    "step_details": "",
    "logs": [],
    "start_time": None,
    "kill_requested": False,
    "thread": None,
}

# Pipeline steps for progress tracking
PIPELINE_STEPS = [
    {"num": 1, "name": "Getting Universe", "description": "Fetching list of tickers to analyze"},
    {"num": 2, "name": "Fetching Prices", "description": "Downloading historical price data"},
    {"num": 3, "name": "Preprocessing", "description": "Computing returns and filtering data"},
    {"num": 4, "name": "Correlations", "description": "Building correlation matrix"},
    {"num": 5, "name": "Clustering", "description": "Running clustering algorithm"},
    {"num": 6, "name": "Validation", "description": "Computing cluster quality metrics"},
    {"num": 7, "name": "Visualization", "description": "Generating t-SNE plot"},
    {"num": 8, "name": "Exporting", "description": "Saving results to files and database"},
]


class PipelineLogCapture:
    """Capture print statements during pipeline execution."""

    def __init__(self, state):
        self.state = state
        self.original_stdout = sys.stdout

    def write(self, text):
        self.original_stdout.write(text)
        if text.strip():
            self.state["logs"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": text.strip()
            })
            # Keep only last 500 log entries
            if len(self.state["logs"]) > 500:
                self.state["logs"] = self.state["logs"][-500:]

            # Detect step changes
            if text.startswith("Step "):
                try:
                    step_num = int(text.split(":")[0].replace("Step ", ""))
                    step_desc = text.split(":", 1)[1].strip() if ":" in text else ""
                    self.state["current_step"] = step_num
                    self.state["step_name"] = step_desc
                except (ValueError, IndexError):
                    pass
            elif text.startswith("  "):
                self.state["step_details"] = text.strip()

    def flush(self):
        self.original_stdout.flush()


def get_fmp_api_key():
    """Get FMP API key from environment."""
    return os.environ.get("FMP_API_KEY")


def get_db_connection():
    """Get database connection."""
    from .db import get_db_config
    import psycopg2

    config = get_db_config()
    try:
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


# =============================================================================
# Template Routes
# =============================================================================


@app.route("/")
def index():
    """Dashboard page."""
    return render_template("index.html")


@app.route("/clusters")
def clusters_page():
    """Clusters view page."""
    return render_template("clusters.html")


@app.route("/correlations")
def correlations_page():
    """Correlations view page."""
    return render_template("correlations.html")


@app.route("/stock/<ticker>")
def stock_detail(ticker):
    """Stock detail page."""
    return render_template("stock_detail.html", ticker=ticker.upper())


@app.route("/charts")
def charts_page():
    """Charts page."""
    return render_template("charts.html")


@app.route("/steps")
def steps_page():
    """Step-by-step pipeline page."""
    return render_template("steps.html")


@app.route("/pipeline")
def pipeline_page():
    """Pipeline control page."""
    return render_template("pipeline.html", steps=PIPELINE_STEPS)


# =============================================================================
# API Routes - Pipeline Control
# =============================================================================


@app.route("/api/pipeline/status")
def pipeline_status():
    """Get pipeline status with progress details."""
    elapsed = None
    if pipeline_state["start_time"] and pipeline_state["running"]:
        elapsed = time.time() - pipeline_state["start_time"]

    return jsonify({
        "running": pipeline_state["running"],
        "last_run": pipeline_state["last_run"],
        "last_result": pipeline_state["last_result"],
        "error": pipeline_state["error"],
        "current_step": pipeline_state["current_step"],
        "total_steps": pipeline_state["total_steps"],
        "step_name": pipeline_state["step_name"],
        "step_details": pipeline_state["step_details"],
        "elapsed_seconds": elapsed,
        "kill_requested": pipeline_state["kill_requested"],
    })


@app.route("/api/pipeline/logs")
def pipeline_logs():
    """Get pipeline logs."""
    since = request.args.get("since", 0, type=int)
    logs = pipeline_state["logs"][since:]
    return jsonify({
        "logs": logs,
        "total": len(pipeline_state["logs"]),
    })


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """Trigger pipeline run."""
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline already running"}), 400

    # Get parameters from request
    data = request.get_json() or {}

    # Support both old and new API format
    if "data_source" in data:
        # New format
        data_source = data.get("data_source", "sample")
        filters = data.get("filters", {})
        max_stocks = data.get("max_stocks", 0)
    else:
        # Old format (backward compatible)
        use_sample = data.get("use_sample", True)
        data_source = "sample" if use_sample else "fmp_filtered"
        filters = {}
        max_stocks = data.get("sample_size", 50) if use_sample else 0

    days = data.get("days", 180)
    method = data.get("method", "hierarchical")

    # Reset state BEFORE starting thread so UI sees it immediately
    pipeline_state["running"] = True
    pipeline_state["error"] = None
    pipeline_state["last_run"] = datetime.now().isoformat()
    pipeline_state["last_result"] = None
    pipeline_state["current_step"] = 0
    pipeline_state["step_name"] = "Initializing"
    pipeline_state["step_details"] = ""
    pipeline_state["logs"] = []
    pipeline_state["start_time"] = time.time()
    pipeline_state["kill_requested"] = False

    def run_in_background():
        # Capture stdout for logging
        log_capture = PipelineLogCapture(pipeline_state)
        old_stdout = sys.stdout

        try:
            sys.stdout = log_capture

            from .pipeline import PipelineConfig, run_pipeline as run

            config = PipelineConfig(
                data_source=data_source,
                filters=filters,
                max_stocks=max_stocks,
                period_months=days // 30,
                clustering_method=method,
                visualize=True,
            )
            result = run(config)
            pipeline_state["last_result"] = result
            pipeline_state["current_step"] = pipeline_state["total_steps"]
            pipeline_state["step_name"] = "Completed"
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            pipeline_state["error"] = str(e)
            pipeline_state["step_name"] = "Error"
        finally:
            sys.stdout = old_stdout
            pipeline_state["running"] = False
            pipeline_state["thread"] = None

    thread = threading.Thread(target=run_in_background)
    pipeline_state["thread"] = thread
    thread.start()

    return jsonify({"message": "Pipeline started", "status": "running"})


@app.route("/api/universe/preview", methods=["POST"])
def preview_universe():
    """Preview universe count based on filters without running pipeline."""
    data = request.get_json() or {}
    data_source = data.get("data_source", "fmp_filtered")
    filters = data.get("filters", {})

    # Handle sample data source directly
    if data_source == "sample":
        from .universe import get_sample_tickers
        sample_tickers = get_sample_tickers(50)
        return jsonify({
            "total_count": len(sample_tickers),
            "by_exchange": {"Sample": len(sample_tickers)},
            "filters_applied": {},
            "sample_tickers": sample_tickers[:10],
        })

    # Check for FMP API key
    api_key = get_fmp_api_key()
    if not api_key:
        return jsonify({
            "error": "FMP API key not configured. Set FMP_API_KEY environment variable."
        }), 400

    try:
        from .fmp_client import FMPClient

        client = FMPClient(api_key=api_key)
        result = client.preview_universe(
            data_source=data_source,
            market_cap_min=filters.get("market_cap_min"),
            market_cap_max=filters.get("market_cap_max"),
            volume_min=filters.get("volume_min"),
            volume_max=filters.get("volume_max"),
        )

        # Don't include full stocks list in response (too large)
        return jsonify({
            "total_count": result["total_count"],
            "by_exchange": result["by_exchange"],
            "filters_applied": result["filters_applied"],
            "sample_tickers": result["sample_tickers"],
        })

    except Exception as e:
        logger.exception(f"Preview universe error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/kill", methods=["POST"])
def kill_pipeline():
    """Request to kill the running pipeline."""
    if not pipeline_state["running"]:
        return jsonify({"error": "No pipeline running"}), 400

    pipeline_state["kill_requested"] = True
    pipeline_state["logs"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": "Kill requested - pipeline will stop after current step"
    })

    return jsonify({
        "message": "Kill requested",
        "status": "kill_pending"
    })


# =============================================================================
# API Routes - Database Queries
# =============================================================================


@app.route("/api/runs")
def get_runs():
    """Get list of analysis runs."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT analysis_date, n_stocks_processed, n_clusters, n_noise,
                   silhouette_score, clustering_method, execution_time_seconds, created_at
            FROM price_correlation.analysis_runs
            ORDER BY analysis_date DESC
            LIMIT 50
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        runs = []
        for row in rows:
            runs.append({
                "analysis_date": row[0].isoformat() if row[0] else None,
                "n_stocks_processed": row[1],
                "n_clusters": row[2],
                "n_noise": row[3],
                "silhouette_score": row[4],
                "clustering_method": row[5],
                "execution_time_seconds": row[6],
                "created_at": row[7].isoformat() if row[7] else None,
            })

        return jsonify(runs)
    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/clusters")
def get_clusters():
    """Get cluster assignments for latest run."""
    analysis_date = request.args.get("date")
    method = request.args.get("method")

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        # Get the latest analysis date if not specified
        if analysis_date:
            date_filter = analysis_date
        else:
            cursor.execute("SELECT MAX(analysis_date) FROM price_correlation.equity_clusters")
            result = cursor.fetchone()
            date_filter = result[0] if result[0] else date.today()

        # Get available methods for this date
        cursor.execute("""
            SELECT DISTINCT clustering_method
            FROM price_correlation.equity_clusters
            WHERE analysis_date = %s
            ORDER BY clustering_method
        """, (date_filter,))
        available_methods = [row[0] for row in cursor.fetchall()]

        # Use first available method if none specified
        if not method and available_methods:
            method = available_methods[0]
        elif not method:
            method = "hierarchical"

        # Get clusters for the specified method
        cursor.execute("""
            SELECT ticker, cluster_id
            FROM price_correlation.equity_clusters
            WHERE analysis_date = %s AND clustering_method = %s
            ORDER BY cluster_id, ticker
        """, (date_filter, method))
        rows = cursor.fetchall()

        # Get insights from analysis_runs
        cursor.execute("""
            SELECT silhouette_score, execution_time_seconds, created_at
            FROM price_correlation.analysis_runs
            WHERE analysis_date = %s AND clustering_method = %s
        """, (date_filter, method))
        run_info = cursor.fetchone()

        cursor.close()
        conn.close()

        # Group by cluster
        clusters = {}
        for ticker, cluster_id in rows:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(ticker)

        # Calculate cluster statistics
        cluster_sizes = [len(members) for cid, members in clusters.items() if cid != -1]
        avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        insights = {
            "silhouette_score": round(run_info[0], 4) if run_info and run_info[0] else None,
            "execution_time": round(run_info[1], 1) if run_info and run_info[1] else None,
            "created_at": run_info[2].isoformat() if run_info and run_info[2] else None,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "avg_cluster_size": round(avg_size, 1),
            "median_cluster_size": sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0,
            "std_cluster_size": round((sum((s - avg_size)**2 for s in cluster_sizes) / len(cluster_sizes))**0.5, 1) if len(cluster_sizes) > 1 else 0,
        }

        return jsonify({
            "analysis_date": str(date_filter),
            "clustering_method": method,
            "available_methods": available_methods,
            "clusters": clusters,
            "total_stocks": len(rows),
            "n_clusters": len([k for k in clusters if k != -1]),
            "n_noise": len(clusters.get(-1, [])),
            "insights": insights,
        })
    except Exception as e:
        logger.error(f"Error fetching clusters: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/correlations")
def get_correlations():
    """Get correlation pairs for latest run."""
    analysis_date = request.args.get("date")
    min_corr = float(request.args.get("min_corr", 0.7))
    limit = int(request.args.get("limit", 100))

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        if analysis_date:
            date_filter = analysis_date
        else:
            cursor.execute("SELECT MAX(analysis_date) FROM price_correlation.pair_correlations")
            result = cursor.fetchone()
            date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker_a, ticker_b, correlation
            FROM price_correlation.pair_correlations
            WHERE analysis_date = %s AND correlation >= %s
            ORDER BY correlation DESC
            LIMIT %s
        """, (date_filter, min_corr, limit))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        pairs = []
        for row in rows:
            pairs.append({
                "ticker_a": row[0],
                "ticker_b": row[1],
                "correlation": round(row[2], 4),
            })

        return jsonify({
            "analysis_date": str(date_filter),
            "pairs": pairs,
            "count": len(pairs),
        })
    except Exception as e:
        logger.error(f"Error fetching correlations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<ticker>/prices")
def get_stock_prices(ticker):
    """Get price history for a stock."""
    import yfinance as yf
    from datetime import datetime, timedelta

    # Get period from query params (default 18 months)
    months = int(request.args.get("months", 18))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)

    try:
        data = yf.download(
            ticker.upper(),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return jsonify({"error": "No price data found", "ticker": ticker}), 404

        prices = []
        for date, row in data.iterrows():
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(float(row["Close"]), 2),
            })

        return jsonify({
            "ticker": ticker.upper(),
            "prices": prices,
            "start_date": prices[0]["date"] if prices else None,
            "end_date": prices[-1]["date"] if prices else None,
            "count": len(prices),
        })

    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<ticker>/fundamentals")
def get_stock_fundamentals(ticker):
    """Get company fundamentals and financial metrics."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Extract key metrics
        fundamentals = {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_percent": info.get("regularMarketChangePercent"),

            # Valuation
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),

            # Earnings
            "eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),

            # Revenue & Profitability
            "revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margin": info.get("grossMargins"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "ebitda": info.get("ebitda"),

            # Returns
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),

            # Balance Sheet
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),

            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),

            # Trading
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_day_avg": info.get("fiftyDayAverage"),
            "two_hundred_day_avg": info.get("twoHundredDayAverage"),
            "avg_volume": info.get("averageVolume"),

            # Company Info
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country"),
            "city": info.get("city"),
            "exchange": info.get("exchange"),
            "website": info.get("website"),
            "description": info.get("longBusinessSummary"),
        }

        return jsonify(fundamentals)

    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return jsonify({"error": str(e), "ticker": ticker}), 500


@app.route("/compare/<ticker_a>/<ticker_b>")
def compare_stocks(ticker_a, ticker_b):
    """Render stock comparison page."""
    correlation = request.args.get("corr", "N/A")
    return render_template(
        "compare.html",
        ticker_a=ticker_a.upper(),
        ticker_b=ticker_b.upper(),
        correlation=correlation,
    )


@app.route("/api/stock/<ticker>/cluster-history")
def get_stock_cluster_history(ticker):
    """Get cluster history for a stock."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT analysis_date, cluster_id
            FROM price_correlation.equity_clusters
            WHERE ticker = %s
            ORDER BY analysis_date DESC
            LIMIT 30
        """, (ticker.upper(),))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        history = []
        for row in rows:
            history.append({
                "date": row[0].isoformat() if row[0] else None,
                "cluster_id": row[1],
            })

        return jsonify({
            "ticker": ticker.upper(),
            "history": history,
        })
    except Exception as e:
        logger.error(f"Error fetching cluster history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/<ticker>/correlations")
def get_stock_correlations(ticker):
    """Get correlations for a specific stock with optional profile data."""
    limit = int(request.args.get("limit", 20))
    include_profiles = request.args.get("include_profiles", "false").lower() == "true"

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        # Get latest date
        cursor.execute("SELECT MAX(analysis_date) FROM price_correlation.pair_correlations")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker_a, ticker_b, correlation
            FROM price_correlation.pair_correlations
            WHERE analysis_date = %s AND (ticker_a = %s OR ticker_b = %s)
            ORDER BY correlation DESC
            LIMIT %s
        """, (date_filter, ticker.upper(), ticker.upper(), limit))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        pairs = []
        tickers_to_fetch = []
        for row in rows:
            other = row[1] if row[0] == ticker.upper() else row[0]
            pairs.append({
                "ticker": other,
                "correlation": round(row[2], 4),
            })
            tickers_to_fetch.append(other)

        # Fetch profiles in parallel if requested
        if include_profiles and tickers_to_fetch:
            profiles = fetch_profiles_batch(tickers_to_fetch)
            for pair in pairs:
                profile = profiles.get(pair["ticker"], {})
                pair["companyName"] = profile.get("companyName", "")
                pair["sector"] = profile.get("sector", "")
                pair["industry"] = profile.get("industry", "")

        return jsonify({
            "ticker": ticker.upper(),
            "analysis_date": str(date_filter),
            "correlations": pairs,
        })
    except Exception as e:
        logger.error(f"Error fetching stock correlations: {e}")
        return jsonify({"error": str(e)}), 500


def fetch_profiles_batch(tickers):
    """Fetch profiles for multiple tickers in parallel."""
    import concurrent.futures
    import requests

    api_key = get_fmp_api_key()
    if not api_key:
        return {}

    profiles = {}

    def fetch_one(ticker):
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
            response = requests.get(url, params={"apikey": api_key}, timeout=5)
            if response.ok:
                data = response.json()
                if data and len(data) > 0:
                    return ticker, data[0]
        except Exception as e:
            logger.debug(f"Profile fetch failed for {ticker}: {e}")
        return ticker, {}

    # Use thread pool to fetch in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_one, t) for t in tickers]
        for future in concurrent.futures.as_completed(futures):
            ticker, profile = future.result()
            profiles[ticker] = profile

    return profiles


@app.route("/api/profiles/batch")
def get_profiles_batch():
    """Get profiles for multiple tickers."""
    tickers_param = request.args.get("tickers", "")
    if not tickers_param:
        return jsonify({}), 200

    tickers = [t.strip() for t in tickers_param.split(",") if t.strip()]
    if not tickers:
        return jsonify({}), 200

    # Limit to 100 tickers max
    tickers = tickers[:100]

    profiles = fetch_profiles_batch(tickers)
    return jsonify(profiles)


@app.route("/api/cluster/<int:cluster_id>/stocks")
def get_cluster_stocks(cluster_id):
    """Get all stocks in a cluster."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        # Get latest date
        cursor.execute("SELECT MAX(analysis_date) FROM price_correlation.equity_clusters")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker
            FROM price_correlation.equity_clusters
            WHERE analysis_date = %s AND cluster_id = %s
            ORDER BY ticker
        """, (date_filter, cluster_id))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        tickers = [row[0] for row in rows]

        return jsonify({
            "cluster_id": cluster_id,
            "analysis_date": str(date_filter),
            "stocks": tickers,
            "count": len(tickers),
        })
    except Exception as e:
        logger.error(f"Error fetching cluster stocks: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# API Routes - FMP Data
# =============================================================================


def fmp_request(endpoint, params=None):
    """Make FMP API request."""
    import requests

    api_key = get_fmp_api_key()
    if not api_key:
        return None, "FMP API key not configured"

    base_url = "https://financialmodelingprep.com/api/v3"
    url = f"{base_url}/{endpoint}"

    params = params or {}
    params["apikey"] = api_key

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        logger.error(f"FMP API error: {e}")
        return None, str(e)


@app.route("/api/fmp/profile/<ticker>")
def get_company_profile(ticker):
    """Get company profile from FMP."""
    data, error = fmp_request(f"profile/{ticker.upper()}")
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data[0] if data else {})


@app.route("/api/fmp/quote/<ticker>")
def get_stock_quote(ticker):
    """Get real-time quote from FMP."""
    data, error = fmp_request(f"quote/{ticker.upper()}")
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data[0] if data else {})


@app.route("/api/fmp/news/<ticker>")
def get_stock_news(ticker):
    """Get news for a stock from FMP."""
    limit = request.args.get("limit", 10)
    data, error = fmp_request(f"stock_news", {"tickers": ticker.upper(), "limit": limit})
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data or [])


@app.route("/api/fmp/ratios/<ticker>")
def get_financial_ratios(ticker):
    """Get financial ratios from FMP."""
    data, error = fmp_request(f"ratios/{ticker.upper()}")
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data[0] if data else {})


@app.route("/api/fmp/peers/<ticker>")
def get_stock_peers(ticker):
    """Get stock peers from FMP."""
    data, error = fmp_request(f"stock_peers", {"symbol": ticker.upper()})
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data[0] if data and len(data) > 0 else {"peersList": []})


@app.route("/api/fmp/historical/<ticker>")
def get_historical_prices(ticker):
    """Get historical prices from FMP."""
    days = int(request.args.get("days", 30))
    data, error = fmp_request(f"historical-price-full/{ticker.upper()}", {"serietype": "line"})
    if error:
        return jsonify({"error": error}), 500

    if data and "historical" in data:
        return jsonify({
            "symbol": ticker.upper(),
            "historical": data["historical"][:days]
        })
    return jsonify({"symbol": ticker.upper(), "historical": []})


@app.route("/api/fmp/sector-performance")
def get_sector_performance():
    """Get sector performance from FMP."""
    data, error = fmp_request("sector-performance")
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data or [])


@app.route("/api/fmp/market-movers")
def get_market_movers():
    """Get market movers (gainers/losers) from FMP."""
    data_gainers, _ = fmp_request("stock_market/gainers")
    data_losers, _ = fmp_request("stock_market/losers")
    return jsonify({
        "gainers": (data_gainers or [])[:10],
        "losers": (data_losers or [])[:10],
    })


# =============================================================================
# API Routes - Charts Data
# =============================================================================


@app.route("/api/charts/cluster-sizes")
def chart_cluster_sizes():
    """Get cluster size data for charts."""
    method = request.args.get("method")

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(analysis_date) FROM price_correlation.equity_clusters")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        # Get default method if not specified
        if not method:
            cursor.execute("""
                SELECT DISTINCT clustering_method
                FROM price_correlation.equity_clusters
                WHERE analysis_date = %s
                ORDER BY clustering_method LIMIT 1
            """, (date_filter,))
            row = cursor.fetchone()
            method = row[0] if row else "hierarchical"

        cursor.execute("""
            SELECT cluster_id, COUNT(*) as size
            FROM price_correlation.equity_clusters
            WHERE analysis_date = %s AND clustering_method = %s
            GROUP BY cluster_id
            ORDER BY size DESC
        """, (date_filter, method))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        labels = []
        sizes = []
        for cluster_id, size in rows:
            if cluster_id == -1:
                labels.append("Noise")
            else:
                labels.append(f"Cluster {cluster_id}")
            sizes.append(size)

        return jsonify({
            "labels": labels,
            "data": sizes,
            "analysis_date": str(date_filter),
            "clustering_method": method,
        })
    except Exception as e:
        logger.error(f"Error fetching cluster sizes: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/charts/correlation-distribution")
def chart_correlation_distribution():
    """Get correlation distribution data for charts."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(analysis_date) FROM price_correlation.pair_correlations")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        # Get correlations and bin them
        cursor.execute("""
            SELECT correlation FROM price_correlation.pair_correlations
            WHERE analysis_date = %s
        """, (date_filter,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Create histogram bins
        bins = {"0.7-0.75": 0, "0.75-0.8": 0, "0.8-0.85": 0, "0.85-0.9": 0, "0.9-0.95": 0, "0.95-1.0": 0}
        for (corr,) in rows:
            if corr >= 0.95:
                bins["0.95-1.0"] += 1
            elif corr >= 0.9:
                bins["0.9-0.95"] += 1
            elif corr >= 0.85:
                bins["0.85-0.9"] += 1
            elif corr >= 0.8:
                bins["0.8-0.85"] += 1
            elif corr >= 0.75:
                bins["0.75-0.8"] += 1
            else:
                bins["0.7-0.75"] += 1

        return jsonify({
            "labels": list(bins.keys()),
            "data": list(bins.values()),
            "analysis_date": str(date_filter),
        })
    except Exception as e:
        logger.error(f"Error fetching correlation distribution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/charts/silhouette-history")
def chart_silhouette_history():
    """Get silhouette score history for charts."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT analysis_date, silhouette_score, n_clusters
            FROM price_correlation.analysis_runs
            WHERE silhouette_score IS NOT NULL
            ORDER BY analysis_date DESC
            LIMIT 30
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        dates = []
        scores = []
        clusters = []
        for row in reversed(rows):
            dates.append(row[0].isoformat() if row[0] else "")
            scores.append(round(row[1], 4) if row[1] else 0)
            clusters.append(row[2] or 0)

        return jsonify({
            "labels": dates,
            "silhouette": scores,
            "clusters": clusters,
        })
    except Exception as e:
        logger.error(f"Error fetching silhouette history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/history")
def get_pipeline_history():
    """Get historical pipeline runs."""
    limit = request.args.get("limit", 20, type=int)

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available", "runs": []}), 503

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                analysis_date,
                n_stocks_processed,
                n_clusters,
                n_noise,
                silhouette_score,
                clustering_method,
                execution_time_seconds,
                created_at
            FROM price_correlation.analysis_runs
            ORDER BY analysis_date DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        runs = []
        for row in rows:
            runs.append({
                "date": row[0].isoformat() if row[0] else "",
                "stocks": row[1] or 0,
                "clusters": row[2] or 0,
                "noise": row[3] or 0,
                "silhouette": round(row[4], 4) if row[4] else None,
                "method": row[5] or "unknown",
                "time_seconds": round(row[6], 1) if row[6] else None,
                "created_at": row[7].isoformat() if row[7] else "",
            })

        return jsonify({"runs": runs})
    except Exception as e:
        logger.error(f"Error fetching pipeline history: {e}")
        return jsonify({"error": str(e), "runs": []}), 500


# =============================================================================
# API Routes - Cache Control
# =============================================================================


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear Redis cache."""
    try:
        from .cache import clear_cache as do_clear
        result = do_clear()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/status")
def cache_status():
    """Get cache status."""
    try:
        from .cache import get_cache_stats
        return jsonify(get_cache_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/db/status")
def db_status():
    """Get database status."""
    try:
        from .db import get_db_stats
        return jsonify(get_db_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# API Routes - Step-by-Step Pipeline
# =============================================================================

# Track step execution state (per session)
step_execution_state = {
    "running": False,
    "session_id": None,
    "current_step": None,
    "logs": [],
    "error": None,
}


@app.route("/api/steps/sessions")
def list_pipeline_sessions():
    """List all pipeline sessions."""
    try:
        from .pipeline_state import list_sessions
        sessions = list_sessions()
        return jsonify({"sessions": sessions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/steps/session/<session_id>")
def get_session_status(session_id):
    """Get status of a specific session."""
    try:
        from .pipeline_state import get_state_manager
        manager = get_state_manager(session_id)
        status = manager.get_status()
        status["execution_running"] = (
            step_execution_state["running"] and
            step_execution_state["session_id"] == session_id
        )
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/steps/session/new", methods=["POST"])
def create_new_session():
    """Create a new pipeline session."""
    try:
        from .pipeline_state import get_state_manager
        data = request.get_json() or {}

        manager = get_state_manager()  # Creates new session
        manager.set_config(data.get("config", {}))

        return jsonify({
            "session_id": manager.session_id,
            "status": manager.get_status(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/steps/run/<step>", methods=["POST"])
def run_pipeline_step(step):
    """Run a single pipeline step."""
    from .pipeline_state import PIPELINE_STEPS

    if step not in PIPELINE_STEPS:
        return jsonify({"error": f"Invalid step: {step}"}), 400

    if step_execution_state["running"]:
        return jsonify({
            "error": "A step is already running",
            "current_step": step_execution_state["current_step"],
        }), 409

    data = request.get_json() or {}
    session_id = data.get("session_id")
    config = data.get("config", {})

    # Reset state
    step_execution_state["running"] = True
    step_execution_state["session_id"] = session_id
    step_execution_state["current_step"] = step
    step_execution_state["logs"] = []
    step_execution_state["error"] = None

    def run_step_background():
        try:
            from .pipeline_steps import run_single_step, StepConfig

            def progress_callback(msg):
                step_execution_state["logs"].append({
                    "time": datetime.now().isoformat(),
                    "message": msg,
                })

            step_config = StepConfig.from_dict(config) if config else StepConfig()
            result = run_single_step(step, step_config, session_id, progress_callback)

            step_execution_state["logs"].append({
                "time": datetime.now().isoformat(),
                "message": f"Step {step} completed successfully",
            })

        except Exception as e:
            logger.exception(f"Step {step} error: {e}")
            step_execution_state["error"] = str(e)
            step_execution_state["logs"].append({
                "time": datetime.now().isoformat(),
                "message": f"ERROR: {e}",
            })
        finally:
            step_execution_state["running"] = False
            step_execution_state["current_step"] = None

    thread = threading.Thread(target=run_step_background)
    thread.start()

    return jsonify({
        "message": f"Step {step} started",
        "session_id": session_id,
        "step": step,
    })


@app.route("/api/steps/run-from/<step>", methods=["POST"])
def run_pipeline_from_step(step):
    """Run pipeline from a specific step onwards."""
    from .pipeline_state import PIPELINE_STEPS

    if step not in PIPELINE_STEPS:
        return jsonify({"error": f"Invalid step: {step}"}), 400

    if step_execution_state["running"]:
        return jsonify({
            "error": "Pipeline is already running",
            "current_step": step_execution_state["current_step"],
        }), 409

    data = request.get_json() or {}
    session_id = data.get("session_id")
    config = data.get("config", {})

    step_execution_state["running"] = True
    step_execution_state["session_id"] = session_id
    step_execution_state["current_step"] = step
    step_execution_state["logs"] = []
    step_execution_state["error"] = None

    def run_from_background():
        try:
            from .pipeline_steps import run_from_step, StepConfig

            def progress_callback(msg):
                step_execution_state["logs"].append({
                    "time": datetime.now().isoformat(),
                    "message": msg,
                })
                # Update current step based on log message
                for s in PIPELINE_STEPS:
                    if f"Step" in msg and s in msg.lower():
                        step_execution_state["current_step"] = s

            step_config = StepConfig.from_dict(config) if config else StepConfig()
            result = run_from_step(step, step_config, session_id, progress_callback)

            step_execution_state["logs"].append({
                "time": datetime.now().isoformat(),
                "message": "Pipeline completed successfully",
            })

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            step_execution_state["error"] = str(e)
            step_execution_state["logs"].append({
                "time": datetime.now().isoformat(),
                "message": f"ERROR: {e}",
            })
        finally:
            step_execution_state["running"] = False
            step_execution_state["current_step"] = None

    thread = threading.Thread(target=run_from_background)
    thread.start()

    return jsonify({
        "message": f"Pipeline started from step {step}",
        "session_id": session_id,
        "step": step,
    })


@app.route("/api/steps/status")
def get_step_execution_status():
    """Get current step execution status."""
    return jsonify({
        "running": step_execution_state["running"],
        "session_id": step_execution_state["session_id"],
        "current_step": step_execution_state["current_step"],
        "logs": step_execution_state["logs"][-50:],  # Last 50 logs
        "error": step_execution_state["error"],
    })


@app.route("/api/steps/clear/<step>", methods=["POST"])
def clear_step_data(step):
    """Clear data for a step and all subsequent steps."""
    data = request.get_json() or {}
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    try:
        from .pipeline_state import get_state_manager
        manager = get_state_manager(session_id)
        manager.clear_step(step)
        return jsonify({
            "message": f"Cleared step {step} and subsequent steps",
            "status": manager.get_status(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/steps/data/<step>")
def get_step_data_preview(step):
    """Get preview of step data."""
    session_id = request.args.get("session_id")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    try:
        from .pipeline_state import get_state_manager
        manager = get_state_manager(session_id)

        if not manager.has_step_data(step):
            return jsonify({"error": f"No data for step {step}"}), 404

        data = manager.load_step_data(step)

        # Create preview based on step type
        preview = {"step": step, "has_data": True}

        if step == "universe":
            preview["count"] = len(data)
            preview["sample"] = data[:20]

        elif step == "prices":
            preview["tickers"] = data.shape[1]
            preview["days"] = data.shape[0]
            preview["columns"] = list(data.columns[:20])
            preview["date_range"] = {
                "start": str(data.index[0]),
                "end": str(data.index[-1]),
            }

        elif step == "preprocess":
            preview["tickers"] = data.shape[1]
            preview["days"] = data.shape[0]
            preview["columns"] = list(data.columns[:20])

        elif step == "correlation":
            preview["matrix_shape"] = list(data["corr_matrix"].shape)
            preview["tickers_count"] = len(data["tickers"])
            preview["sample_tickers"] = data["tickers"][:20]

        elif step == "clustering":
            preview["n_clusters"] = data["stats"]["n_clusters"]
            preview["n_noise"] = data["stats"]["n_noise"]
            preview["silhouette"] = data["silhouette"]
            preview["method"] = data["method_info"]
            preview["sample_assignments"] = [
                {"ticker": t, "cluster": int(l)}
                for t, l in zip(data["tickers"][:20], data["labels"][:20])
            ]

        elif step == "export":
            preview["files"] = list(data.get("output_files", {}).keys())
            preview["execution_time"] = data.get("execution_time")

        return jsonify(preview)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Server Entry Point
# =============================================================================


def create_app():
    """Create and configure the Flask app."""
    return app


def run_server():
    """Run the web server."""
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", 5000))
    debug = os.environ.get("WEB_DEBUG", "false").lower() == "true"

    logger.info(f"Starting web server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
