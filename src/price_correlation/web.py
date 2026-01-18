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
    use_sample = data.get("use_sample", True)  # Default to sample mode for web
    sample_size = data.get("sample_size", 50)
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
                use_sample=use_sample,
                sample_size=sample_size,
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
            FROM analysis_runs
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

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        if analysis_date:
            date_filter = analysis_date
        else:
            cursor.execute("SELECT MAX(analysis_date) FROM equity_clusters")
            result = cursor.fetchone()
            date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker, cluster_id
            FROM equity_clusters
            WHERE analysis_date = %s
            ORDER BY cluster_id, ticker
        """, (date_filter,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Group by cluster
        clusters = {}
        for ticker, cluster_id in rows:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(ticker)

        return jsonify({
            "analysis_date": str(date_filter),
            "clusters": clusters,
            "total_stocks": len(rows),
            "n_clusters": len([k for k in clusters if k != -1]),
            "n_noise": len(clusters.get(-1, [])),
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
            cursor.execute("SELECT MAX(analysis_date) FROM pair_correlations")
            result = cursor.fetchone()
            date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker_a, ticker_b, correlation
            FROM pair_correlations
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
            FROM equity_clusters
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
    """Get correlations for a specific stock."""
    limit = int(request.args.get("limit", 20))

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        # Get latest date
        cursor.execute("SELECT MAX(analysis_date) FROM pair_correlations")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker_a, ticker_b, correlation
            FROM pair_correlations
            WHERE analysis_date = %s AND (ticker_a = %s OR ticker_b = %s)
            ORDER BY correlation DESC
            LIMIT %s
        """, (date_filter, ticker.upper(), ticker.upper(), limit))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        pairs = []
        for row in rows:
            other = row[1] if row[0] == ticker.upper() else row[0]
            pairs.append({
                "ticker": other,
                "correlation": round(row[2], 4),
            })

        return jsonify({
            "ticker": ticker.upper(),
            "analysis_date": str(date_filter),
            "correlations": pairs,
        })
    except Exception as e:
        logger.error(f"Error fetching stock correlations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cluster/<int:cluster_id>/stocks")
def get_cluster_stocks(cluster_id):
    """Get all stocks in a cluster."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        # Get latest date
        cursor.execute("SELECT MAX(analysis_date) FROM equity_clusters")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT ticker
            FROM equity_clusters
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
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 503

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT MAX(analysis_date) FROM equity_clusters")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        cursor.execute("""
            SELECT cluster_id, COUNT(*) as size
            FROM equity_clusters
            WHERE analysis_date = %s
            GROUP BY cluster_id
            ORDER BY size DESC
        """, (date_filter,))
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

        cursor.execute("SELECT MAX(analysis_date) FROM pair_correlations")
        result = cursor.fetchone()
        date_filter = result[0] if result[0] else date.today()

        # Get correlations and bin them
        cursor.execute("""
            SELECT correlation FROM pair_correlations
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
            FROM analysis_runs
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
