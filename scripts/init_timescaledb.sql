-- TimescaleDB Schema for Stock Clustering Results
-- Run this script to initialize the database tables

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- equity_clusters: Cluster assignments per analysis run
-- ============================================================================
CREATE TABLE IF NOT EXISTS equity_clusters (
    analysis_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    cluster_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (analysis_date, ticker)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('equity_clusters', 'analysis_date',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- pair_correlations: Highly correlated pairs per analysis run
-- ============================================================================
CREATE TABLE IF NOT EXISTS pair_correlations (
    analysis_date DATE NOT NULL,
    ticker_a VARCHAR(20) NOT NULL,
    ticker_b VARCHAR(20) NOT NULL,
    correlation DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (analysis_date, ticker_a, ticker_b)
);

-- Convert to hypertable
SELECT create_hypertable('pair_correlations', 'analysis_date',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- analysis_runs: Pipeline execution metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_date DATE PRIMARY KEY,
    n_stocks_processed INTEGER NOT NULL,
    n_clusters INTEGER NOT NULL,
    n_noise INTEGER DEFAULT 0,
    silhouette_score DOUBLE PRECISION,
    clustering_method VARCHAR(50),
    execution_time_seconds DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Indexes for common query patterns
-- ============================================================================

-- Find cluster history for a specific ticker
CREATE INDEX IF NOT EXISTS idx_equity_clusters_ticker
    ON equity_clusters (ticker, analysis_date DESC);

-- Find most correlated pairs
CREATE INDEX IF NOT EXISTS idx_pair_correlations_corr
    ON pair_correlations (correlation DESC);

-- Find correlation history for a specific ticker
CREATE INDEX IF NOT EXISTS idx_pair_correlations_ticker
    ON pair_correlations (ticker_a, analysis_date DESC);

-- ============================================================================
-- Sample Queries (for reference)
-- ============================================================================

-- Latest cluster for a specific ticker:
-- SELECT cluster_id FROM equity_clusters
-- WHERE ticker = 'AAPL'
-- ORDER BY analysis_date DESC LIMIT 1;

-- All stocks in same cluster as AAPL (latest run):
-- WITH latest AS (
--     SELECT cluster_id, analysis_date FROM equity_clusters
--     WHERE ticker = 'AAPL' ORDER BY analysis_date DESC LIMIT 1
-- )
-- SELECT ec.ticker FROM equity_clusters ec
-- JOIN latest l ON ec.cluster_id = l.cluster_id AND ec.analysis_date = l.analysis_date;

-- Top correlated pairs (latest run):
-- SELECT ticker_a, ticker_b, correlation
-- FROM pair_correlations
-- WHERE analysis_date = (SELECT MAX(analysis_date) FROM analysis_runs)
-- ORDER BY correlation DESC LIMIT 10;

-- Cluster history for a ticker:
-- SELECT analysis_date, cluster_id FROM equity_clusters
-- WHERE ticker = 'AAPL' ORDER BY analysis_date;

-- Analysis run summary:
-- SELECT * FROM analysis_runs ORDER BY analysis_date DESC LIMIT 10;
