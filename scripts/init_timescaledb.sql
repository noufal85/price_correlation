-- TimescaleDB Schema for Stock Clustering Results
-- Run this script to initialize the database tables
-- Schema: price_correlation

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create dedicated schema
CREATE SCHEMA IF NOT EXISTS price_correlation;

-- ============================================================================
-- price_correlation.equity_clusters: Cluster assignments per analysis run
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_correlation.equity_clusters (
    analysis_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    cluster_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (analysis_date, ticker)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('price_correlation.equity_clusters', 'analysis_date',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- price_correlation.pair_correlations: Highly correlated pairs per analysis run
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_correlation.pair_correlations (
    analysis_date DATE NOT NULL,
    ticker_a VARCHAR(20) NOT NULL,
    ticker_b VARCHAR(20) NOT NULL,
    correlation DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (analysis_date, ticker_a, ticker_b)
);

-- Convert to hypertable
SELECT create_hypertable('price_correlation.pair_correlations', 'analysis_date',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- price_correlation.analysis_runs: Pipeline execution metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_correlation.analysis_runs (
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
-- price_correlation.exclusions: Stock exclusion tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_correlation.exclusions (
    symbol VARCHAR(20) PRIMARY KEY,
    exclusion_type VARCHAR(50) NOT NULL,  -- 'etf', 'index', 'share_class', 'manual'
    reason TEXT,
    added_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Indexes for common query patterns
-- ============================================================================

-- Find cluster history for a specific ticker
CREATE INDEX IF NOT EXISTS idx_equity_clusters_ticker
    ON price_correlation.equity_clusters (ticker, analysis_date DESC);

-- Find most correlated pairs
CREATE INDEX IF NOT EXISTS idx_pair_correlations_corr
    ON price_correlation.pair_correlations (correlation DESC);

-- Find correlation history for a specific ticker
CREATE INDEX IF NOT EXISTS idx_pair_correlations_ticker
    ON price_correlation.pair_correlations (ticker_a, analysis_date DESC);

-- Exclusion type index
CREATE INDEX IF NOT EXISTS idx_exclusions_type
    ON price_correlation.exclusions (exclusion_type);

-- ============================================================================
-- Sample Queries (for reference)
-- ============================================================================

-- Latest cluster for a specific ticker:
-- SELECT cluster_id FROM price_correlation.equity_clusters
-- WHERE ticker = 'AAPL'
-- ORDER BY analysis_date DESC LIMIT 1;

-- All stocks in same cluster as AAPL (latest run):
-- WITH latest AS (
--     SELECT cluster_id, analysis_date FROM price_correlation.equity_clusters
--     WHERE ticker = 'AAPL' ORDER BY analysis_date DESC LIMIT 1
-- )
-- SELECT ec.ticker FROM price_correlation.equity_clusters ec
-- JOIN latest l ON ec.cluster_id = l.cluster_id AND ec.analysis_date = l.analysis_date;

-- Top correlated pairs (latest run):
-- SELECT ticker_a, ticker_b, correlation
-- FROM price_correlation.pair_correlations
-- WHERE analysis_date = (SELECT MAX(analysis_date) FROM price_correlation.analysis_runs)
-- ORDER BY correlation DESC LIMIT 10;

-- Cluster history for a ticker:
-- SELECT analysis_date, cluster_id FROM price_correlation.equity_clusters
-- WHERE ticker = 'AAPL' ORDER BY analysis_date;

-- Analysis run summary:
-- SELECT * FROM price_correlation.analysis_runs ORDER BY analysis_date DESC LIMIT 10;

-- List all excluded symbols:
-- SELECT * FROM price_correlation.exclusions ORDER BY exclusion_type, symbol;
