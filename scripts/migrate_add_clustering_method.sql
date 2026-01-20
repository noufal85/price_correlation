-- Migration: Add clustering_method column to equity_clusters
-- This allows storing results from multiple clustering methods per analysis date

-- Step 1: Add clustering_method column with default value
ALTER TABLE price_correlation.equity_clusters
ADD COLUMN IF NOT EXISTS clustering_method VARCHAR(50) DEFAULT 'hierarchical';

-- Step 2: Drop the old primary key
ALTER TABLE price_correlation.equity_clusters
DROP CONSTRAINT IF EXISTS equity_clusters_pkey;

-- Step 3: Create new primary key including clustering_method
ALTER TABLE price_correlation.equity_clusters
ADD PRIMARY KEY (analysis_date, ticker, clustering_method);

-- Step 4: Add index for method-based queries
CREATE INDEX IF NOT EXISTS idx_equity_clusters_method
    ON price_correlation.equity_clusters (clustering_method, analysis_date DESC);

-- Step 5: Update analysis_runs to support multiple methods per date
-- First, drop the old primary key on analysis_runs
ALTER TABLE price_correlation.analysis_runs
DROP CONSTRAINT IF EXISTS analysis_runs_pkey;

-- Add primary key with clustering_method
ALTER TABLE price_correlation.analysis_runs
ADD PRIMARY KEY (analysis_date, clustering_method);

-- Verify changes
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'price_correlation'
AND table_name = 'equity_clusters'
ORDER BY ordinal_position;
