# **Quantitative Framework for Unsupervised Clustering and Co-movement Analysis of the US Equity Universe**

## **1\. Introduction and Problem Scope**

The objective of identifying equities that "move together" within the expansive universe of the NASDAQ and NYSE is a foundational problem in quantitative finance, touching upon risk management, portfolio construction, and statistical arbitrage. The request to analyze the "entire universe" over an 18-month horizon transforms a standard data analysis task into a significant engineering and mathematical challenge. This report delineates a comprehensive architecture for a system capable of ingesting high-dimensional financial time series, applying rigorous statistical and machine learning techniques to cluster these assets, and exporting the results in a format optimized for high-performance database querying.

The core complexity lies in the ambiguity of "moving together." In financial markets, this phrase can denote short-term directional correlation, where assets react similarly to immediate market stimuli, or long-term cointegration, where assets share a fundamental economic link that binds their prices over time. Distinguishing between these phenomena is critical, as they imply radically different trading strategies and mathematical treatments. Furthermore, the application of "best-in-class" machine learning requires navigating the trade-offs between computational tractability and theoretical purity. Algorithms that excel in low-dimensional spaces, such as standard K-Means, often falter when applied to the noisy, non-stationary, and high-dimensional reality of thousands of stock price tickers.

This document serves as a blueprint for a robust analytical engine. It proceeds from the ground up: first defining the universe and addressing the silent killer of quantitative backtests—survivorship bias; moving to the rigorous preprocessing required to normalize disparate asset prices; evaluating distance metrics that respect the temporal structure of markets; selecting clustering algorithms that can handle noise and hierarchy; and finally, specifying the technical architecture for persistence and retrieval. By synthesizing insights from recent academic literature on time-series clustering with practical constraints of data engineering, this report aims to provide an exhaustive guide to constructing an institutional-grade clustering system.

## ---

**2\. Universe Definition and Data Acquisition Strategy**

The validity of any cluster analysis is strictly bounded by the quality and completeness of the input data. The user requirement to cover the "entire universe" of NASDAQ and NYSE stocks necessitates a sophisticated approach to data acquisition that goes beyond simple static lists of current tickers.

### **2.1 The Challenge of the Historical Universe**

A naive approach to defining the stock universe involves downloading a list of currently active tickers and retrieving their price history for the last 18 months. This method is fundamentally flawed due to survivorship bias. Markets are dynamic ecosystems where companies are continuously listed, delisted, acquired, or bankrupted. An analysis based solely on today's survivors implicitly filters out the "failures," thereby skewing clustering results towards assets that have successfully navigated the past 18 months. This artificially inflates stability metrics and can lead to overfitting, as the algorithm is trained on a dataset that does not represent the actual opportunity set available to a trader at the beginning of the period.

To mitigate this, the system must reconstruct the universe as it existed at the start of the 18-month window and track changes dynamically. This requires a "point-in-time" or listing status database. The universe for this project is estimated to contain between 5,000 and 7,000 liquid equities, expanding to over 10,000 if micro-cap and illiquid over-the-counter (OTC) securities are included.

### **2.2 Data Source Evaluation and Selection**

Reliable data ingestion is the first engineering hurdle. Several providers offer APIs capable of supporting this scale of analysis, each with distinct advantages regarding the "full universe" requirement.

**Table 1: Comparative Analysis of Data Providers for Full Universe Clustering**

| Feature | Alpha Vantage | Databento | Finnhub | Marketstack |
| :---- | :---- | :---- | :---- | :---- |
| **Listing Status** | Dedicated LISTING\_STATUS endpoint returning active & delisted CSVs.1 | Comprehensive metadata and symbology via definition schema.3 | Fundamental profile and status data for 30k+ companies.4 | Extensive ticker coverage including global exchanges.5 |
| **History Depth** | 20+ years, supports adjusted prices. | Full market replay, nanosecond precision (TotalView-ITCH).3 | 30+ years, institutional grade.4 | 30+ years EOD history.5 |
| **Adjustment** | Pre-calculated Adjusted Close columns. | Raw feed data; adjustments typically applied client-side or via specific datasets. | Comprehensive dividends and splits data. | Adjusted data available via EOD endpoint. |
| **Data Format** | JSON & CSV (optimized for bulk).6 | Binary (DBN), convert to CSV/JSON. Optimized for size.3 | JSON REST API. | JSON REST API. |
| **Best Use** | Universe maintenance (Active/Delisted) & Daily Bars. | High-frequency analysis & microstructure accuracy. | Fundamental analysis integration. | Broad global coverage. |

**Alpha Vantage** emerges as a particularly strong candidate for the specific task of universe definition due to its LISTING\_STATUS endpoint. This API returns a flat CSV file containing the symbol, name, exchange, asset type, IPO date, delisting date, and status (Active/Delisted) for all requested securities.1 By querying this endpoint with a date parameter set to 18 months ago, the system can generate a precise list of every asset that was trading at that time, ensuring the inclusion of delisted stocks in the clustering analysis to paint a true picture of market correlations.7

**Databento** offers a different but equally powerful value proposition. It provides data derived directly from exchange feeds like Nasdaq TotalView-ITCH.3 While this level of granularity (tick-by-tick) might be excessive for daily clustering, it guarantees that the Open, High, Low, and Close (OHLC) bars are constructed with absolute precision, free from the aggregation errors sometimes found in aggregated feeds. For a project aiming for the "most reliable" machine learning, the integrity of the input vector is paramount.

### **2.3 Handling Corporate Actions**

Stock prices are not continuous physical measurements; they are financial values subject to arbitrary discontinuities caused by corporate actions. The most significant of these for clustering are stock splits and dividends.

Consider a stock trading at $100 that undergoes a 2-for-1 split. The price drops to $50 overnight. A standard clustering algorithm using Euclidean distance would interpret this as a massive change in value, likely clustering this stock with assets that crashed 50%. In reality, the economic value held by the investor remains unchanged. Therefore, all analysis must be conducted on **Adjusted Close** prices.

The adjustment mechanism typically involves calculating a "Split Coefficient." Alpha Vantage and similar providers include this in their daily time series responses.8 The system must ingest the raw price and the adjustment factor, systematically recalculating the historical series to be backward-compatible with the current price level. This ensures that the "movement" detected by the clustering algorithm reflects market sentiment and valuation changes, not administrative adjustments.

### **2.4 Data Cleaning and Alignment**

The "entire universe" includes thousands of stocks with varying liquidity profiles. A dense distance matrix calculation requires that all time series are perfectly aligned on the same date index.

* **Missing Data (NaNs):** Illiquid stocks may not trade every day. The dataset will inevitably contain gaps.  
* **Imputation Strategy:** Simple interpolation (e.g., linear fill) introduces look-ahead bias, as it uses future data to fill past gaps. The standard for financial time series is **Forward Filling** (propagating the last known close price forward), which mimics the portfolio value of an investor holding the illiquid asset.9  
* **Filtering:** Assets with insufficient history (e.g., recent IPOs with less than 18 months of data) or excessive missingness (e.g., \>10% of trading days) should be excluded from the primary clustering. Their short vectors would distort distance metrics like Euclidean distance or DTW, which effectively sum differences over time; a shorter series would artificially appear "closer" to others simply because it accumulates less total distance.9

## ---

**3\. Mathematical Foundations: Correlation, Cointegration, and Stationarity**

The user's request to find stocks that are "correlated and moving together" touches upon two distinct concepts in financial econometrics. While often used interchangeably in casual parlance, **Correlation** and **Cointegration** represent fundamentally different relationships with distinct implications for trading and risk management. A nuanced understanding of these concepts is essential for selecting the appropriate machine learning feature set.

### **3.1 Correlation: Short-Term Linear Dependence**

Correlation measures the degree to which two variables move in relation to each other. In finance, this is almost exclusively calculated on **returns**, not raw prices.

* Pearson Correlation ($\\rho$): The most common metric, measuring linear dependence.

  $$\\rho\_{X,Y} \= \\frac{\\text{cov}(X,Y)}{\\sigma\_X \\sigma\_Y}$$

  A correlation of \+1 implies that if Stock A goes up, Stock B goes up (proportional to their volatilities).  
* **Utility:** Correlation is the primary tool for **Risk Management** and **Portfolio Diversification**. If an investor wants to reduce variance, they cluster stocks by correlation and pick assets from different clusters (e.g., Technology vs. Utilities).  
* **Limitation:** High correlation does not imply prices stay close together. Two stocks can have a daily return correlation of 0.99, yet if Stock A rises 1.1% every day while Stock B rises 1.0%, their prices will diverge exponentially over time. They "move together" directionally, but they do not maintain a value equilibrium.11

### **3.2 Cointegration: Long-Term Equilibrium**

Cointegration is a stricter, more powerful property for active trading strategies like statistical arbitrage. It describes a relationship where two non-stationary time series (prices that wander randomly) share a common stochastic drift.

* **Definition:** Two series $X\_t$ and $Y\_t$ are cointegrated if they are individually non-stationary ($I(1)$) but there exists a linear combination $Z\_t \= X\_t \- \\beta Y\_t$ that is stationary ($I(0)$).  
* **The Drunk and the Dog:** The classic analogy involves a drunk man (random walk) walking a dog (another random walk). Because they are connected by a leash, they cannot drift too far apart. The distance between them (the spread) is mean-reverting. Even if the drunk wanders 10 blocks north, the dog must follow.13  
* **Trading Implication:** If a pair is cointegrated, the spread *must* revert to its mean. This allows a trader to open a "Pairs Trade": short the outperforming asset and long the underperforming one when the spread widens beyond a threshold (e.g., 2 standard deviations), confident that they will converge.14

### **3.3 Stationarity and the Necessity of Transformations**

Machine learning algorithms, particularly those based on distance metrics, assume that the statistical properties of the data (mean, variance) are constant. Stock prices violate this; they are **Non-Stationary**.

* **Spurious Correlation:** Calculating Pearson correlation on raw price levels (e.g., apple price vs. orange price over 10 years) often yields high correlation simply because both are driven by inflation or economic growth. This is a statistical mirage.  
* The Log-Return Transformation: To render the data stationary for clustering, we transform prices $P\_t$ into Log-Returns $r\_t$:

  $$r\_t \= \\ln(P\_t) \- \\ln(P\_{t-1})$$

  This removes the trend component and normalizes the data, allowing the clustering algorithm to focus on the pattern of daily moves rather than the trend magnitude.15

### **3.4 The Hybrid Approach for the "Best" System**

While the user asks for "moving together," finding cointegrated pairs across the entire universe is computationally prohibitive. A full cointegration test (like the Engle-Granger test) is computationally expensive. Running it on all $12.5$ million pairs ($5000 \\times 4999 / 2$) is inefficient.

* **Strategy:** The optimal architecture uses **Clustering (on Correlation/Distance)** as a dimensionality reduction step.  
  1. Use unsupervised learning (DBSCAN/K-Means) on log-returns to find clusters of 50-100 stocks that are highly correlated.  
  2. Perform rigorous Cointegration Tests (Engle-Granger or Johansen) only on the pairs within these clusters.  
     This hybrid approach leverages the speed of clustering to filter the universe, then applies the rigor of cointegration to identify actionable arbitrage opportunities.

## ---

**4\. Distance Metrics and Feature Engineering**

Unsupervised learning algorithms are engines that process "distance." The choice of how to measure the distance between Stock A and Stock B determines the entire output of the project.

### **4.1 Euclidean Distance vs. Correlation Distance**

* Euclidean Distance: Measures the straight-line distance between two points in high-dimensional space.

  $$d(x,y) \= \\sqrt{\\sum (x\_i \- y\_i)^2}$$

  For time series, this compares the return on Day 1 for Stock A with the return on Day 1 for Stock B.  
  * *Preprocessing Requirement:* Euclidean distance is sensitive to scale. A stock moving 1% vs. 2% will be "distant." Z-score normalization (subtracting mean, dividing by standard deviation) is mandatory.  
  * *Equivalence:* It has been mathematically shown that for z-normalized vectors, the squared Euclidean distance is proportional to the Pearson Correlation distance. Specifically, $d\_E^2 \= 2m(1 \- \\rho)$, where $m$ is the series length. This validates the use of Euclidean-based algorithms (like K-Means) as effectively clustering on correlation.16

### **4.2 Dynamic Time Warping (DTW)**

Standard metrics like Euclidean distance are brittle to time shifts. If Stock A reacts to news on Monday, and Stock B reacts to the same news on Tuesday, Euclidean distance penalizes this mismatch heavily.

* **Mechanism:** DTW is a non-linear algorithm that finds an optimal alignment between two sequences by "warping" the time axis. It constructs a cost matrix and finds a path that minimizes the total distance, allowing a point at $t$ in series A to map to $t+1$ in series B.18  
* **Relevance:** This is powerful for identifying **Lead-Lag Relationships**. For example, identifying that a semiconductor equipment manufacturer's stock consistently moves 3 days before a chip maker's stock.  
* **Computational Cost:** Classical DTW is $O(T^2)$, where $T$ is the time series length. For 378 days, this is manageable per pair, but expensive for $N=5000$ stocks.  
* **Fast Implementations:** To use DTW at scale, constrained versions like the **Sakoe-Chuba band** (limiting the warp window) or approximations like **FastDTW** (linear time complexity) are required. Libraries such as tslearn and dtaidistance offer optimized C-based implementations for Python.19

### **4.3 Dimensionality Reduction: PCA vs. t-SNE**

Before feeding 5,000 vectors of 378 dimensions into a clustering algorithm, it is often beneficial to reduce the dimensionality to remove noise and isolate the signal.

* **PCA (Principal Component Analysis):** A linear transformation that projects data onto orthogonal axes of maximum variance. In finance, the first Principal Component almost always represents the "Market Factor" (Beta)—the tendency of all stocks to move with the S\&P 500\. By removing this first component and clustering on the residuals, the system can isolate **idiosyncratic** correlations—stocks that move together *independent* of the broad market.22  
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique excellent for visualization. It maps high-dimensional data to 2D or 3D while preserving local neighborhoods. While not typically used for the clustering step itself (due to lack of parametric mapping for new data), it is the "best" tool for **Visual Validation**. Plotting the final 5,000 stocks in 2D using t-SNE allows the researcher to visually inspect if the clusters are well-separated "islands" or a messy continuum.23

## ---

**5\. Selection and Implementation of Clustering Algorithms**

The user requests the "best machine learning." In this domain, "best" is defined by the algorithm's ability to handle the specific characteristics of financial data: high noise, varying cluster densities, and hierarchical structure.

### **5.1 K-Means and TimeSeriesKMeans**

K-Means is the standard baseline. It partitions data into $k$ clusters by minimizing the variance within each cluster.

* **Pros:** Extremely fast and scalable to large $N$.  
* **Cons:** It forces *every* data point into a cluster. In the stock market, many stocks are "noise"—they do not belong to any distinct group. K-Means will arbitrarily assign these outliers to the nearest cluster, potentially degrading the cluster's quality (e.g., assigning a random biotech stock to a Utility cluster).  
* **TimeSeriesKMeans:** The tslearn library provides a specialized version that can use DTW as the distance metric. It computes "DTW Barycenters" (averages) instead of arithmetic means for centroids. This allows the cluster center to represent the *shape* of the members, preserving dynamic patterns that standard K-Means would smooth out.20

### **5.2 DBSCAN (Density-Based Spatial Clustering)**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is widely regarded as superior for pairs trading and financial clustering because of its handling of noise.

* **Mechanism:** It groups points that are closely packed together (high density). Points that lie in low-density regions are labeled as noise (-1).  
* **Why it is "Best":** For a trader, it is better to identify 500 high-confidence pairs and ignore the rest of the universe than to force all 5,000 stocks into weak groups. DBSCAN naturally filters out the "junk," leaving only the tightest, most actionable clusters.  
* **Parameter Tuning:** It requires two parameters: eps (the maximum distance between two samples for one to be considered as in the neighborhood of the other) and min\_samples. The eps parameter is crucial; if set too low, no clusters form. If too high, everything merges. The "Elbow Method" using a k-distance graph is the standard technique to find the optimal eps.

### **5.3 Hierarchical Clustering and Hierarchical Risk Parity (HRP)**

Financial markets are not flat; they are hierarchical (Market $\\to$ Sector $\\to$ Industry $\\to$ Sub-Industry). Hierarchical clustering builds a tree (dendrogram) that captures this nested structure.

* **Hierarchical Risk Parity (HRP):** This is a specific machine learning application in portfolio management that has gained significant traction. Proposed by Marcos Lopez de Prado, HRP uses hierarchical clustering to address the instability of quadratic optimizers (like Markowitz Mean-Variance).  
  * **Step 1: Tree Clustering:** Use hierarchical clustering (e.g., Single Linkage) to group assets.  
  * **Step 2: Quasi-Diagonalization:** Reorder the covariance matrix rows/cols based on the dendrogram leaves. This places similar assets close to each other on the diagonal.  
  * **Step 3: Recursive Bisection:** Allocate capital top-down through the tree.  
* **Relevance:** While HRP is a portfolio allocation technique, the *clustering phase* of HRP is highly relevant for this project. It demonstrates that hierarchical methods often yield more robust economic groupings than flat methods like K-Means. The dendrogram structure allows the user to "cut" the tree at different levels to get broader or more granular clusters depending on the desired correlation threshold.

### **5.4 Validation Metrics**

How do we quantify "best"?

* **Silhouette Score:** Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Ranges from \-1 to \+1. A high score implies dense, well-separated clusters.25  
* **Cophenetic Correlation:** For hierarchical clustering, this measures how faithfully the dendrogram preserves the pairwise distances between the original data points. A high cophenetic correlation indicates the tree structure is a valid representation of the data topology.27  
* **Economic Validation:** The ultimate test is mapping the clusters against GICS (Global Industry Classification Standard) codes. If the algorithm groups "Exxon," "Chevron," and "Schlumberger" together without being told they are energy stocks, the model has successfully learned economic reality from price action.

## ---

**6\. System Architecture and Engineering**

Designing a system to process 18 months of daily data for 5,000+ stocks involves significant computational considerations.

### **6.1 Technology Stack**

* **Language:** **Python** is the undisputed industry standard for quantitative finance due to its rich ecosystem.  
* **Data Ingestion:** yfinance (for prototyping) or databento / alpha\_vantage libraries for production reliability.  
* **Core Processing:** pandas and numpy for vectorization. scipy.spatial for optimized distance calculations.  
* **Machine Learning:**  
  * scikit-learn: For PCA, DBSCAN, Agglomerative Clustering.  
  * tslearn: For DTW-based K-Means.20  
  * statsmodels: For stationarity (ADF) and cointegration (Engle-Granger/Johansen) tests.29  
  * fastdtw: For optimized Dynamic Time Warping.21

### **6.2 Computational Constraints and Optimization**

The distance matrix for 5,000 stocks contains $12.5$ million unique pairs ($N(N-1)/2$). Storing this as 64-bit floats requires approximately 100MB of RAM, which is trivial. However, *computing* it is CPU-bound.

* **Parallelization:** Calculating the distance matrix must be parallelized. Python's Global Interpreter Lock (GIL) limits threads, so multiprocessing or joblib should be used to spawn separate processes.  
* **Vectorization:** scipy.spatial.distance.pdist is implemented in C. It is orders of magnitude faster than a Python loop. For correlation distance, this function is the optimal choice.  
* **Memory Management:** If the universe expands to 20,000+ global stocks, the matrix grows to 400 million floats (3GB+). While still manageable on servers, numpy.memmap can be used to store the matrix on disk and access it as if it were in RAM, preventing memory overflow.

## ---

**7\. Data Persistence: Schema and Database Integration**

The final requirement is to export the results to a file for a database. The choice of format is critical for the downstream performance of queries (e.g., "Find all stocks correlated \> 0.9 with AAPL").

### **7.1 File Format: Apache Parquet vs. CSV**

While CSV is human-readable, it is inefficient for large datasets. **Apache Parquet** is the recommended format for this project.

* **Columnar Storage:** Parquet stores data by column, not by row. If an analyst wants to query just the Cluster\_ID column for 5,000 stocks, Parquet reads only that specific data block, skipping the price history. CSV would require scanning every byte of the file.  
* **Compression:** Parquet uses efficient encoding (Run-Length Encoding, Snappy). For repetitive data (like Cluster IDs or dates), it achieves massive compression ratios, often 10x smaller than CSV.  
* **Schema Enforcement:** Parquet stores metadata (data types). This prevents the common error where a ticker like "0050" is read as the integer 50, or dates are misparsed.31

### **7.2 Database Schema Design**

The output should be structured into normalized tables to support efficient SQL querying.

**Table 2: equity\_clusters.parquet Schema**

| Column Name | Data Type | Description |
| :---- | :---- | :---- |
| analysis\_date | Date | The date the clustering was performed (for historical versioning) |
| ticker | String | The stock symbol (e.g., "AAPL") |
| cluster\_id | Integer | The ID assigned by the algorithm (e.g., 5). \-1 for Noise in DBSCAN. |
| cluster\_confidence | Float | (Optional) Distance to cluster centroid or Silhouette score for this point. |
| listing\_status | String | "Active" or "Delisted" (from the universe definition step) |
| sector | String | (Optional) GICS sector for comparison/validation. |

Table 3: pair\_correlations.parquet Schema (Sparse Representation)  
Storing the full $5000 \\times 5000$ matrix is wasteful. Store only significant pairs.

| Column Name | Data Type | Description |
| :---- | :---- | :---- |
| ticker\_a | String | Symbol of first asset |
| ticker\_b | String | Symbol of second asset |
| correlation | Float | Pearson correlation coefficient (e.g., 0.95) |
| cointegration\_p\_value | Float | Result of Engle-Granger test (if performed) |
| metric\_type | String | "Pearson", "DTW", etc. |

### **7.3 Database Integration**

* **DuckDB:** An embedded OLAP database (analagous to SQLite for analytics). It can query Parquet files directly without an import step. This is ideal for a local research environment.  
  SQL  
  \-- Example DuckDB Query  
  SELECT ticker, cluster\_id   
  FROM 'equity\_clusters.parquet'   
  WHERE cluster\_id \= (SELECT cluster\_id FROM 'equity\_clusters.parquet' WHERE ticker \= 'AAPL');

* **TimescaleDB:** If the system requires a persistent server to power a real-time dashboard, **TimescaleDB** (a PostgreSQL extension) is the industry standard. It handles time-series data partitioning automatically, allowing for efficient queries over the 18-month price history joined with the cluster results.33

## ---

**8\. Conclusion and Strategic Roadmap**

To fulfill the user's request for the "most reliable and best machine learning" to cluster the US equity universe, this report recommends a hierarchical, hybrid approach that addresses the nuances of financial data.

1. **Data Engineering:** Construct the universe using **Alpha Vantage's LISTING\_STATUS** to eliminate survivorship bias. Ingest **Adjusted Close** prices to neutralize corporate actions.  
2. **Preprocessing:** Transform non-stationary prices into **Log-Returns** and apply **Z-score normalization**.  
3. **Dimensionality Reduction:** Utilize **PCA** to remove the dominant market mode, isolating idiosyncratic correlations.  
4. **Clustering:** Deploy **DBSCAN** with **Pearson Correlation Distance** as the primary engine. Its ability to filter noise (unclustered stocks) provides a higher-quality signal than K-Means.  
5. **Refinement:** For the highest value "moving together" signals (pairs trading), perform **Cointegration Tests (Engle-Granger)** specifically on the pairs identified within the dense DBSCAN clusters.  
6. **Persistence:** Export the validated clusters and correlations to **Apache Parquet**, leveraging **DuckDB** for high-performance, serverless SQL analysis.

This architecture moves beyond simple data analysis, establishing a rigorous, institutional-grade quantitative workflow capable of revealing the deep structure of the equity market.

#### **Works cited**

1. API Documentation | Alpha Vantage, accessed January 17, 2026, [https://www.alphavantage.co/documentation/](https://www.alphavantage.co/documentation/)  
2. End-of-Day Historical Stock Market Data API \- EODHD, accessed January 17, 2026, [https://eodhd.com/financial-apis/api-for-historical-data-and-volumes](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)  
3. Stock Market API for Real-Time & Historical Data | $125 Free Credit \- Databento, accessed January 17, 2026, [https://databento.com/stocks](https://databento.com/stocks)  
4. Finnhub Stock APIs \- Real-time stock prices, Company fundamentals, Estimates, and Alternative data., accessed January 17, 2026, [https://finnhub.io/](https://finnhub.io/)  
5. Free Stock Market Data API for Real-Time & Historical Data, accessed January 17, 2026, [https://marketstack.com/](https://marketstack.com/)  
6. Alpha Vantage: Free Stock APIs in JSON & Excel, accessed January 17, 2026, [https://www.alphavantage.co/](https://www.alphavantage.co/)  
7. Alpha Vantage Delisted Stocks \- Macroption, accessed January 17, 2026, [https://www.macroption.com/alpha-vantage-delisted-stocks/](https://www.macroption.com/alpha-vantage-delisted-stocks/)  
8. Download historical data using Alpha Vantage, accessed January 17, 2026, [https://cafim.sssup.it/\~giulio/other/alpha\_vantage/index.html](https://cafim.sssup.it/~giulio/other/alpha_vantage/index.html)  
9. aaronroman/financial-time-series-clustering: Unsupervised clustering to generate predictive features from stock price curves \- GitHub, accessed January 17, 2026, [https://github.com/aaronroman/financial-time-series-clustering](https://github.com/aaronroman/financial-time-series-clustering)  
10. Whats the difference between applying Correlation and DTW in a Time Series, accessed January 17, 2026, [https://stats.stackexchange.com/questions/256015/whats-the-difference-between-applying-correlation-and-dtw-in-a-time-series](https://stats.stackexchange.com/questions/256015/whats-the-difference-between-applying-correlation-and-dtw-in-a-time-series)  
11. How are correlation and cointegration related? \- Quantitative Finance Stack Exchange, accessed January 17, 2026, [https://quant.stackexchange.com/questions/1027/how-are-correlation-and-cointegration-related](https://quant.stackexchange.com/questions/1027/how-are-correlation-and-cointegration-related)  
12. Pairs Trading for Beginners: Correlation, Cointegration, Examples, and Strategy Steps, accessed January 17, 2026, [https://blog.quantinsti.com/pairs-trading-basics/](https://blog.quantinsti.com/pairs-trading-basics/)  
13. 15.2 Cointegration and Correlation | Portfolio Optimization \- Bookdown, accessed January 17, 2026, [https://bookdown.org/palomar/portfoliooptimizationbook/15.2-cointegration-vs-correlation.html](https://bookdown.org/palomar/portfoliooptimizationbook/15.2-cointegration-vs-correlation.html)  
14. An Introduction to Cointegration for Pairs Trading \- Hudson & Thames, accessed January 17, 2026, [https://hudsonthames.org/an-introduction-to-cointegration/](https://hudsonthames.org/an-introduction-to-cointegration/)  
15. Hurst Exponent in Algo Trading \- Robot Wealth, accessed January 17, 2026, [https://robotwealth.com/demystifying-the-hurst-exponent-part-2/](https://robotwealth.com/demystifying-the-hurst-exponent-part-2/)  
16. On Clustering Time Series Using Euclidean Distance and Pearson Correlation \- KOPS, accessed January 17, 2026, [https://kops.uni-konstanz.de/bitstreams/da797d8a-f202-4e29-b942-ea719482c715/download](https://kops.uni-konstanz.de/bitstreams/da797d8a-f202-4e29-b942-ea719482c715/download)  
17. \[1601.02213\] On Clustering Time Series Using Euclidean Distance and Pearson Correlation, accessed January 17, 2026, [https://arxiv.org/abs/1601.02213](https://arxiv.org/abs/1601.02213)  
18. Dynamic time warping \- Wikipedia, accessed January 17, 2026, [https://en.wikipedia.org/wiki/Dynamic\_time\_warping](https://en.wikipedia.org/wiki/Dynamic_time_warping)  
19. z2e2/fastddtw \- GitHub, accessed January 17, 2026, [https://github.com/z2e2/fastddtw](https://github.com/z2e2/fastddtw)  
20. 4\. Time Series Clustering — tslearn 0.7.0 documentation, accessed January 17, 2026, [https://tslearn.readthedocs.io/en/stable/user\_guide/clustering.html](https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html)  
21. fastdtw \- PyPI, accessed January 17, 2026, [https://pypi.org/project/fastdtw/](https://pypi.org/project/fastdtw/)  
22. PCA vs. t-SNE: Unveiling the Best Dimensionality Reduction Technique for Your Data, accessed January 17, 2026, [https://dev.to/sreeni5018/pca-vs-t-sne-unveiling-the-best-dimensionality-reduction-technique-for-your-data-ekc](https://dev.to/sreeni5018/pca-vs-t-sne-unveiling-the-best-dimensionality-reduction-technique-for-your-data-ekc)  
23. Introduction to t-SNE: Nonlinear Dimensionality Reduction and Data Visualization, accessed January 17, 2026, [https://www.datacamp.com/tutorial/introduction-t-sne](https://www.datacamp.com/tutorial/introduction-t-sne)  
24. Machine-Learning/Comparing PCA and t-SNE for Dimensionality Reduction.md at main, accessed January 17, 2026, [https://github.com/xbeat/Machine-Learning/blob/main/Comparing%20PCA%20and%20t-SNE%20for%20Dimensionality%20Reduction.md](https://github.com/xbeat/Machine-Learning/blob/main/Comparing%20PCA%20and%20t-SNE%20for%20Dimensionality%20Reduction.md)  
25. Mastering Cluster Validation with Silhouette Scores and Visualization in Python | CodeSignal Learn, accessed January 17, 2026, [https://codesignal.com/learn/courses/cluster-performance-unveiled/lessons/mastering-cluster-validation-with-silhouette-scores-and-visualization-in-python](https://codesignal.com/learn/courses/cluster-performance-unveiled/lessons/mastering-cluster-validation-with-silhouette-scores-and-visualization-in-python)  
26. Silhouette Score. I understand that learning data science… | by Amit Yadav | Biased-Algorithms | Medium, accessed January 17, 2026, [https://medium.com/biased-algorithms/silhouette-score-d85235e7638b](https://medium.com/biased-algorithms/silhouette-score-d85235e7638b)  
27. Clustering Time Series Using Dynamic Time Warping Distance in Provinces in Indonesia Based on Rice Prices \- ResearchGate, accessed January 17, 2026, [https://www.researchgate.net/publication/383807536\_Clustering\_Time\_Series\_Using\_Dynamic\_Time\_Warping\_Distance\_in\_Provinces\_in\_Indonesia\_Based\_on\_Rice\_Prices](https://www.researchgate.net/publication/383807536_Clustering_Time_Series_Using_Dynamic_Time_Warping_Distance_in_Provinces_in_Indonesia_Based_on_Rice_Prices)  
28. Tslearn \- QuantConnect.com, accessed January 17, 2026, [https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/popular-libraries/tslearn](https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/popular-libraries/tslearn)  
29. Pairs Trading Strategy using Python | by Andre Nagano \- Medium, accessed January 17, 2026, [https://medium.com/@andrejin.nagano/pair-trading-strategy-using-python-7787adb3d2e2](https://medium.com/@andrejin.nagano/pair-trading-strategy-using-python-7787adb3d2e2)  
30. Build a Pairs Trading Strategy in Python: A Step-by-Step Guide \- Interactive Brokers, accessed January 17, 2026, [https://www.interactivebrokers.com/campus/ibkr-quant-news/build-a-pairs-trading-strategy-in-python-a-step-by-step-guide/](https://www.interactivebrokers.com/campus/ibkr-quant-news/build-a-pairs-trading-strategy-in-python-a-step-by-step-guide/)  
31. Parquet vs CSV: Which Format Should You Choose? \- Last9, accessed January 17, 2026, [https://last9.io/blog/parquet-vs-csv/](https://last9.io/blog/parquet-vs-csv/)  
32. Parquet Data Format: Exploring Its Pros and Cons for 2025 \- Edge Delta, accessed January 17, 2026, [https://edgedelta.com/company/blog/parquet-data-format](https://edgedelta.com/company/blog/parquet-data-format)  
33. Compare DuckDB vs TimescaleDB, accessed January 17, 2026, [https://build.influxstaging.com/comparison/duckdb-vs-timescaledb/](https://build.influxstaging.com/comparison/duckdb-vs-timescaledb/)