**Crypto Risk-Return Analysis Script**

Risk-and-Returns-metrics.py is a Python script that analyzes market data and calculates key financial metrics for a handpicked group of cryptocurrencies (​​BTC, ETH, AAVE, ADA, CRV, DOGE, HBAR, SOL, SHIB, SUI, TON, XRP). This script automates the process of fetching historical price data from CoinAPI and calculates comprehensive risk and performance analytics across multiple time frequencies. 

**Overview**

This script processes Open, High, Low, Close, and Volume (OHLCV) data for the selected assets and presents the insights via a console dashboard as well as an Excel report. Segmented into daily, weekly, and monthly intervals, the metrics enable a detailed analysis on the performance of each cryptocurrency against Bitcoin.
To run the script, you'll need several Python packages: requests for API communication, pandas for data manipulation, numpy for numerical computations, statsmodels for statistical analysis, and matplotlib for any potential visualization needs. You will also need a (free) market data API key from CoinAPI.io. 

**Core Metrics and Calculations**

The script calculates several key financial metrics, each providing unique insights into cryptocurrency performance. Here's a detailed breakdown of each metric:

Alpha (α)

Measures excess returns relative to Bitcoin benchmark performance
Alpha is calculated by running an Ordinary Least Squares (OLS) regression of altcoin returns against Bitcoin returns. The risk_free_rate parameter defaults to 0, meaning no risk-free return is subtracted from either the altcoin or Bitcoin returns. This creates a direct comparison between the altcoin and Bitcoin returns, with Bitcoin acting as the benchmark. A positive alpha indicates outperformance versus Bitcoin, while a negative alpha suggests underperformance.

Beta (β)

Measures an asset's volatility relative to Bitcoin
Beta calculation involves computing the covariance of returns divided by Bitcoin's return variance. The metric is available in daily, weekly, and monthly frequencies to provide different temporal perspectives on the relationship. A beta greater than 1 indicates higher volatility than Bitcoin, while a beta less than 1 suggests lower volatility. This provides insight into how responsive an altcoin is to Bitcoin's price movements.

Sharpe Ratio

Risk-adjusted return metric measuring return relative to risk
The Sharpe Ratio calculation is performed monthly using the US 10-Year Treasury rate (4.478%) as the risk-free rate. (Will add code to scrape for the US10Y so I don't have to manually update). The calculate_monthly_sharpe_ratio function first converts the annual risk-free rate to a monthly rate by dividing by 12. It then subtracts this monthly risk-free rate from the average monthly return to get the excess return. This excess return is then divided by the standard deviation of the monthly returns to produce the final Sharpe Ratio. A higher ratio indicates better risk-adjusted performance.

Correlation

Measures price movement relationship with Bitcoin
Correlation is calculated using the Pearson correlation coefficient, which produces values ranging from -1 to +1. The calculation examines the relationship between altcoin and Bitcoin returns across all time frequencies. A correlation of +1 indicates perfect positive correlation, -1 indicates perfect negative correlation, and 0 indicates no correlation. This helps us understand how closely an altcoin's price movements mirror Bitcoin's behavior.

Rolling Beta

Dynamic beta calculation showing how asset's sensitivity to Bitcoin changes over time
The rolling beta calculation begins by merging asset returns with Bitcoin returns over a moving window. The function calculate_rolling_beta takes three parameters: alt_returns, bench_returns, and a window size (30 days for daily data, 4 weeks for weekly, and 3 months for monthly). It computes the rolling covariance between the asset and Bitcoin over the specified window, along with the rolling variance of Bitcoin's returns. The final rolling beta is calculated as the ratio of rolling covariance to rolling variance, with only the most recent value being returned.

Volatility

Standard deviation of log returns, measuring how much an asset's return varies independently of Bitcoin
Volatility calculation is performed through the calculate_volatility function, which computes the standard deviation of the return series. The calculation process differs based on the frequency being analyzed. For daily volatility, it uses daily log returns; for weekly volatility, it uses weekly returns; and for monthly volatility, it uses monthly returns. This provides a comprehensive view of price stability across different time horizons.

Maximum Drawdown

Largest peak-to-trough decline in asset's price, showing worst-case loss scenario
The maximum drawdown calculation is performed by the calculate_max_drawdown function, which first computes a running maximum representing the cumulative highest price up to each point in the series. It then calculates the drawdown at each point as the percentage drop from that running maximum. The final maximum drawdown value is determined by taking the minimum value in this drawdown series, effectively capturing the worst peak-to-trough decline over the period.

**Usage and Output**

When you run the script, it automatically fetches 90 days of historical data, calculates all metrics, displays results in the console, and updates an Excel dashboard. The console output provides a quick overview of current metrics, while the Excel dashboard (saved as "dashboard_metrics.xlsx") maintains a historical record of all runs, complete with metric definitions and formatted for readability. The script will create a file titled, “dashboard_metrics.xlsx”, if there is not already one existing. 
All timestamps use UTC to ensure consistency across different time zones, and Bitcoin's metrics are included as a benchmark row for comparison purposes. The error handling system ensures that individual failures don't halt the entire process, making the script reliable for regular use.

**Technical Implementation**

The script initiates data retrieval through the fetch_ohlcv function, which sends API requests to CoinAPI to obtain OHLCV data for specified time intervals.This data is then processed by the compute_frequency_returns function, which resamples price data to the desired frequency and computes logarithmic returns. The script handles missing data appropriately and supports daily, weekly, and monthly end frequencies.
The dashboard creation is facilitated by the get_dashboard_df function, which structures metrics into an organized DataFrame with consistent column ordering. For Excel export, the append_dashboard_to_excel function appends new results to an existing Excel file, preserving historical data. This file includes timestamps and detailed metric definitions for comprehensive analysis.
