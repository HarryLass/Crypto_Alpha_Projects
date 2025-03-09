Altcoin Analysis Dashboard
-
Building off of the risk and returns metrics, the altcoin alpha dashboard script is a real-time cryptocurrency monitoring tool that combines multiple predictive models, technical indicators, and risk metrics to forecast altcoin performance. 

The script leverages multiple machine learning models, including ARIMA for capturing temporal dependencies, Random Forest, Gradient Boosting Machine (GBM), Linear Regression, and Support Vector Regression (SVR) to forecast next-day prices and returns. The parameters for these models are automated and systematically tuned using techniques like auto_arima for ARIMA and GridSearchCV for Random Forest and GBM to enhance prediction accuracy. 

The dashboard fetches OHLCV data from the CoinAPI.io market data API for various altcoins, calculates technical indicators such as RSI, MACD, and Bollinger Bands, and computes risk metrics like VaR, Alpha, Beta, and Sharpe Ratio. It also provides visualizations for rolling metrics, correlation heatmaps, and technical analysis. The dashboard is built using Dash and Plotly, and tracks metrics in excel, offering a user-friendly interface with real-time data calculations.


**Features**
- Real-time altcoin price monitoring
- Multiple predictive models (ARIMA, Random Forest, GBM, Linear Regression, SVR)
- Comprehensive risk metrics
- Interactive technical analysis charts
- Correlation analysis
- Data tracking in Excel 
- Night/Day mode 
- Automated data refresh capabilities

https://github.com/HarryLass/Crypto_Alpha_Projects/tree/main/Altcoin_analysis_dashboard


Metcalfes and NVM Ratios
-

metcalfes-RSI.py is a terminal-based crypto monitoring tool that displays real-time metrics without clearing previous outputs. The script calculates Bitcoin's Metcalfe's law ratio (CurrentMarketCap/(DailyActiveAddresses)²) and RSI values for Bitcoin and Ethereum using data from Dune Analytics and CoinAPI.io. The expanded nvm-ratio.py adds Network Value to Metcalfe Ratio calculation (ln(CurrentMarketCap)/ln(DailyActiveAddresses²)), which helps identify potential over/undervaluation. This project represents the first step in a larger effort to consolidate preferred market metrics and trading indicators into simple, accessible scripts. 
https://github.com/HarryLass/Crypto_Alpha_Projects/blob/main/Metcalfes_NVM_ratios/Metcalfes_NVM.md


Risk and Returns 
-

risk-and-returns-metrics.py is a Python script that analyzes market data and calculates key financial metrics for a handpicked group of cryptocurrencies (​​BTC, ETH, AAVE, ADA, CRV, DOGE, HBAR, SOL, SHIB, SUI, TON, XRP). This script automates the process of fetching historical price data from CoinAPI and calculates comprehensive risk and performance analytics across multiple time frequencies. This script processes Open, High, Low, Close, and Volume (OHLCV) data for the selected assets and presents the insights via a console dashboard as well as an Excel report. Segmented into daily, weekly, and monthly intervals, the metrics enable a detailed analysis on the performance of each cryptocurrency against Bitcoin. 

Please check https://github.com/HarryLass/Crypto_Alpha_Projects/blob/main/Risk_Returns/Risk-and-Returns.md for full documentation of this script. 


