The Metcalfes-RSI.py script is the first step and a test of a larger project essentially revolving around specific metrics on the cryptocurrency market. 
I began this with the desire of having all of my preferred market metrics and trading indicators viewable with a simple script.        
																		                                                                                    
Metcalfes-RSI.py is a terminal-based real-time crypto market monitoring script that continuously displays live cryptocurrency metrics without clearing previous outputs. 
This project provides an auto-refreshing log calculating Bitcoin's current Metcalfe's law ratio, and relative strength index (RSI) values for Bitcoin (btc) and Ethereum (eth).  
The script utilizes Dune analytics API and CoinAPI.io API to pull DailyActiveAddresses and CurrentMarketCap values of Bitcoin to calculate Bitcoin's current Metcalfe's law ratio. 

The forumla below was used in this script:

	Metcalfe's ratio = CurrentMarketCap/(DailyActiveAddresses)^2

The NVM-Ratio.py file contains the addition of the Network Value to Metcalfe Ratio (NVM Ratio). The NVM ratio is the log of the market capitalization divided by the log of the square of daily active addresses in the specified window. An NVM Ratio close to 1 indicates that the network’s value is in line with what Metcalfe’s Law predicts, while significant deviations suggest potential overvaluation or undervaluation.

The formula below was used in this script: 

	NVM ratio = ln(CurrentMarketCap)/ln(DailyActiveAddresses^2)  

Metcalfe's Law states that the value of a network is proportional to the square of its users (n²). The ratio above gives an indication of how much value 
(in terms of market cap) is generated per pair of active users. For Bitcoin, this highlights how important adoption and network growth is for asset appreciation.
As more people use Bitcoin the network itself becomes exponentially more valuable due to increased trust, liquidity, and utility. 



Risk-and-Returns-metrics.py is a Python script that analyzes market data and calculates key financial metrics for a handpicked group of cryptocurrencies (​​BTC, ETH, AAVE, ADA, CRV, DOGE, HBAR, SOL, SHIB, SUI, TON, XRP). This script automates the process of fetching historical price data from CoinAPI and calculates comprehensive risk and performance analytics across multiple time frequencies. This script processes Open, High, Low, Close, and Volume (OHLCV) data for the selected assets and presents the insights via a console dashboard as well as an Excel report. Segmented into daily, weekly, and monthly intervals, the metrics enable a detailed analysis on the performance of each cryptocurrency against Bitcoin.

Please read Risk-and-Returns.md for full documentation of this script. 


