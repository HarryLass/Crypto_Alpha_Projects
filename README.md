The Metcalfs-RSI.py script is the first step and a test of a larger project essentially revolving around specific metrics on the cryptocurrency market. 
I began this with the desire of having all of my preferred market metrics and trading indicators viewable with a simple script.        
																		                                                                                    
Metcalfs-RSI.py is a terminal-based real-time crypto market monitoring script that continuously displays live cryptocurrency metrics without clearing previous outputs. 
This project provides an auto-refreshing log calculating Bitcoin's current Metcalf's law ratio, and relative strength index (RSI) values for Bitcoin (btc) and Ethereum (eth).  
The script utilizes Dune analytics API and CoinAPI.io API to pull DailyActiveAddresses and CurrentMarketCap values of Bitcoin to calculate Bitcoin's current Metcalf's law ratio. 

The forumla below was used in this script:

                          Metcalf's ratio = CurrentMarketCap/(DailyActiveAddresses)^2



Metcalfe's Law states that the value of a network is proportional to the square of its users (n²). The ratio above gives an indication of how much value 
(in terms of market cap) is generated per pair of active users. For Bitcoin, this highlights how important adoption and network growth is for asset appreciation.
As more people use Bitcoin—whether as investors, miners, or for transactions—the network itself becomes exponentially more valuable due to increased trust, liquidity, and utility. 

