import curses
import time
import requests
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.simplefilter("ignore", FutureWarning)

# API Keys and URLs
COINAPI_KEY = "API KEY HERE"
DUNE_API_KEY = "API KEY HERE"
DUNE_API_URL = "https://api.dune.com/api/v1/query/1889580/results"
COINAPI_PRICE_URL = "https://rest.coinapi.io/v1/exchangerate/BTC/USD"
OHLCV_BASE_URL = "https://rest.coinapi.io/v1/ohlcv"

# RSI Configuration
SYMBOL_IDS = {
    "BTCUSD": "COINBASE_SPOT_BTC_USD",
    "ETHUSD": "COINBASE_SPOT_ETH_USD"
}
PERIODS = {
    "4h": "4HRS",
    "1h": "1HRS",
    "1m": "1MIN"
}
RSI_PERIOD = 14
LIMIT = 100

def format_rsi(rsi_value):
    """Helper function to format RSI values"""
    if rsi_value is None or pd.isna(rsi_value):
        return "N/A     "
    return f"{rsi_value:.2f}".ljust(8)

def compute_rsi(prices, period=14):
    prices_series = pd.Series(prices)
    delta = prices_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, np.nan).ffill()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_ohlcv_data(symbol_id, period, limit=LIMIT):
    now = datetime.utcnow()
    if period == "1MIN":
        delta = timedelta(minutes=limit)
    elif period == "1HRS":
        delta = timedelta(hours=limit)
    elif period == "4HRS":
        delta = timedelta(hours=limit * 4)
    else:
        delta = timedelta(hours=limit)
    
    time_start = (now - delta).replace(microsecond=0).isoformat()
    url = f"{OHLCV_BASE_URL}/{symbol_id}/history"
    params = {
        "period_id": period,
        "time_start": time_start,
        "limit": limit
    }
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching OHLCV data: {e}")
        return None

def get_latest_rsi(symbol, period_label, rsi_period=RSI_PERIOD):
    symbol_id = SYMBOL_IDS[symbol]
    period = PERIODS[period_label]
    
    data = get_ohlcv_data(symbol_id, period, limit=LIMIT)
    if not data:
        return None

    close_prices = [record["price_close"] for record in data]
    if len(close_prices) < rsi_period:
        return None
    
    rsi_series = compute_rsi(close_prices, period=rsi_period)
    return rsi_series.iloc[-1]

def get_bitcoin_market_cap():
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    try:
        response = requests.get(COINAPI_PRICE_URL, headers=headers)
        data = response.json()
        
        if 'rate' not in data:
            return None
            
        price_usd = float(data['rate'])
        current_supply = 19600000
        return price_usd * current_supply
    except Exception as e:
        print(f"Error fetching market cap: {e}")
        return None

def fetch_active_addresses():
    headers = {"X-DUNE-API-KEY": DUNE_API_KEY}
    try:
        response = requests.get(DUNE_API_URL, headers=headers)
        data = response.json()
        if "result" in data and "rows" in data["result"]:
            rows = data["result"]["rows"]
            if rows:
                latest_row = max(rows, key=lambda r: datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S.%f %Z"))
                return latest_row["users"]
        return None
    except Exception as e:
        return None

def calculate_metcalfe_ratio(market_cap, active_addresses):
    if market_cap is None or active_addresses is None:
        return None
    try:
        return market_cap / (int(active_addresses) ** 2)
    except Exception:
        return None

def calculate_NVM_Ratio(market_cap, active_addresses):
    try:
        return math.log(market_cap) / math.log(active_addresses ** 2)
    except Exception:
        return None

def format_large_number(num):
    if num is None:
        return "N/A"
    billions = num / 1_000_000_000
    return f"${billions:.2f}B"

def display_ui(stdscr):
    curses.curs_set(0)
    stdscr.scrollok(True)  # Enable scrolling so new outputs are appended
    while True:
        try:
            # Fetch all data
            market_cap = get_bitcoin_market_cap()
            active_addresses = fetch_active_addresses()
            metcalfe_ratio = calculate_metcalfe_ratio(market_cap, active_addresses)
            nvm_ratio = calculate_NVM_Ratio(market_cap, active_addresses)
            
            # Get RSI values
            btc_4h = get_latest_rsi("BTCUSD", "4h")
            btc_1h = get_latest_rsi("BTCUSD", "1h")
            btc_1m = get_latest_rsi("BTCUSD", "1m")
            eth_4h = get_latest_rsi("ETHUSD", "4h")
            eth_1h = get_latest_rsi("ETHUSD", "1h")
            eth_1m = get_latest_rsi("ETHUSD", "1m")
            
            # Format RSI values
            btc_4h_str = format_rsi(btc_4h)
            btc_1h_str = format_rsi(btc_1h)
            btc_1m_str = format_rsi(btc_1m)
            eth_4h_str = format_rsi(eth_4h)
            eth_1h_str = format_rsi(eth_1h)
            eth_1m_str = format_rsi(eth_1m)

            # Prepare display lines
            lines = [
                "\n📊 Crypto Market Monitor",
                f"💰 BTC Market Cap: {format_large_number(market_cap)}",
                f"👥 Active Addresses: {active_addresses:,}" if active_addresses is not None else "👥 Active Addresses: N/A",
                f"📐 Metcalfe's Ratio: {metcalfe_ratio:.8f}" if metcalfe_ratio is not None else "📐 Metcalfe's Ratio: N/A",
                f"📊 NVM Ratio: {nvm_ratio:.8f}" if nvm_ratio is not None else "📊 NVM Ratio: N/A",
                "📈 RSI Values (14-period):",
                "--------------------------------------------------------------------------------",
                "Symbol   | 4h RSI   | 1h RSI   | 1m RSI",
                "--------------------------------------------------------------------------------",
                f"BTCUSD   | {btc_4h_str} | {btc_1h_str} | {btc_1m_str}",
                f"ETHUSD   | {eth_4h_str} | {eth_1h_str} | {eth_1m_str}",
                "",
                "🔄 Refreshing in 60 seconds..."
            ]
            
            for line in lines:
                stdscr.addstr(line + "\n")
            stdscr.refresh()
            time.sleep(60)
        except curses.error:
            continue
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    curses.wrapper(display_ui)










