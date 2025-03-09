import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

API_KEY = "YOUR API KEY HERE"
BASE_URL = "https://rest.coinapi.io/v1/"

# -----------------------------
# Data Fetching and Processing Functions
# -----------------------------
def fetch_ohlcv(symbol_id, period_id, time_start, time_end):
    """Fetch historical OHLCV data for a given symbol from CoinAPI."""
    endpoint = f"{BASE_URL}ohlcv/{symbol_id}/history"
    headers = {"X-CoinAPI-Key": API_KEY}
    params = {
        "period_id": period_id,  # e.g., "1DAY"
        "time_start": time_start,
        "time_end": time_end,
        "limit": 10000,
        "include_empty_items": "true"
    }
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def compute_frequency_returns(df, freq):
    """
    Resample the closing price at the given frequency and compute log returns.
    Frequency can be 'D' (daily), 'W' (weekly), or 'ME' (monthly end).
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time', inplace=True)
    df['price_close'] = pd.to_numeric(df['price_close'])
    price_series = df['price_close'].resample(freq).last().dropna()
    returns = np.log(price_series / price_series.shift(1)).dropna()
    return returns

def resample_prices(df, freq):
    """Resample the price series at a given frequency."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time', inplace=True)
    df['price_close'] = pd.to_numeric(df['price_close'])
    return df['price_close'].resample(freq).last().dropna()

# -----------------------------
# Metrics Calculation Functions
# -----------------------------
def calculate_beta_for_returns(alt_returns, bench_returns):
    """Calculate beta from two return series after aligning them."""
    merged = pd.merge(
        alt_returns.rename("alt_return"),
        bench_returns.rename("bench_return"),
        left_index=True, right_index=True, how='inner'
    )
    if merged.empty:
        return None
    beta = np.cov(merged['alt_return'], merged['bench_return'])[0, 1] / np.var(merged['bench_return'])
    return beta

def calculate_monthly_sharpe_ratio(returns, risk_free_rate=0.04478):
    """Calculate the Sharpe Ratio using monthly returns."""
    if returns.empty:
        return None
    periods_per_year = 12
    monthly_rf = risk_free_rate / periods_per_year
    mean_return = returns.mean() - monthly_rf
    std_return = returns.std()
    if std_return == 0:
        return np.nan
    return mean_return / std_return

def calculate_correlation(alt_returns, bench_returns):
    """Calculate the Pearson correlation coefficient."""
    merged = pd.merge(
        alt_returns.rename("alt_return"),
        bench_returns.rename("bench_return"),
        left_index=True, right_index=True, how='inner'
    )
    if merged.empty:
        return None
    return merged['alt_return'].corr(merged['bench_return'])

def calculate_max_drawdown(price_series):
    """Calculate the Maximum Drawdown for a given price series."""
    if price_series.empty:
        return None
    running_max = price_series.cummax()
    drawdown = price_series / running_max - 1
    return drawdown.min()

def calculate_volatility(returns):
    """Calculate volatility (standard deviation) of returns."""
    return returns.std() if not returns.empty else None

def calculate_alpha(alt_returns, bench_returns, risk_free_rate=0):
    """
    Calculate alpha (excess return) using OLS regression.
    Alpha is the intercept of regressing alt_returns on bench_returns.
    """
    merged = pd.merge(
        alt_returns.rename("alt_return"),
        bench_returns.rename("bench_return"),
        left_index=True, right_index=True, how='inner'
    )
    if merged.empty:
        return None
    X = sm.add_constant(merged['bench_return'])
    model = sm.OLS(merged['alt_return'], X).fit()
    return model.params[0]

def calculate_rolling_beta(alt_returns, bench_returns, window=30):
    """
    Calculate rolling beta using a moving window.
    Returns the most recent rolling beta value.
    """
    merged = pd.merge(
        alt_returns.rename("alt_return"),
        bench_returns.rename("bench_return"),
        left_index=True, right_index=True, how='inner'
    )
    if len(merged) < window:
        return None
    roll_cov = merged['alt_return'].rolling(window=window).cov(merged['bench_return'])
    roll_var = merged['bench_return'].rolling(window=window).var()
    return (roll_cov / roll_var).iloc[-1]

# -----------------------------
# Dashboard and Excel Export Functions
# -----------------------------
def get_dashboard_df(metrics_dict):
    """
    Build a DataFrame from the nested metrics dictionary with consistent ordering.
    """
    metric_columns = [
        "Daily Beta", "Weekly Beta", "Monthly Beta",
        "Monthly Sharpe",
        "Daily Corr", "Weekly Corr", "Monthly Corr",
        "Daily Drawdown", "Weekly Drawdown", "Monthly Drawdown",
        "Daily Volatility", "Weekly Volatility", "Monthly Volatility",
        "Daily Alpha", "Weekly Alpha", "Monthly Alpha",
        "Daily Rolling Beta", "Weekly Rolling Beta", "Monthly Rolling Beta"
    ]
    # Order assets alphabetically with BTC last
    assets = sorted([asset for asset in metrics_dict if asset != "BTC"])
    if "BTC" in metrics_dict:
        assets.append("BTC")
    data = {}
    for asset in assets:
        freq_metrics = metrics_dict.get(asset, {})
        row = {
            "Daily Beta":         freq_metrics.get("Daily", {}).get("Beta"),
            "Weekly Beta":        freq_metrics.get("Weekly", {}).get("Beta"),
            "Monthly Beta":       freq_metrics.get("Monthly", {}).get("Beta"),
            "Monthly Sharpe":     freq_metrics.get("Monthly", {}).get("Sharpe"),
            "Daily Corr":         freq_metrics.get("Daily", {}).get("Corr"),
            "Weekly Corr":        freq_metrics.get("Weekly", {}).get("Corr"),
            "Monthly Corr":       freq_metrics.get("Monthly", {}).get("Corr"),
            "Daily Drawdown":     freq_metrics.get("Daily", {}).get("Max Drawdown"),
            "Weekly Drawdown":    freq_metrics.get("Weekly", {}).get("Max Drawdown"),
            "Monthly Drawdown":   freq_metrics.get("Monthly", {}).get("Max Drawdown"),
            "Daily Volatility":   freq_metrics.get("Daily", {}).get("Volatility"),
            "Weekly Volatility":  freq_metrics.get("Weekly", {}).get("Volatility"),
            "Monthly Volatility": freq_metrics.get("Monthly", {}).get("Volatility"),
            "Daily Alpha":        freq_metrics.get("Daily", {}).get("Alpha"),
            "Weekly Alpha":       freq_metrics.get("Weekly", {}).get("Alpha"),
            "Monthly Alpha":      freq_metrics.get("Monthly", {}).get("Alpha"),
            "Daily Rolling Beta": freq_metrics.get("Daily", {}).get("Rolling Beta"),
            "Weekly Rolling Beta":freq_metrics.get("Weekly", {}).get("Rolling Beta"),
            "Monthly Rolling Beta":freq_metrics.get("Monthly", {}).get("Rolling Beta"),
        }
        data[asset] = row
    df = pd.DataFrame.from_dict(data, orient="index")
    return df[metric_columns]

def display_dashboard(metrics_dict, run_timestamp):
    """Display the dashboard in the terminal."""
    print(f"\nRun Timestamp: {run_timestamp}")
    df_dashboard = get_dashboard_df(metrics_dict)
    print(df_dashboard.to_string(float_format="%.4f"))
    return df_dashboard

def append_dashboard_to_excel(df_dashboard, run_timestamp, filename="dashboard_metrics.xlsx"):
    """
    Append the dashboard DataFrame to Excel with improved formatting and labeling.
    """
    header_df = pd.DataFrame([{"Asset": f"Run Timestamp: {run_timestamp}", 
                                **{col: "" for col in df_dashboard.columns}}])
    df_with_assets = df_dashboard.reset_index().rename(columns={'index': 'Asset'})
    section_header = pd.DataFrame([{"Asset": "Assets →", 
                                     **{col: col for col in df_dashboard.columns}}])
    metric_definitions = pd.DataFrame({
        "Asset": [
            "Metric Definitions:",
            "Beta: Volatility measure relative to BTC",
            "Sharpe: Risk-adjusted return metric",
            "Correlation: Price correlation with BTC",
            "Drawdown: Maximum price decline from peak",
            "Volatility: Standard deviation of log returns",
            "Alpha: OLS regression intercept of returns vs. BTC",
            "Rolling Beta: Beta over a moving window"
        ]
    })
    for col in df_dashboard.columns:
        metric_definitions[col] = ""
    final_columns = ["Asset"] + list(df_dashboard.columns)
    metric_definitions = metric_definitions[final_columns]
    blank_row = pd.DataFrame([{col: "" for col in ["Asset"] + list(df_dashboard.columns)}])
    final_df = pd.concat([
        header_df,
        blank_row,
        section_header,
        df_with_assets,
        blank_row,
        metric_definitions
    ], ignore_index=True)
    try:
        if os.path.exists(filename):
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                workbook = writer.book
                last_row = workbook['Sheet1'].max_row if 'Sheet1' in workbook.sheetnames else 0
                final_df.to_excel(writer, sheet_name='Sheet1', startrow=last_row + 2, index=False, header=False)
        else:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
        print(f"\nDashboard successfully appended to {filename}")
    except Exception as e:
        print(f"\nError appending/creating Excel file: {e}")

# -----------------------------
# Main Code: Data Acquisition and Metrics Calculation
# -----------------------------
def main():
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=90)
    time_start = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_end = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    period = "1DAY"  # Daily data
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Data run at:", run_timestamp)
    
    symbols = {
        "ETH":  "BINANCE_SPOT_ETH_USDT",
        "XRP":  "BINANCE_SPOT_XRP_USDT",
        "ADA":  "BINANCE_SPOT_ADA_USDT",
        "SOL":  "BINANCE_SPOT_SOL_USDT",
        "HBAR": "BINANCE_SPOT_HBAR_USDT",
        "CRV":  "BINANCE_SPOT_CRV_USDT",
        "DOGE": "BINANCE_SPOT_DOGE_USDT",
        "SUI":  "BINANCE_SPOT_SUI_USDT",
        "TON":  "BINANCE_SPOT_TON_USDT",
        "SHIB": "BINANCE_SPOT_SHIB_USDT",
        "AAVE": "BINANCE_SPOT_AAVE_USDT"
    }
    
    benchmark_symbol = "BINANCE_SPOT_BTC_USDT"
    print("Fetching Bitcoin (benchmark) daily data...")
    btc_data = fetch_ohlcv(benchmark_symbol, period, time_start, time_end)
    
    btc_daily_returns = compute_frequency_returns(btc_data.copy(), 'D')
    btc_weekly_returns = compute_frequency_returns(btc_data.copy(), 'W')
    btc_monthly_returns = compute_frequency_returns(btc_data.copy(), 'ME')
    
    btc_monthly_sharpe = calculate_monthly_sharpe_ratio(btc_monthly_returns, risk_free_rate=0.04478)
    btc_monthly_prices = resample_prices(btc_data.copy(), 'ME')
    btc_monthly_drawdown = calculate_max_drawdown(btc_monthly_prices)
    
    btc_daily_prices = resample_prices(btc_data.copy(), 'D')
    btc_weekly_prices = resample_prices(btc_data.copy(), 'W')
    btc_daily_drawdown = calculate_max_drawdown(btc_daily_prices)
    btc_weekly_drawdown = calculate_max_drawdown(btc_weekly_prices)
    
    # Additional Bitcoin metrics
    btc_daily_vol = calculate_volatility(btc_daily_returns)
    btc_weekly_vol = calculate_volatility(btc_weekly_returns)
    btc_monthly_vol = calculate_volatility(btc_monthly_returns)
    btc_daily_alpha = 0
    btc_weekly_alpha = 0
    btc_monthly_alpha = 0
    btc_daily_roll_beta = 1.0
    btc_weekly_roll_beta = 1.0
    btc_monthly_roll_beta = 1.0
    
    print(f"\nBitcoin Monthly Metrics:\n  Monthly Sharpe: {btc_monthly_sharpe:.4f}, Monthly Drawdown: {btc_monthly_drawdown:.4f}")
    
    metrics = {}
    for alt, symbol in symbols.items():
        try:
            print(f"\nFetching data for {alt}...")
            alt_data = fetch_ohlcv(symbol, period, time_start, time_end)
            
            alt_daily_returns = compute_frequency_returns(alt_data.copy(), 'D')
            alt_weekly_returns = compute_frequency_returns(alt_data.copy(), 'W')
            alt_monthly_returns = compute_frequency_returns(alt_data.copy(), 'ME')
            
            daily_beta = calculate_beta_for_returns(alt_daily_returns, btc_daily_returns)
            weekly_beta = calculate_beta_for_returns(alt_weekly_returns, btc_weekly_returns)
            monthly_beta = calculate_beta_for_returns(alt_monthly_returns, btc_monthly_returns)
            
            monthly_sharpe = calculate_monthly_sharpe_ratio(alt_monthly_returns, risk_free_rate=0.04478)
            
            daily_corr = calculate_correlation(alt_daily_returns, btc_daily_returns)
            weekly_corr = calculate_correlation(alt_weekly_returns, btc_weekly_returns)
            monthly_corr = calculate_correlation(alt_monthly_returns, btc_monthly_returns)
            
            daily_prices = resample_prices(alt_data.copy(), 'D')
            weekly_prices = resample_prices(alt_data.copy(), 'W')
            monthly_prices = resample_prices(alt_data.copy(), 'ME')
            
            daily_drawdown = calculate_max_drawdown(daily_prices)
            weekly_drawdown = calculate_max_drawdown(weekly_prices)
            monthly_drawdown = calculate_max_drawdown(monthly_prices)
            
            daily_vol = calculate_volatility(alt_daily_returns)
            weekly_vol = calculate_volatility(alt_weekly_returns)
            monthly_vol = calculate_volatility(alt_monthly_returns)
            
            daily_alpha = calculate_alpha(alt_daily_returns, btc_daily_returns, risk_free_rate=0)
            weekly_alpha = calculate_alpha(alt_weekly_returns, btc_weekly_returns, risk_free_rate=0)
            monthly_alpha = calculate_alpha(alt_monthly_returns, btc_monthly_returns, risk_free_rate=0)
            
            daily_roll_beta = calculate_rolling_beta(alt_daily_returns, btc_daily_returns, window=30)
            weekly_roll_beta = calculate_rolling_beta(alt_weekly_returns, btc_weekly_returns, window=4)
            monthly_roll_beta = calculate_rolling_beta(alt_monthly_returns, btc_monthly_returns, window=3)
            
            metrics[alt] = {
                "Daily": {
                    "Beta": daily_beta, "Corr": daily_corr, "Max Drawdown": daily_drawdown,
                    "Volatility": daily_vol, "Alpha": daily_alpha, "Rolling Beta": daily_roll_beta
                },
                "Weekly": {
                    "Beta": weekly_beta, "Corr": weekly_corr, "Max Drawdown": weekly_drawdown,
                    "Volatility": weekly_vol, "Alpha": weekly_alpha, "Rolling Beta": weekly_roll_beta
                },
                "Monthly": {
                    "Beta": monthly_beta, "Corr": monthly_corr, "Max Drawdown": monthly_drawdown,
                    "Sharpe": monthly_sharpe, "Volatility": monthly_vol, "Alpha": monthly_alpha,
                    "Rolling Beta": monthly_roll_beta
                }
            }
            print(f"{alt} Metrics:")
            print(f"  Daily   -> Beta: {daily_beta:.4f}, Drawdown: {daily_drawdown:.4f}, Vol: {daily_vol:.4f}, Alpha: {daily_alpha:.4f}, Rolling Beta: {daily_roll_beta:.4f}")
            print(f"  Weekly  -> Beta: {weekly_beta:.4f}, Drawdown: {weekly_drawdown:.4f}, Vol: {weekly_vol:.4f}, Alpha: {weekly_alpha:.4f}, Rolling Beta: {weekly_roll_beta:.4f}")
            print(f"  Monthly -> Beta: {monthly_beta:.4f}, Sharpe: {monthly_sharpe:.4f}, Corr: {monthly_corr:.4f}, Drawdown: {monthly_drawdown:.4f}, Vol: {monthly_vol:.4f}, Alpha: {monthly_alpha:.4f}, Rolling Beta: {monthly_roll_beta:.4f}")
            
        except Exception as e:
            print(f"Error processing data for {alt}: {e}")
            metrics[alt] = {
                "Daily": {"Beta": None, "Corr": None, "Max Drawdown": None, "Sharpe": None, "Volatility": None, "Alpha": None, "Rolling Beta": None},
                "Weekly": {"Beta": None, "Corr": None, "Max Drawdown": None, "Sharpe": None, "Volatility": None, "Alpha": None, "Rolling Beta": None},
                "Monthly": {"Beta": None, "Corr": None, "Max Drawdown": None, "Sharpe": None, "Volatility": None, "Alpha": None, "Rolling Beta": None}
            }
    
    # Add Bitcoin's own metrics (benchmark row)
    metrics["BTC"] = {
        "Daily": {"Beta": 1.0, "Corr": 1.0, "Max Drawdown": btc_daily_drawdown, "Sharpe": None, "Volatility": btc_daily_vol, "Alpha": 0, "Rolling Beta": 1.0},
        "Weekly": {"Beta": 1.0, "Corr": 1.0, "Max Drawdown": btc_weekly_drawdown, "Sharpe": None, "Volatility": btc_weekly_vol, "Alpha": 0, "Rolling Beta": 1.0},
        "Monthly": {"Beta": 1.0, "Corr": 1.0, "Max Drawdown": btc_monthly_drawdown, "Sharpe": btc_monthly_sharpe, "Volatility": btc_monthly_vol, "Alpha": 0, "Rolling Beta": 1.0}
    }
    
    dashboard_df = get_dashboard_df(metrics)
    display_dashboard(metrics, run_timestamp)
    append_dashboard_to_excel(dashboard_df, run_timestamp, filename="dashboard_metrics.xlsx")
    
if __name__ == "__main__":
    main()
