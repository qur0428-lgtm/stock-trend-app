import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="股票趨勢分析 APP", layout="wide")

st.title("股票趨勢分析 APP")
st.write("輸入台股代號，自動判斷 MA、推動波、修正波、費波轉折提醒。")

# ===== 使用者輸入 =====
stock_code = st.text_input("輸入股票代號，例如 1582、2308、2330", value="1582")
cost = st.number_input("輸入你的成本價", min_value=0.0, value=93.0, step=1.0)

# ===== 抓資料函式 =====
def get_stock_data(code):
    """
    台灣上市股票通常是 .TW
    台灣上櫃股票通常是 .TWO
    先抓 .TW，抓不到再抓 .TWO
    """
    ticker_tw = f"{code}.TW"
    data = yf.download(ticker_tw, period="1y", interval="1d", progress=False)

    if data.empty:
        ticker_two = f"{code}.TWO"
        data = yf.download(ticker_two, period="1y", interval="1d", progress=False)
        ticker_used = ticker_two
    else:
        ticker_used = ticker_tw

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data, ticker_used

# ===== 分析函式 =====
def analyze_stock(df, cost):
    df = df.copy()
    df = df.dropna()

    # 計算均線
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # 狀態判斷
    def trend_status(row):
        if pd.isna(row["MA20"]):
            return "資料不足"
        if row["Close"] > row["MA5"] > row["MA10"] > row["MA20"]:
            return "偏多"
        elif row["Close"] < row["MA20"]:
            return "轉弱"
        else:
            return "觀察"

    df["狀態"] = df.apply(trend_status, axis=1)

    # 用 MA5 斜率判斷推動波 / 修正波
    df["MA5斜率"] = df["MA5"].diff()

    def wave_direction(row):
        if pd.isna(row["MA5斜率"]):
            return "資料不足"
        if row["MA5斜率"] > 0:
            return "推動波"
        elif row["MA5斜率"] < 0:
            return "修正波"
        else:
            return "盤整"

    df["波段方向"] = df.apply(wave_direction, axis=1)

    # 計算連續第幾天
    wave_days = []
    count = 0
    prev_wave = None

    for wave in df["波段方向"]:
        if wave in ["資料不足", "盤整"]:
            count = 0
        elif wave == prev_wave:
            count += 1
        else:
            count = 1

        wave_days.append(count)
        prev_wave = wave

    df["波段天數"] = wave_days

    # 費波轉折提醒
    def fib_alert(row):
        wave = row["波段方向"]
        days = row["波段天數"]

        if wave == "修正波" and days in [3, 5, 8, 13]:
            return f"修正波第 {days} 日：留意止跌/反彈"

        if wave == "推動波" and days in [8, 13, 21, 34, 55]:
            return f"推動波第 {days} 日：留意高檔轉折"

        return ""

    df["費波提醒"] = df.apply(fib_alert, axis=1)

    # 20日壓力與支撐
    df["20日壓力"] = df["Close"].shift(1).rolling(20).max()
    df["20日支撐"] = df["Close"].shift(1).rolling(20).min()

    # 損益率
    if cost > 0:
        df["損益率"] = (df["Close"] - cost) / cost * 100
    else:
        df["損益率"] = np.nan

    # 操作提醒
    def action_reminder(row):
        close = row["Close"]

        if cost <= 0:
            return "未輸入成本"

        if close >= cost * 1.03:
            return "成本上方 3%，偏強"
        elif close >= cost:
            return "回到成本區"
        elif close <= cost * 0.95:
            return "跌破成本 5%，停損警戒"
        elif not pd.isna(row["MA20"]) and close < row["MA20"]:
            return "跌破 MA20，轉弱警戒"
        else:
            return "觀察"

    df["操作提醒"] = df.apply(action_reminder, axis=1)

    return df

# ===== 主程式 =====
if stock_code:
    df, ticker_used = get_stock_data(stock_code)

    if df.empty:
        st.error("抓不到資料，請確認股票代號是否正確。")
    else:
        result = analyze_stock(df, cost)
        latest = result.iloc[-1]

        st.subheader(f"股票代號：{stock_code}，資料來源代號：{ticker_used}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("最新收盤價", f"{latest['Close']:.2f}")
        col2.metric("狀態", latest["狀態"])
        col3.metric("波段方向", latest["波段方向"])
        col4.metric("波段天數", f"第 {int(latest['波段天數'])} 日")

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("MA5", f"{latest['MA5']:.2f}" if not pd.isna(latest["MA5"]) else "資料不足")
        col6.metric("MA10", f"{latest['MA10']:.2f}" if not pd.isna(latest["MA10"]) else "資料不足")
        col7.metric("MA20", f"{latest['MA20']:.2f}" if not pd.isna(latest["MA20"]) else "資料不足")
        col8.metric("損益率", f"{latest['損益率']:.2f}%" if not pd.isna(latest["損益率"]) else "未輸入成本")

        st.markdown("### 操作提醒")
        st.info(latest["操作提醒"])

        if latest["費波提醒"] != "":
            st.warning(latest["費波提醒"])
        else:
            st.write("目前沒有費波轉折提醒。")

        st.markdown("### 支撐與壓力")
        st.write(f"20 日壓力：{latest['20日壓力']:.2f}" if not pd.isna(latest["20日壓力"]) else "20 日壓力：資料不足")
        st.write(f"20 日支撐：{latest['20日支撐']:.2f}" if not pd.isna(latest["20日支撐"]) else "20 日支撐：資料不足")

        st.markdown("### K 線收盤價與均線")
        chart_data = result[["Close", "MA5", "MA10", "MA20"]].dropna()
        st.line_chart(chart_data)

        st.markdown("### 最近 20 筆資料")
        show_cols = [
            "Close", "Volume", "MA5", "MA10", "MA20",
            "狀態", "波段方向", "波段天數", "費波提醒",
            "損益率", "操作提醒"
        ]
        st.dataframe(result[show_cols].tail(20))