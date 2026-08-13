import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False


st.set_page_config(page_title="股票趨勢分析 APP", layout="wide")

# ===== 股票名稱對照表 =====
STOCK_MAP = {
    "台積電": "2330",
    "台達電": "2308",
    "信錦": "1582",
    "台郡": "6269",
    "群聯": "8299",
    "英業達": "2356",
    "緯創": "3231",
    "緯穎": "6669",
    "廣達": "2382",
    "神達": "3706",
    "鴻海": "2317",
    "聯發科": "2454",
    "技嘉": "2376",
    "華碩": "2357",
    "勤誠": "8210",
    "雙鴻": "3324",
    "奇鋐": "3017",
    "華擎": "3515",
    "永擎": "7711",
    "佳必琪": "6197",
    "所羅門": "2359",
    "金像電": "2368",
    "台光電": "2383",
    "健策": "3653",
    "川湖": "2059",
    "大立光": "3008",
}

st.title("股票趨勢分析 APP")
st.write("輸入台股名稱或代號，自動判斷均線、K線型態、5K結構、量價關係、推動波 / 修正波、進場評分、獲利調節與停損風險。")

# ===== 使用者輸入 =====
stock_input = st.text_input("輸入股票名稱或代號，例如：信錦、台達電、1582、2308", value="信錦")
stock_input = stock_input.strip()

if stock_input in STOCK_MAP:
    stock_code = STOCK_MAP[stock_input]
    stock_name = stock_input
else:
    stock_code = stock_input
    stock_name = stock_input

# 成本改成選填：沒有持股也可以分析進場時機
use_cost = st.checkbox("我要輸入持股成本", value=False)

if use_cost:
    cost = st.number_input("輸入你的成本價", min_value=0.0, value=93.0, step=1.0)
else:
    cost = 0.0

st.markdown("---")
realtime_mode = st.checkbox("開啟盤中即時監控（1分鐘資料）", value=True)

if realtime_mode:
    refresh_seconds = st.number_input("自動刷新秒數", min_value=30, max_value=300, value=60, step=30)
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=int(refresh_seconds) * 1000, key="stock_realtime_refresh")
    else:
        st.warning("尚未安裝 streamlit-autorefresh，APP 不會自動刷新；請更新 requirements.txt 後重新部署。")
else:
    refresh_seconds = None

# ===== 抓資料函式 =====
def _clean_yfinance_data(data):
    """整理 yfinance 回傳欄位，避免 MultiIndex 造成後面計算錯誤。"""
    if data is None:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.copy()


def _download_stock_by_code(code, period="1y", interval="1d"):
    """
    先抓 .TW，抓不到再抓 .TWO。
    回傳 data, ticker_used。
    """
    ticker_tw = f"{code}.TW"
    data = yf.download(ticker_tw, period=period, interval=interval, progress=False)
    data = _clean_yfinance_data(data)

    if data.empty:
        ticker_two = f"{code}.TWO"
        data = yf.download(ticker_two, period=period, interval=interval, progress=False)
        data = _clean_yfinance_data(data)
        ticker_used = ticker_two
    else:
        ticker_used = ticker_tw

    return data, ticker_used


def _make_latest_daily_bar_from_intraday(intra_df):
    """
    將最近一個交易日的 1 分 K 合成一根日 K。
    這可以補上 yfinance 日K尚未更新的最新交易日。
    """
    if intra_df is None or intra_df.empty:
        return None, None

    d = _clean_yfinance_data(intra_df).dropna()
    if d.empty:
        return None, None

    # 只取 intraday 資料中「最後一個交易日」
    latest_date = d.index[-1].date()
    d_latest = d[d.index.date == latest_date].copy()

    if d_latest.empty:
        return None, None

    bar = {
        "Open": d_latest["Open"].iloc[0],
        "High": d_latest["High"].max(),
        "Low": d_latest["Low"].min(),
        "Close": d_latest["Close"].iloc[-1],
        "Volume": d_latest["Volume"].sum(),
    }

    # 如果日K有 Adj Close 欄位，也補上，避免 dropna 時最新列被刪掉
    if "Adj Close" in d_latest.columns:
        bar["Adj Close"] = d_latest["Adj Close"].iloc[-1]

    return pd.Timestamp(latest_date), bar


def _append_or_replace_latest_intraday_bar(daily_df, ticker_used):
    """
    用最近 1 分 K 合成的日K，補進日K資料。
    若日K已經有同一天，則用盤中合成資料覆蓋；
    若日K缺最新交易日，則新增一列。
    """
    if daily_df is None or daily_df.empty:
        return daily_df, False

    daily = _clean_yfinance_data(daily_df).copy()
    daily = daily.sort_index()

    intra = yf.download(ticker_used, period="5d", interval="1m", progress=False)
    intra = _clean_yfinance_data(intra)

    latest_index, latest_bar = _make_latest_daily_bar_from_intraday(intra)
    if latest_index is None or latest_bar is None:
        return daily, False

    last_daily_date = daily.index[-1].date()
    latest_intraday_date = latest_index.date()

    # 只在 intraday 日期 >= 日K最後日期時處理
    if latest_intraday_date < last_daily_date:
        return daily, False

    # 建立資料來源欄位，方便你在資料表確認最新一列是日K或盤中合成
    if "資料來源" not in daily.columns:
        daily["資料來源"] = "日K"

    # 確保新列欄位完整
    new_row = {}
    for col in daily.columns:
        if col in latest_bar:
            new_row[col] = latest_bar[col]
        elif col == "資料來源":
            new_row[col] = "1分K合成日K"
        elif col == "Adj Close" and "Close" in latest_bar:
            new_row[col] = latest_bar["Close"]
        else:
            new_row[col] = np.nan

    # 如果同一天已存在，覆蓋；如果日K還沒更新到最新交易日，新增
    same_day_mask = [idx.date() == latest_intraday_date for idx in daily.index]
    if any(same_day_mask):
        replace_idx = daily.index[same_day_mask.index(True)]
        for col, val in new_row.items():
            daily.loc[replace_idx, col] = val
    else:
        daily.loc[latest_index, list(new_row.keys())] = list(new_row.values())
        daily = daily.sort_index()

    return daily, True


def get_stock_data(code):
    """
    抓日K資料，並用最近 1 分 K 合成最新交易日的日K。

    為什麼要這樣做：
    yfinance 的日K有時候收盤後不會馬上更新，
    但 1 分 K通常比較快有最新交易日資料，
    所以這裡會把最近一個交易日的 1 分 K 合成一根日K補進資料表。
    """
    data, ticker_used = _download_stock_by_code(code, period="1y", interval="1d")

    if data.empty:
        return data, ticker_used

    data, patched = _append_or_replace_latest_intraday_bar(data, ticker_used)

    return data, ticker_used


# ===== 盤中即時資料 =====
def get_intraday_data(code):
    """
    抓最近 5 個交易日的 1 分 K 資料。
    注意：yfinance 不是券商等級即時報價，可能延遲。
    """
    data, ticker_used = _download_stock_by_code(code, period="5d", interval="1m")
    data = data.dropna()
    return data, ticker_used


def analyze_intraday_monitor(intra_df):
    """
    盤中監控：只分析最近一個交易日，回傳最新價、開盤價、最高、最低、漲跌幅、成交量、盤中均線與盤中狀態。
    """
    if intra_df is None or intra_df.empty:
        return None

    d = _clean_yfinance_data(intra_df).dropna()
    if d.empty:
        return None

    # period=5d 時，只取最後一個交易日，避免把多天成交量加在一起
    latest_date = d.index[-1].date()
    d = d[d.index.date == latest_date].copy()

    if d.empty:
        return None

    latest = d.iloc[-1]
    first = d.iloc[0]

    latest_price = latest["Close"]
    open_price = first["Open"]
    high_price = d["High"].max()
    low_price = d["Low"].min()
    total_volume = d["Volume"].sum() / 1000

    day_pct = (latest_price - open_price) / open_price * 100 if open_price != 0 else np.nan

    d["MA5_intraday"] = d["Close"].rolling(5).mean()
    d["MA20_intraday"] = d["Close"].rolling(20).mean()

    latest_ma5 = d["MA5_intraday"].iloc[-1]
    latest_ma20 = d["MA20_intraday"].iloc[-1]

    if pd.isna(latest_ma20):
        intraday_state = "資料不足"
    elif latest_price > latest_ma5 > latest_ma20:
        intraday_state = "盤中偏多"
    elif latest_price < latest_ma20:
        intraday_state = "盤中偏弱"
    else:
        intraday_state = "盤中震盪"

    return {
        "latest_price": latest_price,
        "open_price": open_price,
        "high_price": high_price,
        "low_price": low_price,
        "day_pct": day_pct,
        "total_volume": total_volume,
        "ma5": latest_ma5,
        "ma20": latest_ma20,
        "state": intraday_state,
        "data": d,
        "last_time": d.index[-1],
    }


# ===== 單根K線判斷 =====
def classify_kbar(row):
    """
    判斷單根K線型態：
    大紅K / 中紅K / 小紅K
    大黑K / 中黑K / 小黑K
    十字線 / T字線 / 倒T線
    長上影 / 長下影 / 紡錘線 / 一字線
    """
    o = row["Open"]
    h = row["High"]
    l = row["Low"]
    c = row["Close"]

    if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
        return pd.Series(["資料不足", "OHLC資料不足，無法判斷K線型態"])

    total_range = h - l

    if total_range == 0:
        return pd.Series(["一字線", "開高低收幾乎相同，可能為鎖漲停、鎖跌停或成交不活躍"])

    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    body_ratio = body / total_range
    upper_ratio = upper_shadow / total_range
    lower_ratio = lower_shadow / total_range

    if c > o:
        color = "紅K"
        direction_text = "收盤價高於開盤價，多方勝出"
    elif c < o:
        color = "黑K"
        direction_text = "收盤價低於開盤價，空方勝出"
    else:
        color = "平盤K"
        direction_text = "收盤價接近開盤價，多空拉鋸"

    # 波動極小
    if c != 0 and total_range / c < 0.002:
        return pd.Series(["一字線", "當日波動極小，可能為鎖價或成交不活躍"])

    # 十字線系列
    if body_ratio <= 0.08:
        if lower_ratio >= 0.60 and upper_ratio <= 0.20:
            return pd.Series(["T字線", "下影線長，代表低檔有買盤承接，但仍需搭配位置判斷"])
        elif upper_ratio >= 0.60 and lower_ratio <= 0.20:
            return pd.Series(["倒T線", "上影線長，代表上方賣壓明顯，短線追高需小心"])
        else:
            return pd.Series(["十字線", "實體很小，多空力道接近平衡，常見於轉折或盤整區"])

    # 長上影線
    if upper_ratio >= 0.55 and lower_ratio <= 0.20:
        return pd.Series([f"長上影{color}", f"{direction_text}，但上影線很長，代表上方賣壓偏重"])

    # 長下影線
    if lower_ratio >= 0.55 and upper_ratio <= 0.20:
        return pd.Series([f"長下影{color}", f"{direction_text}，但下影線很長，代表下方買盤承接明顯"])

    # 紡錘線
    if body_ratio <= 0.35 and upper_ratio >= 0.20 and lower_ratio >= 0.20:
        return pd.Series([f"紡錘{color}", f"{direction_text}，但上下影線都明顯，代表多空拉鋸"])

    # 實體K分類
    if body_ratio >= 0.70:
        return pd.Series([f"大{color}", f"{direction_text}，實體長，趨勢力道較明顯"])
    elif body_ratio >= 0.40:
        return pd.Series([f"中{color}", f"{direction_text}，實體中等，方向明確但力道普通"])
    else:
        return pd.Series([f"小{color}", f"{direction_text}，實體偏小，方向較不明確"])


# ===== 最近5根K線組合分析 =====
def analyze_5k_window(window):
    """
    分析最近 5 根 K 線的組合型態。
    輸入：5筆OHLCV資料
    輸出：5K型態、5K後續狀態、5K解讀
    """
    if len(window) < 5:
        return "資料不足", "資料不足", "K線數量不足 5 根，無法分析"

    o = window["Open"]
    h = window["High"]
    l = window["Low"]
    c = window["Close"]
    v = window["Volume"]

    if o.isna().any() or h.isna().any() or l.isna().any() or c.isna().any():
        return "資料不足", "資料不足", "OHLC資料不足，無法分析"

    total_range = h - l
    total_range = total_range.replace(0, np.nan)

    body = (c - o).abs()
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l

    upper_ratio = (upper_shadow / total_range).fillna(0)
    lower_ratio = (lower_shadow / total_range).fillna(0)
    body_ratio = (body / total_range).fillna(0)

    red_count = int((c > o).sum())
    black_count = int((c < o).sum())

    if c.iloc[0] == 0:
        close_change = 0
    else:
        close_change = (c.iloc[-1] - c.iloc[0]) / c.iloc[0]

    higher_highs = (h.diff().dropna() > 0).sum() >= 3
    higher_lows = (l.diff().dropna() > 0).sum() >= 3
    lower_highs = (h.diff().dropna() < 0).sum() >= 3
    lower_lows = (l.diff().dropna() < 0).sum() >= 3

    latest_close = c.iloc[-1]
    previous_4_high = h.iloc[:-1].max()
    previous_4_low = l.iloc[:-1].min()

    avg_vol_4 = v.iloc[:-1].mean()
    latest_vol = v.iloc[-1]

    if avg_vol_4 == 0 or pd.isna(avg_vol_4):
        vol_ratio = 1
    else:
        vol_ratio = latest_vol / avg_vol_4

    latest_break_up = latest_close > previous_4_high
    latest_break_down = latest_close < previous_4_low

    long_upper_count = int((upper_ratio >= 0.45).sum())
    long_lower_count = int((lower_ratio >= 0.45).sum())

    avg_body_ratio = body_ratio.mean()

    if c.iloc[0] == 0:
        five_day_range_ratio = 0
    else:
        five_day_range_ratio = (h.max() - l.min()) / c.iloc[0]

    # 1. 放量突破
    if latest_break_up and vol_ratio >= 1.3:
        return (
            "5K放量突破",
            "偏多轉強",
            "最新收盤價突破前 4 根 K 線高點，且成交量放大，代表短線買盤有轉強跡象。"
        )

    # 2. 放量跌破
    if latest_break_down and vol_ratio >= 1.3:
        return (
            "5K放量跌破",
            "偏空轉弱",
            "最新收盤價跌破前 4 根 K 線低點，且成交量放大，代表短線賣壓轉強。"
        )

    # 3. 五日急漲
    if close_change >= 0.08:
        return (
            "5K急漲過熱",
            "偏多但留意拉回",
            "最近 5 根 K 線漲幅超過 8%，短線強勢，但容易出現獲利了結或震盪。"
        )

    # 4. 五日急跌
    if close_change <= -0.08:
        return (
            "5K急跌超賣",
            "偏空但留意反彈",
            "最近 5 根 K 線跌幅超過 8%，短線偏弱，但若出現長下影或量縮，可能有反彈機會。"
        )

    # 5. 多方推動
    if red_count >= 4 and close_change > 0 and (higher_highs or higher_lows):
        return (
            "5K多方推動",
            "偏多續強",
            "最近 5 根 K 線紅 K 數量偏多，且高點或低點逐漸墊高，代表短線多方仍占優勢。"
        )

    # 6. 空方修正
    if black_count >= 4 and close_change < 0 and (lower_highs or lower_lows):
        return (
            "5K空方修正",
            "偏空續弱",
            "最近 5 根 K 線黑 K 數量偏多，且高點或低點逐漸下降，代表短線賣壓仍在。"
        )

    # 7. 上影線賣壓
    if long_upper_count >= 3 and close_change > 0:
        return (
            "5K上影賣壓",
            "偏多轉觀察",
            "最近 5 根 K 線有多根長上影線，代表上方賣壓偏重，短線追高風險增加。"
        )

    # 8. 下影線承接
    if long_lower_count >= 3 and close_change < 0:
        return (
            "5K下影承接",
            "止跌觀察",
            "最近 5 根 K 線有多根長下影線，代表下方有買盤承接，若搭配量縮或站回均線，可留意反彈。"
        )

    # 9. 收斂盤整
    if five_day_range_ratio <= 0.05 and avg_body_ratio <= 0.35:
        return (
            "5K收斂盤整",
            "等待突破方向",
            "最近 5 根 K 線波動收斂且實體偏小，代表多空拉鋸，後續需觀察突破或跌破。"
        )

    # 10. 一般偏多
    if close_change > 0 and red_count >= 3:
        return (
            "5K震盪偏多",
            "偏多觀察",
            "最近 5 根 K 線收盤價整體走高，紅 K 略多，短線偏多但還未形成強勢突破。"
        )

    # 11. 一般偏空
    if close_change < 0 and black_count >= 3:
        return (
            "5K震盪偏空",
            "偏空觀察",
            "最近 5 根 K 線收盤價整體走低，黑 K 略多，短線偏空但還未形成明確跌破。"
        )

    return (
        "5K盤整",
        "中性觀察",
        "最近 5 根 K 線沒有明確多空方向，建議搭配 MA5、MA10、MA20 與支撐壓力觀察。"
    )


def add_5k_analysis(df):
    """
    對整份資料逐列加入 5K 型態分析。
    """
    df = df.copy()

    patterns = []
    states = []
    comments = []

    for i in range(len(df)):
        if i < 4:
            patterns.append("資料不足")
            states.append("資料不足")
            comments.append("K線數量不足 5 根，無法分析")
        else:
            window = df.iloc[i - 4:i + 1]
            pattern, state, comment = analyze_5k_window(window)
            patterns.append(pattern)
            states.append(state)
            comments.append(comment)

    df["5K型態"] = patterns
    df["5K後續狀態"] = states
    df["5K解讀"] = comments

    return df


# ===== 量價分析 =====
def volume_analysis(row):
    if pd.isna(row["MV5"]) or pd.isna(row["MV20"]) or pd.isna(row["PrevClose"]):
        return pd.Series(["資料不足", "成交量資料不足"])

    close = row["Close"]
    prev_close = row["PrevClose"]
    volume = row["成交量_張"]
    mv20 = row["MV20"]

    if prev_close == 0:
        return pd.Series(["資料不足", "前一日收盤價異常，無法分析"])

    price_up = close > prev_close
    price_down = close < prev_close
    volume_up = volume > mv20 * 1.5
    volume_low = volume < mv20 * 0.8

    if price_up and volume_up:
        return pd.Series(["價漲量增", "多方買盤積極，若同時站上均線，偏多。"])
    elif price_up and volume_low:
        return pd.Series(["價漲量縮", "上漲但量能不足，追高要小心。"])
    elif price_down and volume_up:
        return pd.Series(["價跌量增", "賣壓放大，短線偏弱。"])
    elif price_down and volume_low:
        return pd.Series(["價跌量縮", "可能是正常回檔，觀察支撐是否守住。"])
    elif abs(close - prev_close) / prev_close < 0.01 and volume_low:
        return pd.Series(["量縮整理", "多空觀望，等待突破方向。"])
    else:
        return pd.Series(["量價普通", "成交量沒有明顯訊號。"])


# ===== 進場評分 =====
def entry_score(row):
    """
    進場時機評分：
    同時考慮「趨勢型進場」與「止跌反彈觀察」。

    這版加入緩衝邏輯：
    1. 小幅跌破 MA20 不直接重扣分。
    2. 價跌量增依照是否跌破 MA20 分級扣分。
    3. 小幅跌破支撐先列為觀察，不直接打到最低分。
    """

    score = 50
    reasons = []
    rebound_score = 0
    rebound_reasons = []

    close = row["Close"]
    ma5 = row["MA5"]
    ma10 = row["MA10"]
    ma20 = row["MA20"]

    if pd.isna(ma20):
        return pd.Series([0, "資料不足", "MA20資料不足，暫時不評估"])

    ma5_gap = (close - ma5) / ma5 if ma5 and not pd.isna(ma5) else 0
    ma20_gap = (close - ma20) / ma20 if ma20 and not pd.isna(ma20) else 0
    rsi = row["RSI14"] if "RSI14" in row and not pd.isna(row["RSI14"]) else np.nan

    # ===== 趨勢基礎：加入小幅跌破緩衝 =====
    trend_ok = False
    soft_trend_ok = False

    if close > ma5 > ma10 > ma20:
        score += 20
        trend_ok = True
        soft_trend_ok = True
        reasons.append("均線多頭排列")
    elif close > ma20 and ma5 > ma10:
        score += 10
        trend_ok = True
        soft_trend_ok = True
        reasons.append("股價站上MA20，短線均線偏多")
    elif close > ma20:
        score += 5
        trend_ok = True
        soft_trend_ok = True
        reasons.append("股價仍站上MA20")
    elif ma20_gap > -0.02:
        score -= 5
        soft_trend_ok = True
        reasons.append("股價小幅跌破MA20（2%內），先觀察是否為假跌破")
    else:
        score -= 20
        reasons.append("股價明顯跌破MA20，趨勢偏弱")

    # ===== 避免追高 =====
    if ma5_gap > 0.06:
        score -= 20
        reasons.append("股價高於MA5超過6%，短線有追高風險")
    elif ma5_gap > 0.035:
        score -= 10
        reasons.append("股價高於MA5較多，追高需小心")

    if ma20_gap > 0.15:
        score -= 20
        reasons.append("股價高於MA20超過15%，波段乖離過大")
    elif ma20_gap > 0.10:
        score -= 10
        reasons.append("股價高於MA20超過10%，已有一定漲幅")

    if row["5K後續狀態"] == "偏多但留意拉回":
        score -= 15
        reasons.append("5K急漲過熱，不適合追高")

    if row["5K型態"] == "5K上影賣壓":
        score -= 15
        reasons.append("近5K上影線偏多，上方賣壓較重")

    # ===== 健康回檔加分 =====
    if soft_trend_ok and row["波段方向"] == "修正波":
        score += 12
        reasons.append("趨勢尚未完全轉弱但進入修正波，列為回檔觀察")

    if row["量價型態"] == "價跌量縮" and soft_trend_ok:
        score += 15
        reasons.append("價跌量縮，可能是健康回檔")

    if row["5K後續狀態"] == "止跌觀察":
        score += 15
        reasons.append("5K出現下影承接，留意止跌")

    if "修正波第" in row["費波提醒"]:
        score += 10
        reasons.append(row["費波提醒"])

    # ===== 突破型進場 =====
    if row["5K後續狀態"] == "偏多轉強":
        if ma5_gap <= 0.06 and ma20_gap <= 0.15:
            score += 20
            reasons.append("5K放量突破且乖離未過大")
        else:
            score += 5
            reasons.append("5K放量突破，但乖離偏大，避免重倉追高")

    # ===== 量價訊號：價跌量增分級扣分 =====
    if row["量價型態"] == "價漲量增":
        if ma5_gap <= 0.05:
            score += 15
            reasons.append("價漲量增，且未明顯追高")
        else:
            score += 5
            reasons.append("價漲量增，但位置偏高")
    elif row["量價型態"] == "價漲量縮":
        score -= 10
        reasons.append("價漲量縮，上漲力道不足")
    elif row["量價型態"] == "價跌量增":
        if close < ma20 and ma20_gap <= -0.02:
            score -= 20
            reasons.append("價跌量增且明顯跌破MA20，賣壓放大")
        elif close < ma20:
            score -= 10
            reasons.append("價跌量增但僅小幅跌破MA20，先觀察是否止跌")
        else:
            score -= 10
            reasons.append("價跌量增，但尚未跌破MA20，先觀察")

    # ===== 跌破支撐扣分：小幅跌破先觀察 =====
    if not pd.isna(row["20日支撐"]):
        support_gap = (close - row["20日支撐"]) / row["20日支撐"]

        if support_gap < -0.02:
            score -= 20
            reasons.append("明顯跌破20日支撐，短線轉弱")
        elif support_gap < 0:
            score -= 10
            rebound_score += 5
            rebound_reasons.append("小幅跌破20日支撐，觀察是否快速站回")
            reasons.append("小幅跌破20日支撐，先觀察是否為假跌破")
        elif support_gap <= 0.03:
            score += 10
            rebound_score += 15
            reasons.append("接近20日支撐且未跌破，可觀察承接")
            rebound_reasons.append("接近20日支撐且尚未跌破")

    # ===== 止跌 / 反彈觀察分數 =====
    if not pd.isna(rsi):
        if rsi < 30:
            rebound_score += 25
            rebound_reasons.append("RSI低於30，短線跌深")
        elif rsi < 40:
            rebound_score += 15
            rebound_reasons.append("RSI低於40，短線偏弱但可能接近反彈區")

    if row["量價型態"] == "價跌量縮":
        rebound_score += 20
        rebound_reasons.append("價跌量縮，賣壓有縮小跡象")

    if "長下影" in row["K線型態"]:
        rebound_score += 20
        rebound_reasons.append("單日K線出現長下影，下方有承接")

    if row["5K後續狀態"] == "止跌觀察":
        rebound_score += 20
        rebound_reasons.append("5K出現下影承接")

    if "修正波第" in row["費波提醒"]:
        rebound_score += 10
        rebound_reasons.append(row["費波提醒"])

    score = max(0, min(100, score))
    rebound_score = max(0, min(100, rebound_score))

    # ===== 判斷是否為過熱，不是偏弱 =====
    is_overheated = False

    if ma5_gap > 0.035 or ma20_gap > 0.10:
        is_overheated = True

    if row["5K後續狀態"] == "偏多但留意拉回":
        is_overheated = True

    if "RSI14" in row and not pd.isna(row["RSI14"]) and row["RSI14"] >= 70:
        is_overheated = True

    # ===== 評估文字 =====
    if score >= 80:
        level = "可觀察進場，但仍需分批"
    elif score >= 65:
        level = "偏適合觀察進場"
    elif score >= 50:
        level = "中性，等更明確訊號"
    elif is_overheated and trend_ok:
        level = "偏多但過熱，等待回檔"
    elif rebound_score >= 60:
        level = "止跌觀察，可小部位留意"
        reasons.extend(rebound_reasons)
    elif rebound_score >= 40:
        level = "偏弱，但有反彈觀察訊號"
        reasons.extend(rebound_reasons)
    elif score >= 35:
        level = "偏弱，等待止跌"
    else:
        level = "不建議進場"

    reason_text = "、".join(reasons) if reasons else "目前沒有明顯訊號"
    return pd.Series([score, level, reason_text])


def score_to_entry_level(score, reason_text=""):
    """
    依照平滑後分數重新產生進場評估文字。
    """
    if score >= 80:
        return "可觀察進場，但仍需分批"
    elif score >= 65:
        return "偏適合觀察進場"
    elif score >= 50:
        return "中性，等更明確訊號"
    elif "反彈觀察" in str(reason_text) or "止跌" in str(reason_text) or "長下影" in str(reason_text):
        return "偏弱，但有反彈觀察訊號"
    elif score >= 35:
        return "偏弱，等待止跌"
    else:
        return "不建議進場"


def smooth_entry_scores(raw_scores, max_step=20):
    """
    避免進場分數因為單日小幅破線而暴衝暴跌。
    每一天分數最多只比前一天變動 max_step 分。
    """
    smoothed = []

    for value in raw_scores:
        if pd.isna(value):
            smoothed.append(value)
            continue

        value = float(value)
        if not smoothed or pd.isna(smoothed[-1]):
            smoothed.append(value)
            continue

        prev = float(smoothed[-1])
        upper = prev + max_step
        lower = prev - max_step
        smoothed.append(max(lower, min(upper, value)))

    return smoothed


# ===== 獲利調節 / 停損風險評分 =====
def holding_risk_scores(row, cost=0):
    """
    拆成兩個分數：
    1. 獲利調節分數：有賺時，判斷是否過熱、賣壓、該不該先賣一部分。
    2. 停損風險分數：虧損時，判斷是否跌破防守、是否需要減碼或停損。
    """
    close = row["Close"]
    ma5 = row["MA5"]
    ma10 = row["MA10"]
    ma20 = row["MA20"]

    profit_score = 0
    stop_score = 0
    profit_reasons = []
    stop_reasons = []

    if pd.isna(ma20):
        return pd.Series([0, "資料不足", "MA20資料不足", 0, "資料不足", "MA20資料不足", "資料不足"])

    ma5_gap = (close - ma5) / ma5 if ma5 and not pd.isna(ma5) else 0
    ma20_gap = (close - ma20) / ma20 if ma20 and not pd.isna(ma20) else 0
    profit_pct = (close - cost) / cost * 100 if cost > 0 else np.nan

    # ===== 獲利調節：只有有輸入成本且有獲利時才啟動 =====
    if cost > 0 and profit_pct > 0:
        if profit_pct >= 50:
            profit_score += 25
            profit_reasons.append("獲利超過50%，可考慮分批鎖利")
        elif profit_pct >= 30:
            profit_score += 18
            profit_reasons.append("獲利超過30%，可留意分批調節")
        elif profit_pct >= 15:
            profit_score += 12
            profit_reasons.append("獲利超過15%，可設定移動停利")
        elif profit_pct >= 8:
            profit_score += 6
            profit_reasons.append("已有小幅獲利，可觀察是否轉弱")

        if ma20_gap > 0.15:
            profit_score += 20
            profit_reasons.append("股價高於MA20超過15%，波段乖離偏大")
        elif ma20_gap > 0.10:
            profit_score += 10
            profit_reasons.append("股價高於MA20超過10%，可留意調節")

        if ma5_gap > 0.06:
            profit_score += 15
            profit_reasons.append("股價高於MA5超過6%，短線追高風險增加")

        if row["5K後續狀態"] == "偏多但留意拉回":
            profit_score += 15
            profit_reasons.append("5K急漲過熱，留意獲利了結")

        if "長上影" in row["K線型態"] or row["K線型態"] == "倒T線":
            profit_score += 15
            profit_reasons.append("K線出現上影賣壓")

        if row["量價型態"] == "價漲量縮":
            profit_score += 10
            profit_reasons.append("價漲量縮，上漲動能不足")

        if row["量價型態"] == "價跌量增":
            profit_score += 15
            profit_reasons.append("價跌量增，可能出現獲利了結賣壓")

        if not pd.isna(ma5) and close < ma5:
            profit_score += 10
            profit_reasons.append("股價跌破MA5，短線動能降溫")

        if not pd.isna(ma10) and close < ma10:
            profit_score += 15
            profit_reasons.append("股價跌破MA10，獲利部位可考慮保守")

        if close > ma5 > ma10 > ma20 and ma5_gap <= 0.06 and ma20_gap <= 0.15:
            profit_score -= 10
            profit_reasons.append("均線多頭且乖離未過大，不急著調節")
    else:
        if cost <= 0:
            profit_reasons.append("未輸入成本，無法判斷獲利調節")
        else:
            profit_reasons.append("目前沒有獲利，不屬於獲利調節情境")

    # ===== 停損風險：主要針對虧損或技術破線 =====
    if cost > 0 and profit_pct < 0:
        if profit_pct <= -20:
            stop_score += 30
            stop_reasons.append("虧損超過20%，需重新檢查持股理由")
        elif profit_pct <= -10:
            stop_score += 22
            stop_reasons.append("虧損超過10%，停損風險升高")
        elif profit_pct <= -5:
            stop_score += 12
            stop_reasons.append("虧損超過5%，進入停損警戒")
    elif cost <= 0:
        stop_reasons.append("未輸入成本，停損風險僅能看技術面")

    if close < ma20:
        stop_score += 20
        stop_reasons.append("股價跌破MA20，波段偏弱")

    if row["狀態"] == "轉弱":
        stop_score += 15
        stop_reasons.append("APP狀態判斷為轉弱")

    if row["波段方向"] == "修正波":
        stop_score += 8
        stop_reasons.append("MA5斜率向下，處於修正波")

    if row["量價型態"] == "價跌量增":
        stop_score += 20
        stop_reasons.append("價跌量增，賣壓放大")

    if row["5K型態"] == "5K放量跌破" or row["5K後續狀態"] == "偏空轉弱":
        stop_score += 20
        stop_reasons.append("5K放量跌破，短線賣壓轉強")

    if row["5K型態"] == "5K空方修正" or row["5K後續狀態"] == "偏空續弱":
        stop_score += 10
        stop_reasons.append("5K結構偏空")

    if not pd.isna(row["20日支撐"]) and close < row["20日支撐"]:
        stop_score += 20
        stop_reasons.append("跌破20日支撐，防守位置失守")

    # 跌深但有承接時，避免把「殺低」和「停損」混在一起
    rsi = row["RSI14"] if "RSI14" in row and not pd.isna(row["RSI14"]) else np.nan
    if not pd.isna(rsi) and rsi < 30:
        stop_score -= 8
        stop_reasons.append("RSI低於30，已偏跌深，避免情緒性殺低")

    if "長下影" in row["K線型態"] or row["5K後續狀態"] == "止跌觀察":
        stop_score -= 10
        stop_reasons.append("出現下影承接，可觀察是否止跌")

    profit_score = max(0, min(100, profit_score))
    stop_score = max(0, min(100, stop_score))

    if profit_score >= 80:
        profit_level = "獲利調節訊號高，可考慮分批賣出"
    elif profit_score >= 60:
        profit_level = "偏高，可考慮先賣一部分"
    elif profit_score >= 40:
        profit_level = "中等，設定移動停利"
    elif profit_score > 0:
        profit_level = "偏低，續抱觀察"
    else:
        profit_level = "無獲利調節訊號"

    if stop_score >= 80:
        stop_level = "停損風險高，需檢查是否減碼"
    elif stop_score >= 60:
        stop_level = "停損風險偏高，等待反彈或設停損線"
    elif stop_score >= 40:
        stop_level = "中等風險，觀察能否站回均線"
    elif stop_score > 0:
        stop_level = "風險偏低，續觀察"
    else:
        stop_level = "無明顯停損風險"

    if profit_score >= 60:
        holding_advice = "有獲利且調節訊號偏高，較適合分批賣出，不建議一次全賣。"
    elif stop_score >= 60:
        holding_advice = "這是停損風險，不是獲利賣出訊號；可等反彈或設定明確停損線。"
    elif profit_score >= 40:
        holding_advice = "有獲利但未明顯過熱，可續抱並設定移動停利。"
    elif stop_score >= 40:
        holding_advice = "技術面偏弱，先觀察能否站回MA5或MA10。"
    else:
        holding_advice = "目前沒有強烈調節或停損訊號，續抱觀察。"

    profit_reason_text = "、".join(profit_reasons) if profit_reasons else "目前沒有獲利調節訊號"
    stop_reason_text = "、".join(stop_reasons) if stop_reasons else "目前沒有停損風險訊號"

    return pd.Series([
        profit_score, profit_level, profit_reason_text,
        stop_score, stop_level, stop_reason_text,
        holding_advice
    ])

# ===== 分析函式 =====
def analyze_stock(df, cost):
    df = df.copy()
    df = df.dropna()

    # 成交量換算
    df["成交量_張"] = df["Volume"] / 1000
    df["MV5"] = df["成交量_張"].rolling(5).mean()
    df["MV20"] = df["成交量_張"].rolling(20).mean()
    df["PrevClose"] = df["Close"].shift(1)

    # 計算均線
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # RSI14：輔助判斷跌深反彈或短線過熱
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # 判斷單根K線型態
    df[["K線型態", "K線解讀"]] = df.apply(classify_kbar, axis=1)

    # 加入5根K線組合分析
    df = add_5k_analysis(df)

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

    # 量價分析
    df[["量價型態", "量價解讀"]] = df.apply(volume_analysis, axis=1)

    # 損益率
    if cost > 0:
        df["損益率"] = (df["Close"] - cost) / cost * 100
    else:
        df["損益率"] = np.nan

    # 操作提醒，只針對持股成本
    def action_reminder(row):
        close = row["Close"]

        if cost <= 0:
            return "未輸入成本，僅顯示進場評估"

        if close >= cost * 1.03:
            return "成本上方 3%，持股偏強"
        elif close >= cost:
            return "回到成本區"
        elif close <= cost * 0.95:
            return "跌破成本 5%，停損警戒"
        elif not pd.isna(row["MA20"]) and close < row["MA20"]:
            return "跌破 MA20，轉弱警戒"
        else:
            return "觀察"

    df["操作提醒"] = df.apply(action_reminder, axis=1)

    # 進場評估
    df[["進場分數", "進場評估", "評估原因"]] = df.apply(entry_score, axis=1)

    # 保留原始分數，再加入平滑緩衝，避免分數一天內大幅跳動
    df["原始進場分數"] = df["進場分數"]
    df["進場分數"] = smooth_entry_scores(df["原始進場分數"], max_step=20)
    df["進場評估"] = df.apply(lambda row: score_to_entry_level(row["進場分數"], row["評估原因"]), axis=1)
    df["分數緩衝說明"] = df.apply(
        lambda row: "已啟用每日最大20分變動緩衝" if abs(row["進場分數"] - row["原始進場分數"]) > 0.01 else "未觸發分數緩衝",
        axis=1
    )

    # 獲利調節 / 停損風險評估
    df[[
        "獲利調節分數", "獲利調節評估", "獲利調節原因",
        "停損風險分數", "停損風險評估", "停損風險原因",
        "持股建議"
    ]] = df.apply(lambda row: holding_risk_scores(row, cost), axis=1)

    return df


# ===== 主程式 =====
if stock_code:
    try:
        df, ticker_used = get_stock_data(stock_code)

        if df is None or df.empty:
            st.error("抓不到資料，請確認股票名稱或股票代號是否正確。")
        else:
            result = analyze_stock(df, cost)

            if result is None or result.empty:
                st.error("分析後沒有資料，可能是股價資料不足。")
            else:
                latest = result.iloc[-1]

                monitor = None
                intra_ticker = None
                if realtime_mode:
                    intra_df, intra_ticker = get_intraday_data(stock_code)
                    monitor = analyze_intraday_monitor(intra_df)

                display_price = monitor["latest_price"] if monitor is not None else latest["Close"]
                display_pnl = (display_price - cost) / cost * 100 if cost > 0 else np.nan

                st.subheader(f"股票：{stock_name} / {stock_code}，日K資料來源代號：{ticker_used}")
                st.caption(f"日K最後資料日期：{latest.name.date()}｜最新列資料來源：{latest['資料來源'] if '資料來源' in result.columns else '日K'}")

                tab1, tab2, tab3, tab4, tab5 = st.tabs(["總覽", "K線與均線", "5K與量價", "資料表", "即時監控"])

                with tab1:
                    st.markdown("## 總覽")

                    col1, col2, col3, col4 = st.columns(4)

                    price_label = "盤中即時價" if monitor is not None else "最新收盤價"
                    col1.metric(price_label, f"{display_price:.2f}")
                    col2.metric("日K狀態", latest["狀態"])
                    col3.metric("進場分數", f"{latest['進場分數']:.0f}")
                    col4.metric("進場評估", latest["進場評估"])

                    col5, col6, col7, col8 = st.columns(4)

                    col5.metric("MA5（日K）", f"{latest['MA5']:.2f}" if not pd.isna(latest["MA5"]) else "資料不足")
                    col6.metric("MA10（日K）", f"{latest['MA10']:.2f}" if not pd.isna(latest["MA10"]) else "資料不足")
                    col7.metric("MA20（日K）", f"{latest['MA20']:.2f}" if not pd.isna(latest["MA20"]) else "資料不足")

                    if cost > 0:
                        col8.metric("即時/最新損益率", f"{display_pnl:.2f}%")
                    else:
                        col8.metric("損益率", "未輸入成本")

                    col9, col10, col11 = st.columns(3)
                    col9.metric("獲利調節分數", f"{latest['獲利調節分數']:.0f}")
                    col10.metric("停損風險分數", f"{latest['停損風險分數']:.0f}")
                    col11.metric("RSI14", f"{latest['RSI14']:.2f}" if not pd.isna(latest["RSI14"]) else "資料不足")

                    st.markdown("### 進場評估原因")
                    st.info(latest["評估原因"])
                    st.caption(f"原始進場分數：{latest['原始進場分數']:.0f}｜{latest['分數緩衝說明']}")

                    st.markdown("### 持股建議")
                    st.warning(latest["持股建議"])

                    st.markdown("### 獲利調節原因")
                    st.info(latest["獲利調節原因"])

                    st.markdown("### 停損風險原因")
                    st.error(latest["停損風險原因"])

                    st.markdown("### 持股操作提醒")
                    st.write(latest["操作提醒"])

                    if latest["費波提醒"] != "":
                        st.warning(latest["費波提醒"])
                    else:
                        st.write("目前沒有費波轉折提醒。")

                    st.markdown("### 支撐與壓力")
                    st.write(f"20 日壓力：{latest['20日壓力']:.2f}" if not pd.isna(latest["20日壓力"]) else "20 日壓力：資料不足")
                    st.write(f"20 日支撐：{latest['20日支撐']:.2f}" if not pd.isna(latest["20日支撐"]) else "20 日支撐：資料不足")

                with tab2:
                    st.markdown("## K線與均線")

                    st.markdown("### 最新單根 K 線型態")
                    st.success(f"{latest['K線型態']}")
                    st.write(latest["K線解讀"])

                    st.markdown("### 收盤價與均線")
                    chart_data = result[["Close", "MA5", "MA10", "MA20"]].dropna()
                    st.line_chart(chart_data, use_container_width=True)

                with tab3:
                    st.markdown("## 5K 與量價")

                    st.markdown("### 最近 5 根 K 線綜合判斷")
                    st.warning(f"{latest['5K型態']}｜{latest['5K後續狀態']}")
                    st.write(latest["5K解讀"])

                    st.markdown("### 量價關係")
                    st.success(f"{latest['量價型態']}")
                    st.write(latest["量價解讀"])

                    st.markdown("### 成交量")
                    colv1, colv2, colv3 = st.columns(3)
                    colv1.metric("成交量", f"{latest['成交量_張']:.0f} 張")
                    colv2.metric("5日均量", f"{latest['MV5']:.0f} 張" if not pd.isna(latest["MV5"]) else "資料不足")
                    colv3.metric("20日均量", f"{latest['MV20']:.0f} 張" if not pd.isna(latest["MV20"]) else "資料不足")

                    volume_chart = result[["成交量_張", "MV5", "MV20"]].dropna()
                    st.line_chart(volume_chart, use_container_width=True)

                with tab4:
                    st.markdown("## 最近 30 筆資料")

                    show_cols = [
                        "Open", "High", "Low", "Close", "資料來源",
                        "Volume", "成交量_張", "MV5", "MV20",
                        "K線型態", "K線解讀",
                        "MA5", "MA10", "MA20", "RSI14",
                        "狀態", "波段方向", "波段天數", "費波提醒",
                        "5K型態", "5K後續狀態", "5K解讀",
                        "量價型態", "量價解讀",
                        "原始進場分數", "進場分數", "進場評估", "分數緩衝說明", "評估原因",
                        "獲利調節分數", "獲利調節評估", "獲利調節原因",
                        "停損風險分數", "停損風險評估", "停損風險原因", "持股建議",
                        "損益率", "操作提醒"
                    ]

                    existing_cols = [col for col in show_cols if col in result.columns]
                    st.dataframe(result[existing_cols].tail(30), use_container_width=True)

                with tab5:
                    st.markdown("## 即時監控")
                    st.caption("此功能使用 yfinance 1分鐘盤中資料，可能延遲；適合監控，不適合當成券商即時下單報價。")

                    if not realtime_mode:
                        st.info("尚未開啟盤中即時監控。請在上方勾選「開啟盤中即時監控（1分鐘資料）」。")
                    elif monitor is None:
                        st.error("抓不到盤中資料，可能是非交易時間、資料源延遲，或 yfinance 暫時無資料。")
                    else:
                        st.caption(f"盤中資料來源代號：{intra_ticker}；最後資料時間：{monitor['last_time']}；每 {refresh_seconds} 秒刷新一次")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("盤中最新價", f"{monitor['latest_price']:.2f}")
                        c2.metric("今日漲跌幅", f"{monitor['day_pct']:.2f}%")
                        c3.metric("盤中狀態", monitor["state"])
                        c4.metric("成交量", f"{monitor['total_volume']:.0f} 張")

                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("開盤價", f"{monitor['open_price']:.2f}")
                        c6.metric("最高價", f"{monitor['high_price']:.2f}")
                        c7.metric("最低價", f"{monitor['low_price']:.2f}")
                        c8.metric("盤中MA20", f"{monitor['ma20']:.2f}" if not pd.isna(monitor["ma20"]) else "資料不足")

                        st.markdown("### 盤中價格走勢")
                        chart = monitor["data"][["Close", "MA5_intraday", "MA20_intraday"]].dropna()
                        st.line_chart(chart, use_container_width=True)

    except Exception as e:
        st.error("程式執行時發生錯誤")
        st.exception(e)
