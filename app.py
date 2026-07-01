import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

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
}

st.title("股票趨勢分析 APP")
st.write("輸入台股名稱或代號，自動判斷均線、K線型態、5K結構、量價關係、推動波 / 修正波與進場評分。")

# ===== 使用者輸入 =====
stock_input = st.text_input("輸入股票名稱或代號，例如：信錦、台達電、1582、2308", value="信錦")

if stock_input in STOCK_MAP:
    stock_code = STOCK_MAP[stock_input]
    stock_name = stock_input
else:
    stock_code = stock_input.strip()
    stock_name = stock_input.strip()

use_cost = st.checkbox("我要輸入持股成本", value=False)

if use_cost:
    cost = st.number_input("輸入你的成本價", min_value=0.0, value=93.0, step=1.0)
else:
    cost = 0.0


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
    不是判斷股票強不強，而是判斷「現在這個位置適不適合進場」。

    核心邏輯：
    1. 趨勢不能太差
    2. 避免追高
    3. 偏好強勢股回檔、量縮整理、支撐承接、合理突破
    """

    score = 50
    reasons = []

    close = row["Close"]

    ma5 = row["MA5"]
    ma10 = row["MA10"]
    ma20 = row["MA20"]

    # ===== 資料不足 =====
    if pd.isna(ma20):
        return pd.Series([0, "資料不足", "MA20資料不足，暫時不評估"])

    # ===== 計算乖離 =====
    ma5_gap = (close - ma5) / ma5 if ma5 and not pd.isna(ma5) else 0
    ma20_gap = (close - ma20) / ma20 if ma20 and not pd.isna(ma20) else 0

    # ===== 先判斷趨勢基礎 =====
    trend_ok = False

    if close > ma5 > ma10 > ma20:
        score += 20
        trend_ok = True
        reasons.append("均線多頭排列")
    elif close > ma20 and ma5 > ma10:
        score += 10
        trend_ok = True
        reasons.append("股價站上MA20，短線均線偏多")
    elif close > ma20:
        score += 5
        trend_ok = True
        reasons.append("股價仍站上MA20")
    else:
        score -= 25
        reasons.append("股價跌破MA20，趨勢偏弱")

    # ===== 避免追高：漲太遠要扣分 =====
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

    # ===== 5K過熱扣分 =====
    if row["5K後續狀態"] == "偏多但留意拉回":
        score -= 15
        reasons.append("5K急漲過熱，不適合追高")

    if row["5K型態"] == "5K上影賣壓":
        score -= 15
        reasons.append("近5K上影線偏多，上方賣壓較重")

    # ===== 健康回檔加分 =====
    # 條件：趨勢還沒壞，MA5斜率修正，且沒有跌破MA20
    if trend_ok and row["波段方向"] == "修正波" and close >= ma20:
        score += 15
        reasons.append("趨勢未壞但進入修正波，可能是回檔觀察點")

    # 價跌量縮：可能是正常回檔
    if row["量價型態"] == "價跌量縮" and close >= ma20:
        score += 15
        reasons.append("價跌量縮，可能是健康回檔")

    # 下影承接：支撐有買盤
    if row["5K後續狀態"] == "止跌觀察":
        score += 15
        reasons.append("5K出現下影承接，留意止跌")

    # 修正波費波日數：3、5、8、13日
    if "修正波第" in row["費波提醒"]:
        score += 10
        reasons.append(row["費波提醒"])

    # ===== 突破型進場 =====
    # 突破可以加分，但如果已經過熱，不加太多
    if row["5K後續狀態"] == "偏多轉強":
        if ma5_gap <= 0.06 and ma20_gap <= 0.15:
            score += 20
            reasons.append("5K放量突破且乖離未過大")
        else:
            score += 5
            reasons.append("5K放量突破，但乖離偏大，避免重倉追高")

    # ===== 量價訊號 =====
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
        score -= 25
        reasons.append("價跌量增，賣壓放大")

    # ===== 跌破支撐扣分 =====
    if not pd.isna(row["20日支撐"]) and close < row["20日支撐"]:
        score -= 30
        reasons.append("跌破20日支撐，短線轉弱")

    # ===== 接近支撐但未跌破，加觀察分 =====
    if not pd.isna(row["20日支撐"]):
        support_gap = (close - row["20日支撐"]) / row["20日支撐"]
        if 0 <= support_gap <= 0.03 and close >= ma20:
            score += 10
            reasons.append("接近20日支撐且未跌破，可觀察承接")

    # ===== 分數限制 =====
    score = max(0, min(100, score))

    # ===== 評估文字 =====
    if score >= 80:
        level = "可觀察進場，但仍需分批"
    elif score >= 65:
        level = "偏適合觀察進場"
    elif score >= 50:
        level = "中性，等更明確訊號"
    elif score >= 35:
        level = "偏弱，不急著進場"
    else:
        level = "不建議進場"

    if len(reasons) == 0:
        reason_text = "目前沒有明顯訊號"
    else:
        reason_text = "、".join(reasons)

    return pd.Series([score, level, reason_text])

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

    return df


# ===== 主程式 =====
if stock_code:
    df, ticker_used = get_stock_data(stock_code)

    if df.empty:
        st.error("抓不到資料，請確認股票名稱或股票代號是否正確。")
    else:
        result = analyze_stock(df, cost)
        latest = result.iloc[-1]

        st.subheader(f"股票：{stock_name} / {stock_code}，資料來源代號：{ticker_used}")

        tab1, tab2, tab3, tab4 = st.tabs(["總覽", "K線與均線", "5K與量價", "資料表"])

        with tab1:
            st.markdown("## 總覽")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("最新收盤價", f"{latest['Close']:.2f}")
            col2.metric("狀態", latest["狀態"])
            col3.metric("進場分數", f"{latest['進場分數']:.0f}")
            col4.metric("進場評估", latest["進場評估"])

            col5, col6, col7, col8 = st.columns(4)

            col5.metric("MA5", f"{latest['MA5']:.2f}" if not pd.isna(latest["MA5"]) else "資料不足")
            col6.metric("MA10", f"{latest['MA10']:.2f}" if not pd.isna(latest["MA10"]) else "資料不足")
            col7.metric("MA20", f"{latest['MA20']:.2f}" if not pd.isna(latest["MA20"]) else "資料不足")

            if cost > 0:
                col8.metric("損益率", f"{latest['損益率']:.2f}%")
            else:
                col8.metric("損益率", "未輸入成本")

            st.markdown("### 進場評估原因")
            st.info(latest["評估原因"])

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
                "Open", "High", "Low", "Close",
                "Volume", "成交量_張", "MV5", "MV20",
                "K線型態", "K線解讀",
                "MA5", "MA10", "MA20",
                "狀態", "波段方向", "波段天數", "費波提醒",
                "5K型態", "5K後續狀態", "5K解讀",
                "量價型態", "量價解讀",
                "進場分數", "進場評估", "評估原因",
                "損益率", "操作提醒"
            ]

            st.dataframe(result[show_cols].tail(30), use_container_width=True)
