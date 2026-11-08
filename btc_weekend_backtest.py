#!/usr/bin/env python3
"""
BTC Day-of-Week + Weekend Backtest (5 years, IST days)
Run:   python btc_weekend_backtest.py            (real Binance data)
       python btc_weekend_backtest.py --demo     (fake data, just to test the code)
Optional Telegram: set env TG_TOKEN and TG_CHAT
Needs: pip install pandas numpy requests
"""
import os, sys, time
import numpy as np
import pandas as pd
import requests

SYMBOL = "BTCUSDT"
YEARS = 5
FEE = 0.001        # 0.1% per side
SLIP = 0.0005      # 0.05% per side
COST = 2 * (FEE + SLIP)   # round trip
TZ = "Asia/Kolkata"
BASES = ["https://data-api.binance.vision", "https://api.binance.com"]


# ---------------------------------------------------------- 1. DATA
def fetch_1h(symbol=SYMBOL, years=YEARS):
    end = int(time.time() * 1000)
    start = end - int((years + 0.1) * 365.25 * 24 * 3600 * 1000)
    rows = []
    base = None
    while start < end:
        data = None
        for b in ([base] if base else BASES):
            try:
                r = requests.get(f"{b}/api/v3/klines",
                                 params=dict(symbol=symbol, interval="1h",
                                             startTime=start, limit=1000),
                                 timeout=30)
                if r.status_code == 200:
                    data, base = r.json(), b
                    break
            except Exception:
                pass
        if data is None:
            raise RuntimeError("Binance data fetch failed")
        if not data:
            break
        rows += data
        start = data[-1][0] + 3600 * 1000
        if len(data) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume",
                                     "ct", "qv", "n", "tb", "tq", "ig"])
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ)
    df = df.set_index("t")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df[~df.index.duplicated()]


def demo_1h(years=YEARS):
    idx = pd.date_range(end=pd.Timestamp.now(tz=TZ).floor("h"),
                        periods=int(years * 365 * 24), freq="h")
    rng = np.random.default_rng(1)
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, .002, len(idx))))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, .002, len(idx))))
    return pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=1.0), index=idx)


def to_daily(df1h):
    d = df1h.resample("1D").agg(dict(Open="first", High="max", Low="min",
                                     Close="last", Volume="sum")).dropna()
    d = d.iloc[:-1]  # drop today's incomplete candle
    d["ret"] = (d["Close"] / d["Open"] - 1) * 100
    d["range"] = (d["High"] - d["Low"]) / d["Open"] * 100
    d["day"] = d.index.day_name()
    d["dow"] = d.index.dayofweek
    d["year"] = d.index.year
    return d


# ---------------------------------------------------------- 2. DAY-OF-WEEK STATS
def dow_stats(d):
    g = d.groupby("dow")
    out = pd.DataFrame(dict(
        day=g["day"].first(),
        days=g["ret"].count(),
        avg_ret=g["ret"].mean(),
        median_ret=g["ret"].median(),
        win_pct=g["ret"].apply(lambda x: (x > 0).mean() * 100),
        avg_range=g["range"].mean(),
        avg_volume=g["Volume"].mean(),
    )).round(3)
    by_year = d.pivot_table(index="dow", columns="year", values="ret", aggfunc="mean").round(3)
    by_year.index = out["day"]
    return out.set_index("day"), by_year


# ---------------------------------------------------------- 3. STRATEGIES (weekly trades)
def build_weeks(d):
    nxt_close = d["Close"].shift(-1)
    recs = []
    for _, g in d.groupby(d.index.tz_localize(None).to_period("W-SUN")):
        rows = {x.dayofweek: r for x, r in g.iterrows()}
        if not all(k in rows for k in (4, 5, 6)):
            continue
        fri, sat, sun = rows[4], rows[5], rows[6]
        mon_close = nxt_close.loc[sun.name]
        if np.isnan(mon_close):
            continue
        fri_up = fri.Close > fri.Open
        wk = sun.Close / sat.Open - 1                    # Sat open -> Sun close
        recs.append(dict(
            date=sun.name,
            weekend_long=wk,
            weekend_short=-wk,
            fri_to_sun_long=sun.Close / fri.Open - 1,
            fri_mom=wk if fri_up else -wk,               # follow Friday direction
            fri_rev=-wk if fri_up else wk,               # against Friday direction
            sat_only_long=sat.Close / sat.Open - 1,
            sun_to_mon_long=mon_close / sun.Close - 1,
            fri_close_to_mon_close=mon_close / fri.Close - 1,
        ))
    w = pd.DataFrame(recs).set_index("date")
    return w - COST        # fees + slippage on every trade


def perf(s):
    s = s.dropna()
    if len(s) < 5:
        return {}
    eq = (1 + s).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else 0
    return dict(trades=len(s), avg_pct=s.mean() * 100, win_pct=(s > 0).mean() * 100,
                total_pct=(eq.iloc[-1] - 1) * 100, max_dd_pct=dd * 100, t_stat=t)


def strategy_table(w):
    return pd.DataFrame({c: perf(w[c]) for c in w.columns}).T.round(2)


def split_test(w, frac=0.6):
    k = int(len(w) * frac)
    return strategy_table(w.iloc[:k]), strategy_table(w.iloc[k:])


# ---------------------------------------------------------- 4. MAIN
def tg(text):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data=dict(chat_id=chat, text=text[:4000]), timeout=30)


def main():
    demo = "--demo" in sys.argv
    d = to_daily(demo_1h() if demo else fetch_1h())
    print(f"Days: {len(d)}  {d.index[0].date()} -> {d.index[-1].date()}  (IST days)\n")

    stats, by_year = dow_stats(d)
    print("=== DAY-OF-WEEK STATS ===\n", stats, "\n")
    print("=== AVG RETURN % BY YEAR ===\n", by_year, "\n")

    w = build_weeks(d)
    full = strategy_table(w)
    ins, oos = split_test(w)
    print(f"=== STRATEGIES (after {COST*100:.2f}% round-trip cost) ===\n", full, "\n")
    print("=== FIRST 60% (in-sample) ===\n", ins, "\n")
    print("=== LAST 40% (out-of-sample) ===\n", oos, "\n")
    print("Rule of thumb: t_stat > 2 AND positive in both in-sample and out-of-sample = worth a closer look.")

    stats.to_csv("dow_stats.csv"); by_year.to_csv("dow_by_year.csv")
    full.to_csv("strategies_full.csv"); ins.to_csv("strategies_in.csv"); oos.to_csv("strategies_out.csv")

    tg("BTC weekend backtest\n\n" + stats.to_string() + "\n\n" + full.to_string())


if __name__ == "__main__":
    main()
