#!/usr/bin/env python3
"""
BTC weekend CONTAINMENT study (1h candles, IST times, ~5 years)

Question: starting from Sat 06:00 IST price, how far did BTC travel
(up / down) before Sun 22:00 IST?  How often did it stay inside +/-X% ?
This tells you how far a short-option strike (iron condor etc.) must be
to survive most weekends.  It does NOT include option premiums.

Control group: Tue 06:00 -> Wed 22:00 (same 40 hours) to show weekend vs weekday.

Run:  python btc_weekend_containment.py          (real Binance data)
      python btc_weekend_containment.py --demo   (fake data, code test only)
Optional Telegram: env TG_TOKEN, TG_CHAT
Needs: pip install pandas numpy requests
"""
import os, sys, time
import numpy as np
import pandas as pd
import requests

SYMBOL = "BTCUSDT"
YEARS = 5
TZ = "Asia/Kolkata"
BASES = ["https://data-api.binance.vision", "https://api.binance.com"]
LEVELS = [1, 1.5, 2, 3, 4, 5, 6]     # +/- % bands to test
HOURS = 40                            # Sat 06:00 -> Sun 22:00


def fetch_1h(symbol=SYMBOL, years=YEARS):
    end = int(time.time() * 1000)
    start = end - int((years + 0.1) * 365.25 * 24 * 3600 * 1000)
    rows, base = [], None
    while start < end:
        data = None
        for b in ([base] if base else BASES):
            try:
                r = requests.get(f"{b}/api/v3/klines",
                                 params=dict(symbol=symbol, interval="1h",
                                             startTime=start, limit=1000), timeout=30)
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
    rng = np.random.default_rng(3)
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0035, len(idx))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, .0015, len(idx))))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, .0015, len(idx))))
    return pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=1.0), index=idx)


def windows(df, dow, start_h=6, hours=HOURS):
    rows = []
    for s in df.index[(df.index.dayofweek == dow) & (df.index.hour == start_h)]:
        w = df[(df.index >= s) & (df.index < s + pd.Timedelta(hours=hours))]
        if len(w) < hours * 0.9:
            continue
        ref = w["Open"].iloc[0]
        up = (w["High"].max() / ref - 1) * 100
        dn = (1 - w["Low"].min() / ref) * 100
        fin = (w["Close"].iloc[-1] / ref - 1) * 100
        rows.append(dict(start=s, year=s.year, up=up, dn=dn, max_exc=max(up, dn), fin=fin))
    return pd.DataFrame(rows)


def containment(w):
    out = []
    for x in LEVELS:
        out.append(dict(band_pct=x,
                        stayed_inside_pct=(w["max_exc"] <= x).mean() * 100,
                        up_breach_pct=(w["up"] > x).mean() * 100,
                        down_breach_pct=(w["dn"] > x).mean() * 100,
                        ended_outside_pct=(w["fin"].abs() > x).mean() * 100))
    return pd.DataFrame(out).round(1)


def percentiles(w):
    q = [50, 75, 90, 95, 99]
    return pd.DataFrame({
        "max_travel_pct": np.percentile(w["max_exc"], q),
        "abs_final_move_pct": np.percentile(w["fin"].abs(), q)}, index=[f"p{k}" for k in q]).round(2)


def tg(text):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data=dict(chat_id=chat, text=text[:4000]), timeout=30)


def main():
    demo = "--demo" in sys.argv
    df = demo_1h() if demo else fetch_1h()
    print(f"Candles: {len(df)}  {df.index[0].date()} -> {df.index[-1].date()}\n")

    we = windows(df, dow=5)       # Saturday 06:00
    wd = windows(df, dow=1)       # Tuesday 06:00 (control)
    last = we.tail(52)
    print(f"Weekend windows: {len(we)}   Weekday control windows: {len(wd)}\n")

    c_all, c_last, c_wd = containment(we), containment(last), containment(wd)
    print("=== WEEKEND (Sat 06:00 -> Sun 22:00 IST), all years ===\n", c_all.to_string(index=False), "\n")
    print("=== WEEKEND, last 52 weeks only ===\n", c_last.to_string(index=False), "\n")
    print("=== CONTROL (Tue 06:00 -> Wed 22:00 IST) ===\n", c_wd.to_string(index=False), "\n")
    print("=== WEEKEND percentiles (% travel from Sat 06:00 price) ===\n", percentiles(we), "\n")
    print("=== CONTROL percentiles ===\n", percentiles(wd), "\n")

    yr = pd.DataFrame({f"inside_{x}pct": we.groupby("year")["max_exc"].apply(lambda s, x=x: (s <= x).mean() * 100)
                       for x in (2, 3, 4)}).round(1)
    yr["weeks"] = we.groupby("year")["max_exc"].count()
    print("=== WEEKEND: % of weeks inside +/-2/3/4% BY YEAR ===\n", yr, "\n")

    print("How to read: if the +/-3% band was breached in 20% of weekends, a short strike "
          "at +/-3% loses on ~1 weekend in 5, so the premium you collect must pay for that. "
          "Check the last-52-weeks table too; recent behaviour matters more than 2021.")

    c_all.to_csv("contain_weekend_all.csv", index=False)
    c_last.to_csv("contain_weekend_last52.csv", index=False)
    c_wd.to_csv("contain_weekday_control.csv", index=False)
    yr.to_csv("contain_by_year.csv")
    we.to_csv("contain_weekend_windows.csv", index=False)

    tg("BTC weekend containment\n\nALL YEARS\n" + c_all.to_string(index=False) +
       "\n\nLAST 52 WEEKS\n" + c_last.to_string(index=False) +
       "\n\nWEEKDAY CONTROL\n" + c_wd.to_string(index=False))


if __name__ == "__main__":
    main()
