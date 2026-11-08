#!/usr/bin/env python3
"""
BTC Weekend RANGE-trade backtest (1h candles, IST times, ~5 years)

Idea
  * Range  = High/Low from Fri 18:00 IST to Sat 06:00 IST
  * Trade window = Sat 06:00 -> Sun 22:00 IST (flat before CME open)
  * Buy limit near range low, sell limit near range high
  * Stop outside the range, target inside the range, max 2 trades / weekend
  * Intra-candle order is unknown -> conservative: stop is assumed hit before target

Run:  python btc_range_backtest.py          (real Binance data)
      python btc_range_backtest.py --demo   (fake data, code test only)
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

# ---- EDIT THESE to your real Delta Exchange costs (per side, as fraction) ----
COSTS = {
    "low_cost":  dict(maker=0.0002, taker=0.0005, slip=0.0003),
    "high_cost": dict(maker=0.0010, taker=0.0010, slip=0.0005),
}
RISK_PER_TRADE = 0.01   # 1% of capital risked per trade (for the equity curve)

BASE = dict(range_start_h=18, range_end_h=6, trade_end_h=22,
            entry=0.10, stop=0.15, target=0.60,
            min_w=1.0, max_w=6.0, skip_fri=None, max_trades=2)
VARIANTS = {
    "base":          BASE,
    "skip_big_fri":  {**BASE, "skip_fri": 3.0},
    "target_far":    {**BASE, "target": 0.90},
    "safer_entry":   {**BASE, "entry": 0.20, "stop": 0.20, "target": 0.50},
}


# ------------------------------------------------------------- DATA
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
    rng = np.random.default_rng(7)
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0035, len(idx))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, .0015, len(idx))))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, .0015, len(idx))))
    return pd.DataFrame(dict(Open=o, High=h, Low=l, Close=c, Volume=1.0), index=idx)


# ------------------------------------------------------------- SIMULATION
def simulate(df, p):
    """Returns raw trades (gross, before fees). One row per trade."""
    out = []
    H1 = pd.Timedelta(hours=1)
    fridays = df.index[(df.index.dayofweek == 4) & (df.index.hour == 0)]
    for fr in fridays:
        rs = fr + pd.Timedelta(hours=p["range_start_h"])
        re_ = fr + pd.Timedelta(hours=24 + p["range_end_h"])
        te = fr + pd.Timedelta(hours=48 + p["trade_end_h"])

        rng = df[(df.index >= rs) & (df.index < re_)]
        win = df[(df.index >= re_) & (df.index < te)]
        need_rng = (re_ - rs) / H1
        need_win = (te - re_) / H1
        if len(rng) < need_rng * 0.9 or len(win) < need_win * 0.9:
            continue

        if p["skip_fri"] is not None:
            fri = df[(df.index >= fr) & (df.index < fr + pd.Timedelta(hours=24))]
            if len(fri) and abs((fri["Close"].iloc[-1] / fri["Open"].iloc[0] - 1) * 100) > p["skip_fri"]:
                continue

        Hh, Ll = rng["High"].max(), rng["Low"].min()
        W = Hh - Ll
        w_pct = W / ((Hh + Ll) / 2) * 100
        if not (p["min_w"] <= w_pct <= p["max_w"]):
            continue

        le, se = Ll + p["entry"] * W, Hh - p["entry"] * W
        pos, n = None, 0
        for t, r in win.iterrows():
            o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
            if pos is None:
                if n >= p["max_trades"]:
                    break
                long_hit, short_hit = l <= le, h >= se
                if not (long_hit or short_hit):
                    continue
                side = "long" if (long_hit and not short_hit) or \
                                 (long_hit and short_hit and abs(o - le) <= abs(o - se)) else "short"
                if side == "long":
                    entry = min(o, le)
                    stop, target = Ll - p["stop"] * W, Ll + p["target"] * W
                else:
                    entry = max(o, se)
                    stop, target = Hh + p["stop"] * W, Hh - p["target"] * W
                pos = dict(side=side, entry=entry, stop=stop, target=target, t_in=t)
                n += 1
                # entry candle: only the stop is checked (conservative)
                if (side == "long" and l <= stop) or (side == "short" and h >= stop):
                    exit_p = min(stop, entry) if side == "long" else max(stop, entry)
                    out.append(close_trade(pos, exit_p, "stop", t))
                    pos = None
                continue

            # manage open position
            if pos["side"] == "long":
                if l <= pos["stop"]:
                    out.append(close_trade(pos, min(pos["stop"], o), "stop", t)); pos = None
                elif h >= pos["target"]:
                    out.append(close_trade(pos, max(pos["target"], o), "target", t)); pos = None
            else:
                if h >= pos["stop"]:
                    out.append(close_trade(pos, max(pos["stop"], o), "stop", t)); pos = None
                elif l <= pos["target"]:
                    out.append(close_trade(pos, min(pos["target"], o), "target", t)); pos = None

        if pos is not None:   # time exit at end of window
            out.append(close_trade(pos, win["Close"].iloc[-1], "time", win.index[-1]))
    return pd.DataFrame(out)


def close_trade(pos, exit_p, reason, t):
    e = pos["entry"]
    gross = (exit_p - e) / e if pos["side"] == "long" else (e - exit_p) / e
    risk = abs(e - pos["stop"]) / e
    return dict(t_in=pos["t_in"], t_out=t, side=pos["side"], entry=e, exit=exit_p,
                reason=reason, gross=gross, risk=max(risk, 1e-6))


def apply_costs(tr, c):
    if tr.empty:
        return tr
    t = tr.copy()
    exit_fee = np.where(t["reason"] == "target", c["maker"], c["taker"])
    slip = np.where(t["reason"] == "stop", c["slip"], 0.0)
    t["net"] = t["gross"] - c["maker"] - exit_fee - slip
    t["R"] = t["net"] / t["risk"]
    return t.sort_values("t_out").reset_index(drop=True)


# ------------------------------------------------------------- STATS
def stats(t):
    if t is None or len(t) < 5:
        return dict(trades=len(t) if t is not None else 0)
    R = t["R"]
    eq = (1 + RISK_PER_TRADE * R).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    gp, gl = R[R > 0].sum(), -R[R < 0].sum()
    return dict(trades=len(t), win_pct=(R > 0).mean() * 100, avg_R=R.mean(),
                PF=gp / gl if gl > 0 else np.inf, equity_pct=(eq.iloc[-1] - 1) * 100,
                maxDD_pct=dd, t_stat=R.mean() / R.std() * np.sqrt(len(R)) if R.std() > 0 else 0)


def tg(text):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data=dict(chat_id=chat, text=text[:4000]), timeout=30)


# ------------------------------------------------------------- MAIN
def main():
    demo = "--demo" in sys.argv
    df = demo_1h() if demo else fetch_1h()
    print(f"Candles: {len(df)}  {df.index[0].date()} -> {df.index[-1].date()}\n")

    rows, keep = [], {}
    for vn, p in VARIANTS.items():
        raw = simulate(df, p)
        if raw.empty:
            print(f"{vn}: no trades"); continue
        for cn, c in COSTS.items():
            t = apply_costs(raw, c)
            k = int(len(t) * 0.6)
            full, oos = stats(t), stats(t.iloc[k:])
            rows.append(dict(variant=vn, cost=cn, **full,
                             oos_avg_R=oos.get("avg_R", np.nan), oos_t=oos.get("t_stat", np.nan)))
            keep[(vn, cn)] = t
    res = pd.DataFrame(rows).round(2)
    print("=== RESULTS (risk 1% per trade for equity curve) ===")
    print(res.to_string(index=False), "\n")

    if ("base", "low_cost") in keep:
        t = keep[("base", "low_cost")]
        t["year"] = pd.to_datetime(t["t_out"]).dt.year
        print("=== BASE (low cost): BY YEAR ===")
        print(t.groupby("year")["R"].agg(trades="count", avg_R="mean", total_R="sum").round(2), "\n")
        print("=== BASE (low cost): BY SIDE ===")
        print(t.groupby("side")["R"].agg(trades="count", avg_R="mean").round(2), "\n")
        print("=== BASE (low cost): BY EXIT REASON ===")
        print(t.groupby("reason")["R"].agg(trades="count", avg_R="mean").round(2), "\n")
        t.to_csv("range_trades_base.csv", index=False)
    res.to_csv("range_results.csv", index=False)

    print("Worth a closer look ONLY if: avg_R > 0 in BOTH low_cost and high_cost, "
          "t_stat > 2, oos_avg_R > 0, and positive in most years.")
    tg("BTC weekend RANGE backtest\n\n" + res.to_string(index=False))


if __name__ == "__main__":
    main()
