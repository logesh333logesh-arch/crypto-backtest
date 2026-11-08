#!/usr/bin/env python3
"""
BTC ALL-IN-ONE backtest (one file, one data download)  -  IST times, ~5 years of 1h candles

Runs 3 studies in order:
  1) Day-of-week stats + weekend directional strategies
  2) Weekend RANGE-trade backtest (4 variants, 2 cost levels)
  3) Weekend CONTAINMENT study (how far BTC travels Sat 06:00 -> Sun 22:00)

Run:  python btc_all_in_one.py           (real Binance data)
      python btc_all_in_one.py --demo    (fake data, code test only)
Optional Telegram: env TG_TOKEN, TG_CHAT
Needs: pip install pandas numpy requests
Outputs (CSV): dow_*.csv, strategies_*.csv, range_*.csv, contain_*.csv
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



LEVELS = [1, 1.5, 2, 3, 4, 5, 6]     # +/- % bands to test
HOURS = 40                            # Sat 06:00 -> Sun 22:00


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


def main_dow(df1h):
    d = to_daily(df1h)
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
def main_range(df):

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


def main_contain(df):

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




# ------------------------------------------------------------- RUN ALL
def main():
    demo = "--demo" in sys.argv
    df1h = demo_1h() if demo else fetch_1h()
    print(f"Candles: {len(df1h)}  {df1h.index[0].date()} -> {df1h.index[-1].date()}\n")
    for name, fn in (("1) DAY-OF-WEEK", lambda: main_dow(df1h)),
                     ("2) RANGE TRADE", lambda: main_range(df1h)),
                     ("3) CONTAINMENT", lambda: main_contain(df1h))):
        print("\n" + "#" * 70 + f"\n#  {name}\n" + "#" * 70 + "\n")
        try:
            fn()
        except Exception as e:
            print(f"!! {name} failed: {e}")


if __name__ == "__main__":
    main()
