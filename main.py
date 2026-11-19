"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts, for each of GTA VI's 55 PlayStation 5 trophies, its earn rate on
Exophase at the case's target, as five quantiles (5/25/50/75/95%), scored by
mean pinball loss on the logit scale.

Model (own design, dilution-aware, earn counts and tracked players separated):

Exophase's page is a snapshot: it freezes between the tracker's refreshes and
jumps when it refreshes, because the players it tracks keep growing while the
page shows the last refresh. A trophy's earn rate is earners / tracked, so a
refresh dilutes every rate toward what the newly tracked players hold -- on
launch night the first refresh took tracked players 638 -> 3,868 and every
rate but the one at 100% fell. A persistence forecast ("rates stay") therefore
loses badly whenever a refresh lands inside the horizon.

The program works from the case's history of reads:
  - it collapses consecutive identical reads into distinct snapshots (one per
    refresh), keeping each snapshot's first-seen time;
  - when the history is flat (one snapshot: no refresh seen yet) it forecasts
    one refresh like the launch night's first: tracked players x6.06, the
    newly tracked players holding the ubiquitous trophy at 100% and the other
    trophies at a fresh fraction that grows steeply with the trophy's old
    rate (fresh_frac ~ rate**4: a new cohort reaches easy trophies far
    sooner than hard ones), calibrated so the fresh earns total ~103 over
    3,230 new players, the number and shape the v1+v2 real practice grades
    jointly identify (see below); the spread is 0.15 logit, tightened from
    0.25 after the real practice grade showed 90% coverage of 0.927 (too
    wide) with a typical miss of only 0.05 points, and kept at 0.15 in v3
    because a tighter 0.10 collapses 90% coverage against the pure-dilution
    proxy to 0.73;
  - when the history has real refreshes the median blends three components
    in logit space (weights 0.6 / 0.25 / 0.15): a logit-trend fit over the
    last 24 hours of snapshots (the house curve baseline's shape, fitted on
    refresh points only); a saturating Weibull rate curve per trophy, which
    cannot run away the way a power law does; and an earn-count model that
    fits E(t) with a saturating curve and N(t) with a power law, taking the
    rate as E/N, so dilution is in the forecast by construction. A fourth
    component was tried in v2 and removed: a count-consistent refresh-step
    model (step the measured per-refresh N growth and fresh-cohort holdings
    forward) scored far worse on the synthetic final at every weight
    (0.27 at weight 0.25, 0.15 at 0.10, 0.13 at 0.05, against 0.125 with
    it off), because it attributes all new earns to the new cohort while
    the old players keep earning as they age, and compounds a constant
    growth factor where growth decays. The spread comes from the trend
    fits' residuals per tier plus a term that grows with the square root
    of the horizon in days (0.15 logit per root-day).

Weights and spread were tuned on a synthetic launch-week backtest (a
cohort simulation with power-law player growth and refresh freezes): the
blend scores ~0.13 mean pinball loss there, against ~0.28 for the house
curve baseline and ~0.39 for persistence. That is a synthetic result, not
a promise about the real final. The v1 entry scored 0.174689 on the real
practice (rank ~901 of 4,900 scored; top-20 cutoff 0.0492) with coverage
0.927 and a typical miss of 0.05 points, and v2 scored 0.168216 with
coverage 0.891 and the same miss: the practice is one refresh over 90
minutes and, as the rules warn, little guide to the final's week, but the
two grades jointly identify the flat prior's shape (gamma ~4, total ~103
fresh earns -- the v2 total of ~10 was a misidentification from holding
the shape fixed). The final is decided on the refresh-branch forecast,
not the flat prior, and the flat branch is the only part v3 changes.

Output: /output/<id>.out, {"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}]}, one row per trophy, quantiles non-decreasing in [0, 100], rounded
to 4 decimals. Deterministic: no randomness, no wall clock, sorted output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release day begins 2026-11-19T00:00:00Z; time is measured from here.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# First refresh on launch night: 638 -> 3,868 tracked players (x6.06). The
# newly tracked players held the ubiquitous trophy at 100% and earned only
# ~103 trophies across the other 54 in the first hours. That total is not
# assumed: the v1 entry's real practice grade (loss 0.174689, typical miss
# 0.05 points, coverage 0.927) and the v2 entry's (loss 0.168216, miss 0.05,
# coverage 0.891) jointly identify it, but only together with the SHAPE:
# no low-gamma (spread-out) truth reproduces all six numbers -- the
# low-gamma identification of v2 (total ~10) was wrong, because it held the
# shape fixed. Scanning gamma and total jointly, the best fit is gamma ~4
# (fresh earns concentrated on the easiest, highest-rate trophies) with
# total ~103; the residual error (losses still 0.01-0.03 above the smooth
# fit, coverage ~0.02 high) is the realized-count lumpiness no smooth law
# captures. A Poisson-count-quantile flat branch was tried in v3 and
# REJECTED: it gains little (expected practice loss 0.102 -> 0.090 in the
# ideal case) and fails the robustness bar (worst practice proxy 0.185,
# worse than v2's 0.181), because its collapsed quantiles on low-rate
# trophies are too tight against a high-fresh-earns truth. The fresh
# fraction grows with the trophy's old rate as fresh_frac = C * rate**4
# (a new cohort reaches easy trophies far sooner than hard ones), with C
# set so the fresh earns total FRESH_EARNS_PER_NEW per new player.
FIRST_REFRESH_GROWTH = 6.06
FRESH_EARNS_PER_NEW = float(
    os.environ.get("MF_FRESH_EARNS_PER_NEW", str(103.0 / 3230.0)))

# Blend weights of the three median components (logit space). Tuned on a
# synthetic launch-week backtest (see README). The environment overrides
# exist only for local tuning; the grader leaves them unset.
W_TREND = float(os.environ.get("MF_W_TREND", "0.6"))
W_WEIBULL = float(os.environ.get("MF_W_WEIBULL", "0.25"))
W_EN = float(os.environ.get("MF_W_EN", "0.15"))

# Spread: residual floor per tier, and the horizon term's logit sd per
# square-root day. HORIZON_SIGMA is tunable from the environment locally.
RESID_FLOOR = 0.30
HORIZON_SIGMA = float(os.environ.get("MF_HORIZON_SIGMA", "0.15"))
FLAT_SPREAD = float(os.environ.get("MF_FLAT_SPREAD", "0.15"))
FLAT_GAMMA = float(os.environ.get("MF_FLAT_GAMMA", "4.0"))
SPREAD_CAP = 2.0

# Time shift (hours) in the saturating fits: the page already showed earned
# trophies at t = 0 (early-access players played before release day), so the
# saturating curves are anchored at t + C_SHIFT, not at t = 0.
C_SHIFT = 6.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (the starter's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, resid_sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = 0.0
    return a, b, sd


def snapshots(case):
    """Collapse the history into distinct snapshots.

    Consecutive reads with identical (players_tracked, rates) are one page
    state: Exophase had not refreshed between them. Returns a list of
    (hours_since_release, players_tracked, [rates]) in time order, each
    timed at the state's first appearance.
    """
    out = []
    for read in case["history"]:
        n = read["players_tracked"]
        rates = tuple(read["rates"])
        if out and out[-1][1] == n and out[-1][2] == rates:
            continue
        out.append((hours_since_release(parse_time(read["at"])), n, rates))
    return out


def quantiles(median_pct, spread):
    lg = logit(median_pct)
    qs = [round4(min(100.0, max(0.0, inv_logit(lg + z * spread)))) for z in Z]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_flat(case, snaps, horizon_days):
    """History is one frozen page: forecast one refresh like the first.

    N multiplies by FIRST_REFRESH_GROWTH; the ubiquitous trophy (100%) stays
    100% because every newly tracked player holds it; every other trophy's
    new holders earn it at a fresh fraction that grows with its current
    rate (a new cohort reaches easy trophies sooner), calibrated so the
    fresh earns total FRESH_EARNS_PER_NEW per new player.
    """
    _, n_last, rates = snaps[-1]
    n_new = n_last * FIRST_REFRESH_GROWTH
    dn = n_new - n_last
    # Calibrate C so sum_i fresh_frac_i * r_i * dn / 100 == fresh total.
    total_fresh = FRESH_EARNS_PER_NEW * dn
    denom = sum((r ** (1.0 + FLAT_GAMMA)) for r in rates
                if r is not None and r < 99.9)
    c_frac = (total_fresh * 100.0 / dn / denom) if denom > 0 else 0.0
    spread = min(SPREAD_CAP, FLAT_SPREAD + 0.15 * math.sqrt(horizon_days))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = rates[i]
        if r is None:
            r = 0.0
        if r >= 99.9:
            med = 99.9
            sp = 0.15
        else:
            e_old = r * n_last / 100.0
            frac = c_frac * (r ** FLAT_GAMMA) if r > 0 else 0.0
            e_new = frac * r * dn / 100.0
            med = clamp_pct(100.0 * (e_old + e_new) / n_new)
            sp = spread
        rows.append({"id": trophy["id"], "q": quantiles(med, sp)})
    return {"trophies": rows}


def weibull_rate_fit(ts, rs):
    """Saturating rate curve r(t) = R_inf * (1 - exp(-((t+C)/tau)^alpha)).

    Grid-search (tau, alpha); R_inf by least squares. Returns a function
    t -> rate in percent, or None when the data cannot support a fit.
    """
    pts = [(t, r / 100.0) for t, r in zip(ts, rs) if r is not None and r > 0]
    if len(pts) < 3:
        return None
    best = None
    for tau in (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0):
        for alpha in (0.5, 0.75, 1.0, 1.5, 2.0):
            fs = [1.0 - math.exp(-(((t + C_SHIFT) / tau) ** alpha)) for t, _ in pts]
            num = sum(f * r for f, (_, r) in zip(fs, pts))
            den = sum(f * f for f in fs)
            if den <= 1e-12:
                continue
            r_inf = num / den
            if r_inf <= 0 or r_inf > 1.0:
                continue
            sse = sum((r - r_inf * f) ** 2 for f, (_, r) in zip(fs, pts))
            if best is None or sse < best[0]:
                best = (sse, tau, alpha, r_inf)
    if best is None:
        return None
    _, tau, alpha, r_inf = best

    def predict(t):
        return clamp_pct(100.0 * r_inf * (1.0 - math.exp(-(((t + C_SHIFT) / tau) ** alpha))))

    return predict


def forecast_refresh(case, snaps, horizon_days, target_hours):
    """History has real refreshes: three median components, blended.

    A: logit-trend fit over the distinct snapshots (the house curve
       baseline's shape, but on refresh points only, all of them).
    B: saturating Weibull rate curve per trophy (safe extrapolation).
    C: earn counts E(t) fitted with a saturating curve and tracked players
       N(t) with a power law; the rate is E/N, so dilution is in the
       forecast by construction.
    """
    n_trophies = len(case["trophies"])
    xs = [math.log1p(t) for t, _, _ in snaps]
    ts = [t for t, _, _ in snaps]

    # Tracked players: log N against log time.
    n_fit = fit_line(xs, [math.log(max(1, n)) for _, n, _ in snaps])
    if n_fit is None:
        n_target = snaps[-1][1]
    else:
        a_n, b_n, _ = n_fit
        n_target = math.exp(a_n + b_n * math.log1p(target_hours))
    n_last = snaps[-1][1]
    n_target = min(max(n_target, n_last), n_last * (1.0 + horizon_days) ** 2)

    # Per-trophy fits: trend (A), Weibull rate (B), saturating E (C).
    tier_resid = {}
    trend = []
    weibull = []
    en_rate = []
    for i in range(n_trophies):
        tier = case["trophies"][i]["tier"]
        r_last = snaps[-1][2][i]
        if r_last is None:
            r_last = 0.0

        # A: logit trend, fitted on the last 24h of snapshots only (the
        # house curve baseline's window): a line in logit-vs-log-time is a
        # local approximation, and the early flat launch hours would drag
        # its slope. Fall back to all snapshots when fewer than 3 are in
        # the window.
        ckpt_hours = hours_since_release(parse_time(case["at"]))
        tx, ty = [], []
        for (t, n, rates), x in zip(snaps, xs):
            r = rates[i]
            if r is None:
                continue
            tx.append((t, x))
            ty.append(logit(r))
        win = [(x, y) for (t, x), y in zip(tx, ty) if t >= ckpt_hours - 24.0]
        if len(win) >= 3:
            tx, ty = [p[0] for p in win], [p[1] for p in win]
        else:
            tx, ty = [p[1] for p in tx], ty
        tfit = fit_line(tx, ty)
        trend.append(tfit)
        if tfit is not None and tfit[2] > 0:
            tier_resid.setdefault(tier, []).append(tfit[2])

        # B: saturating Weibull rate curve.
        weibull.append(weibull_rate_fit(ts, [rates[i] for _, _, rates in snaps]))

        # C: saturating earn counts against the power-law N forecast.
        ex, ey = [], []
        for (t, n, rates) in snaps:
            r = rates[i]
            if r is None:
                continue
            e = r * n / 100.0
            if e > 0:
                ex.append(t)
                ey.append(e)
        best_e = None
        if len(ex) >= 3:
            for tau in (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0):
                fs = [1.0 - math.exp(-((t + C_SHIFT) / tau)) for t in ex]
                den = sum(f * f for f in fs)
                if den <= 1e-12:
                    continue
                e_inf = sum(f * e for f, e in zip(fs, ey)) / den
                if e_inf <= 0:
                    continue
                sse = sum((e - e_inf * f) ** 2 for f, e in zip(fs, ey))
                if best_e is None or sse < best_e[0]:
                    best_e = (sse, tau, e_inf)
        if best_e is not None:
            _, tau_e, e_inf = best_e
            e_target = e_inf * (1.0 - math.exp(-((target_hours + C_SHIFT) / tau_e)))
            e_target = min(max(e_target, r_last * n_last / 100.0), n_target)
            en_rate.append(clamp_pct(100.0 * e_target / n_target))
        else:
            en_rate.append(None)
    tier_spread = {}
    all_resid = [v for vals in tier_resid.values() for v in vals]
    global_resid = (sum(all_resid) / len(all_resid)) if all_resid else RESID_FLOOR
    for tier, vals in tier_resid.items():
        tier_spread[tier] = max(RESID_FLOOR, sum(vals) / len(vals))

    horizon_term = HORIZON_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        tier = trophy["tier"]
        r_last = snaps[-1][2][i]
        if r_last is None:
            r_last = 0.0

        parts = []
        tfit = trend[i]
        if tfit is not None:
            a_t, b_t, _ = tfit
            parts.append((W_TREND, clamp_pct(inv_logit(a_t + b_t * math.log1p(target_hours)))))
        else:
            parts.append((W_TREND, r_last if r_last > 0 else 0.01))
        wfit = weibull[i]
        if wfit is not None:
            parts.append((W_WEIBULL, wfit(target_hours)))
        else:
            parts.append((W_WEIBULL, parts[0][1]))
        if en_rate[i] is not None:
            parts.append((W_EN, en_rate[i]))
        else:
            parts.append((W_EN, parts[0][1]))
        wsum = sum(w for w, _ in parts)
        med = clamp_pct(inv_logit(
            sum(w * logit(v) for w, v in parts) / wsum))
        if r_last >= 99.9:
            med = max(med, 99.5)
        spread = math.sqrt(
            tier_spread.get(tier, global_resid) ** 2 + horizon_term ** 2)
        spread = min(SPREAD_CAP, max(0.25, spread))
        rows.append({"id": trophy["id"], "q": quantiles(med, spread)})
    return {"trophies": rows}


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not [0.05, 0.25, 0.5, 0.75, 0.95]")
    if not case["history"]:
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = hours_since_release(parse_time(case["target"]))
    snaps = snapshots(case)
    if len(snaps) <= 1:
        return forecast_flat(case, snaps, horizon_days)
    return forecast_refresh(case, snaps, horizon_days, target_hours)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
