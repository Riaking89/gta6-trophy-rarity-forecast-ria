# gta6-trophy-rarity-forecast-ria

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riaking89/gta6-trophy-rarity-forecast-ria/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riaking89/gta6-trophy-rarity-forecast-ria/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that forecasts, for each of GTA VI's 55 PlayStation 5 trophies, its earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5/25/50/75/95%). It reads the house-recorded history up to the case checkpoint and is scored by mean pinball loss on the logit scale. Python 3.13, standard library only, deterministic, runs in ~0.06 s.

## The model (own design, dilution-aware)

Exophase's page is a snapshot: it freezes between the tracker's refreshes and jumps when it refreshes, because the players it tracks keep growing while the page shows the last refresh. A trophy's earn rate is earners / tracked, so a refresh dilutes every rate toward what the newly tracked players hold — on launch night the first refresh took tracked players 638 → 3,868 and every rate but the one at 100% fell. A persistence forecast ("rates stay") therefore loses badly whenever a refresh lands inside the horizon.

The program works from the case's history of reads:

- It collapses consecutive identical reads into distinct snapshots (one per refresh), keeping each snapshot's first-seen time.
- When the history is flat (one snapshot: no refresh seen yet) it forecasts one refresh like launch night's first: tracked players ×6.06, the newly tracked players holding the ubiquitous trophy at 100% and every other trophy at ~6.7% of its current rate (the split the first refresh's totals imply), with a spread that still admits the refresh may land small, large, or not at all.
- When the history has real refreshes, the median blends three components in logit space (weights 0.6 / 0.25 / 0.15): a logit-trend fit over the last 24 hours of snapshots (the house curve baseline's shape, fitted on refresh points only); a saturating Weibull rate curve per trophy, which cannot run away the way a power law does; and an earn-count model that fits E(t) with a saturating curve and N(t) with a power law, taking the rate as E/N, so dilution is in the forecast by construction. The spread comes from the trend fits' residuals per tier plus a term that grows with the square root of the horizon in days.

## Local validation (`validate.py`)

- Scorer reproduces the RULES worked example exactly (mean 0.403404).
- Practice cases (p00, p01) vs three proxy answers for the first refresh: **0.097 / 0.108 / 0.193** (uniform 6.7% dilution / r²-concentrated fresh earns / zero fresh earns), against **1.40–1.69** for the persistence and curve baselines. The practice answers are proxies, not the real answers — the kit ships no answer files — so these are a robustness check, not a score claim.
- Synthetic launch-week backtest (a cohort simulation with power-law player growth and refresh freezes; deadline 48 h → target day 7, 5 seeds): **0.141** mean pinball loss, against 0.398 persistence and 0.278 house curve; holdout seeds 6–10: **0.142** vs 0.378 / 0.283. A simulation result, not a promise about the real final.
- Output format checked: 55 rows, ids 1–55, five non-decreasing quantiles each in [0, 100], 2,762 bytes (cap 65,536). Two runs are byte-identical.

## Run it

```sh
python3 main.py
```

Reads `/input/<id>.in`, writes `/output/<id>.out`. Override the directories with `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`. Weights and spreads are tunable by environment (`MF_W_TREND`, `MF_W_WEIBULL`, `MF_W_EN`, `MF_HORIZON_SIGMA`, `MF_FLAT_SPREAD`); defaults are the tuned values above. Run `python3 validate.py` for the full local validation (practice + synthetic backtest + weight/spread sweeps).

---

<sub>Entered on [make.fun](https://make.fun/?ref=v8jsvbt9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
