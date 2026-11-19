# gta6-trophy-rarity-forecast-ria

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riaking89/gta6-trophy-rarity-forecast-ria/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riaking89/gta6-trophy-rarity-forecast-ria/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that forecasts, for each of GTA VI's 55 PlayStation 5 trophies, its earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5/25/50/75/95%). It reads the house-recorded history up to the case checkpoint and is scored by mean pinball loss on the logit scale. Python 3.13, standard library only, deterministic, runs in ~0.06 s.

## The model (own design, dilution-aware)

Exophase's page is a snapshot: it freezes between the tracker's refreshes and jumps when it refreshes, because the players it tracks keep growing while the page shows the last refresh. A trophy's earn rate is earners / tracked, so a refresh dilutes every rate toward what the newly tracked players hold — on launch night the first refresh took tracked players 638 → 3,868 and every rate but the one at 100% fell. A persistence forecast ("rates stay") therefore loses badly whenever a refresh lands inside the horizon.

The program works from the case's history of reads:

- It collapses consecutive identical reads into distinct snapshots (one per refresh), keeping each snapshot's first-seen time.
- When the history is flat (one snapshot: no refresh seen yet) it forecasts one refresh like launch night's first: tracked players ×6.06, the newly tracked players holding the ubiquitous trophy at 100% and every other trophy at a fresh fraction that grows with the trophy's old rate (~sqrt(rate): a new cohort reaches easy trophies far sooner than hard ones), calibrated so the fresh earns total ~10 over 3,230 new players. That total is not assumed: it is identified from the v1 entry's real practice grade (loss 0.174689, typical miss 0.05 points), which the v1 assumption of 103 fresh earns cannot reproduce. The spread is 0.15 logit.
- When the history has real refreshes, the median blends three components in logit space (weights 0.6 / 0.25 / 0.15): a logit-trend fit over the last 24 hours of snapshots (the house curve baseline's shape, fitted on refresh points only); a saturating Weibull rate curve per trophy, which cannot run away the way a power law does; and an earn-count model that fits E(t) with a saturating curve and N(t) with a power law, taking the rate as E/N, so dilution is in the forecast by construction. The spread comes from the trend fits' residuals per tier plus a term that grows with the square root of the horizon in days.

## Local validation (`validate.py`)

- Scorer reproduces the RULES worked example exactly (mean 0.403404).
- Real server practice grade, v1 entry (entry 01a1f5a4…, run 01a1f5a7…): **loss 0.174689**, coverage_90 0.927273, typical_miss_pp 0.05, rank ~901 of 4,900 scored (top-20 cutoff 0.0492). The v1 spread was too wide (coverage 0.927 > 0.9 nominal) and the median too high — v2 addresses both. **v2 has not been submitted or scored yet; its practice score will come from the server.**
- Practice cases (p00, p01) vs three proxy answers for the first refresh, v2 defaults: **0.181 / 0.170 / 0.083** (uniform 6.7% dilution / r²-concentrated fresh earns / zero fresh earns), against **1.40–1.69** for the persistence and curve baselines — worst proxy 0.181 vs v1's 0.193. Read honestly: proxies A/B encode the old, disproven 103-earn total, so v2 scores worse against them and much better against C (0.083 vs v1's 0.193). The proxies are a robustness check, not a score claim — the kit ships no answer files. The flat-prior identification in `validate.py` (scoring the v1 forecast against gamma-law truths) finds total fresh earns ~10: at total 10 the v1 forecast scores loss 0.174 / miss 0.046, matching the server's 0.174689 / 0.05; at 103 it scores 0.102 / 0.007, which the server contradicts.
- Synthetic launch-week backtest (a cohort simulation with power-law player growth and refresh freezes; deadline 48 h → target day 7, 5 seeds): **0.125** mean pinball loss, against 0.398 persistence and 0.278 house curve; weight sweep confirms the v2 default blend (0.6 / 0.25 / 0.15) is best (trend-only 0.147, equal thirds 0.160). A simulation result, not a promise about the real final. A fourth median component was tried in v2 and removed: a count-consistent refresh-step model scored far worse on the synthetic final at every weight (0.27 at weight 0.25, 0.15 at 0.10, 0.13 at 0.05, vs 0.125 with it off), because it attributes all new earns to the new cohort while the old players keep earning as they age, and compounds a constant growth factor where growth decays.
- Output format checked: 55 rows, ids 1–55, five non-decreasing quantiles each in [0, 100], 2,795 bytes (cap 65,536). Two runs are byte-identical; runtime ~0.06 s for both practice cases (limit 8 s).

## Run it

```sh
python3 main.py
```

Reads `/input/<id>.in`, writes `/output/<id>.out`. Override the directories with `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`. Weights and spreads are tunable by environment (`MF_W_TREND`, `MF_W_WEIBULL`, `MF_W_EN`, `MF_HORIZON_SIGMA`, `MF_FLAT_SPREAD`, `MF_FLAT_GAMMA`, `MF_FRESH_EARNS_PER_NEW`); defaults are the tuned values above. Run `python3 validate.py` for the full local validation (practice + synthetic backtest + weight/spread sweeps + flat-prior identification).

---

<sub>Entered on [make.fun](https://make.fun/?ref=v8jsvbt9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
