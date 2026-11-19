# gta6-trophy-rarity-forecast-ria

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riaking89/gta6-trophy-rarity-forecast-ria/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riaking89/gta6-trophy-rarity-forecast-ria/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that forecasts, for each of GTA VI's 55 PlayStation 5 trophies, its earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5/25/50/75/95%). It reads the house-recorded history up to the case checkpoint and is scored by mean pinball loss on the logit scale. Python 3.13, standard library only, deterministic, runs in ~0.06 s.

## The model (own design, dilution-aware)

Exophase's page is a snapshot: it freezes between the tracker's refreshes and jumps when it refreshes, because the players it tracks keep growing while the page shows the last refresh. A trophy's earn rate is earners / tracked, so a refresh dilutes every rate toward what the newly tracked players hold — on launch night the first refresh took tracked players 638 → 3,868 and every rate but the one at 100% fell. A persistence forecast ("rates stay") therefore loses badly whenever a refresh lands inside the horizon.

The program works from the case's history of reads:

- It collapses consecutive identical reads into distinct snapshots (one per refresh), keeping each snapshot's first-seen time.
- When the history is flat (one snapshot: no refresh seen yet) it forecasts one refresh like launch night's first: tracked players ×6.06, the newly tracked players holding the ubiquitous trophy at 100% and every other trophy at a fresh fraction that grows steeply with the trophy's old rate (~rate⁴: a new cohort reaches easy trophies far sooner than hard ones), calibrated so the fresh earns total ~103 over 3,230 new players. That total and shape are not assumed: they are jointly identified from the v1 and v2 entries' real practice grades (v1: loss 0.174689, miss 0.05, coverage 0.927; v2: loss 0.168216, miss 0.05, coverage 0.891). The v2 total of ~10 was a misidentification — it held the shape fixed at low gamma, where only the total can move; scanning gamma and total jointly, the best fit is gamma ~4 with total ~103 (the v1 total was right, its uniform shape was wrong). No smooth truth matches all six server numbers exactly: the realized counts are lumpier than any smooth law. The spread is 0.15 logit.
- When the history has real refreshes, the median blends three components in logit space (weights 0.6 / 0.25 / 0.15): a logit-trend fit over the last 24 hours of snapshots (the house curve baseline's shape, fitted on refresh points only); a saturating Weibull rate curve per trophy, which cannot run away the way a power law does; and an earn-count model that fits E(t) with a saturating curve and N(t) with a power law, taking the rate as E/N, so dilution is in the forecast by construction. The spread comes from the trend fits' residuals per tier plus a term that grows with the square root of the horizon in days.

## Local validation (`validate.py`)

- Scorer reproduces the RULES worked example exactly (mean 0.403404).
- Real server practice grades: v1 entry (entry 01a1f5a4…, run 01a1f5a7…): **loss 0.174689**, coverage_90 0.927273, typical_miss_pp 0.05, rank ~901 of 4,900 scored (top-20 cutoff 0.0492). v2 entry (entry 01a1f5b8…, run 01a1f5b8-96ac…): **loss 0.168216**, coverage_90 0.890909, typical_miss_pp 0.05, rank ~#850 — barely better, still far from the cutoff. **v3 has not been submitted or scored yet; its practice score will come from the server.**
- Practice cases (p00, p01) vs three proxy answers for the first refresh, v3 defaults: **0.162 / 0.133 / 0.152** (uniform 6.7% dilution / r²-concentrated fresh earns / zero fresh earns), against **1.40–1.69** for the persistence and curve baselines — worst proxy 0.162 vs v2's 0.181, and 90% coverage against proxy C is 0.909 (nominal 0.9; v2's was 0.982, too wide). Read honestly: the proxies are a robustness check, not a score claim — the kit ships no answer files. Against Poisson draws around the identified truth (gamma ~4, total ~103), v3's expected practice loss is ~0.102 vs v2's ~0.153. A Poisson-count-quantile flat branch was tried in v3 and rejected: it gains little (expected loss 0.102 → 0.090 in the ideal case) and fails the robustness bar (worst proxy 0.185, worse than v2's 0.181).
- Synthetic launch-week backtest (a cohort simulation with power-law player growth and refresh freezes; deadline 48 h → target day 7, 5 seeds): **0.125** mean pinball loss, against 0.398 persistence and 0.278 house curve; weight sweep confirms the default blend (0.6 / 0.25 / 0.15) is best (trend-only 0.147, equal thirds 0.160). A simulation result, not a promise about the real final. v3 changes only the flat branch, so the synthetic final is unchanged. A fourth median component was tried in v2 and removed: a count-consistent refresh-step model scored far worse on the synthetic final at every weight (0.27 at weight 0.25, 0.15 at 0.10, 0.13 at 0.05, vs 0.125 with it off), because it attributes all new earns to the new cohort while the old players keep earning as they age, and compounds a constant growth factor where growth decays.
- Output format checked: 55 rows, ids 1–55, five non-decreasing quantiles each in [0, 100], 2,795 bytes (cap 65,536). Two runs are byte-identical; runtime ~0.06 s for both practice cases (limit 8 s).

## Run it

```sh
python3 main.py
```

Reads `/input/<id>.in`, writes `/output/<id>.out`. Override the directories with `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`. Weights and spreads are tunable by environment (`MF_W_TREND`, `MF_W_WEIBULL`, `MF_W_EN`, `MF_HORIZON_SIGMA`, `MF_FLAT_SPREAD`, `MF_FLAT_GAMMA`, `MF_FRESH_EARNS_PER_NEW`); defaults are the tuned values above. Run `python3 validate.py` for the full local validation (practice + synthetic backtest + weight/spread sweeps + flat-prior identification).

---

<sub>Entered on [make.fun](https://make.fun/?ref=v8jsvbt9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
