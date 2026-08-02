# Improvement roadmap: where this model can still get better

> **Status update:** Tier 1 items 1.1, 1.2(a)(b)(d), 1.3, and 1.6 (SPA) are now implemented in `final4.ipynb`, which is the project's final model and performance record. Remaining open: 1.2(c) signal sizing, 1.4 auction features, 1.5 realized-vol features, 1.6 MCS, and Tier 2.

Synthesized from a four-sweep literature review (data sources, statistical methods, trading rules, machine learning) plus empirical tests run directly on the stored final2/final3 forecasts. Items are annotated:

- ✅ **DONE** — already implemented in this repo
- 🧪 **TESTED** — empirically verified on stored forecasts this session (results below)
- ⬜ **OPEN** — not yet attempted

Ground rules carried through every item: `final.ipynb` / `final2.ipynb` / `final3.ipynb` are frozen — all experiments go in **new** notebooks, reusing the stored per-origin forecasts in `data/generated/final2_results/yield_forecasts.csv` and `data/generated/final3_results/yield_forecasts.csv` (join on `origin_date + horizon + maturity`, not on positions). Every parameter is fixed a priori or chosen on pre-holdout tuning origins. Every run is one more look at an already-iterated holdout, so any positive result is hypothesis-generating until the 3-month forward test.

**The honest ceiling, stated up front:** the full Tier 1+2 program moves h=20 RMSE from 26.7 to *maybe* 25.7–26.3 bp and weighted DA from 0.67 to 0.68–0.69. Anything claiming more should be treated as a bug or leakage until proven otherwise. The big improvements available are in **net-of-cost economic value** and **inference quality**, not raw RMSE.

---

## Tier 1 — do first (documented gains, low risk)

### 1.1 ✅ DONE — Overlap-robust DM inference
The review flagged uncorrected DM p-values on overlapping errors as "the single most likely referee kill-shot." Already handled: the notebooks use the classic rectangular-kernel long-run variance with lags up to h−1, the Harvey–Leybourne–Newbold small-sample correction, and Benjamini–Hochberg q-values across all comparisons. What remains from this item is 1.6 below (SPA/MCS).

### 1.2 🧪 PARTIALLY TESTED — Transaction-cost realism package (`notebooks/trading_rules.ipynb`, to be created)
The biggest net-economic-value lever, requiring **zero** model changes:

- **(a) ⬜ Turnover netting across overlapping origins.** Hold the average of the last h origins' positions and pay cost only on the *change* in the net position, not a full round trip per forecast. At h=20, consecutive-day signals agree ~80–90% of the time, so effective round trips fall 3–5×. Expected: 1bp-cost capture at h=20 from ~0.29 toward ~0.32–0.35. (Frazzini–Israel–Moskowitz 2018 on real-world trading costs.)
- **(b) 🧪 TESTED, WORKS — No-trade threshold.** Verified on stored final2 forecasts at 0.5bp/trade cost:

  | h | threshold | trades | net weighted DA | NW t |
  |---|---|---|---|---|
  | 1 | none | 7,645 | 0.487 (loses) | −0.7 |
  | 1 | ≥1bp | 431 | **0.673** | **4.6** |
  | 5 | ≥2bp | 1,639 | **0.700** | **3.0** |
  | 5 | ≥4bp | 556 | **0.889** | **5.0** |
  | 20 | ≥4bp | 3,242 | **0.764** | **2.4** |

  Monotone in threshold at every horizon — the model's conviction is informative. Pre-commit the economically motivated rule "trade only when predicted edge ≥ cost" (≈1bp, the H.15 quantization unit) rather than tuning the threshold. (Gârleanu–Pedersen 2013 no-trade band.)
- **(c) ⬜ Continuous signal sizing**: position ∝ clip(predicted change / trailing EWMA vol, −1, 1); documented ~25–35% turnover cut at unchanged gross performance (Baltas–Kosowski).
- **(d) 🧪 TESTED, IMPORTANT CAVEAT — Futures-implementable universe.** Restricting to 2Y/5Y/10Y/30Y (tenors with liquid futures) at 0.5bp cost: h=20 net weighted DA drops to 0.584 (t=0.94, not significant). **Much of the paper P&L lives in short-end CMT points that are not directly tradable.** The paper must disclose this; combining (a)+(b) on the futures universe is the test of whether an implementable version survives.

### 1.3 ⬜ EWA/Hedge combiner replacing the trailing-window argmin
The current adaptive weight is near Follow-the-Leader, which has no worst-case regret guarantee. Exponentially weighted averaging (weight ∝ exp(−η × discounted loss), discount ≈0.99, fixed-share α≈0.01) is the textbook fix (Cesa-Bianchi & Lugosi 2006), kills two hyperparameters (trailing window = 60, min = 10), gives continuous weights for free, and comes with a citable guarantee. Expected RMSE change ~0–0.5bp; main payoff is robustness in the forward test and a smoother auditable weight path.

### 1.4 ⬜ Treasury auction calendar + results (the one genuinely new data source)
Free TreasuryDirect API. Documented V-shaped pre-auction concession / post-auction reversion of 1–4bp at the issued maturity (Lou–Yan–Yuan 2013 RFS; NY Fed SR 1188) — the same order of magnitude as the entire current edge. Features: days-to/since auction per maturity sector, offering size, bid-to-cover z-score, indirect share (lagged one full day to be safe). Expected: −0.2 to −0.6bp RMSE at h=1/5 concentrated on auction-adjacent days; ~nothing at h=20. Run as a controlled A/B on identical origins — the final3 lesson (spanning) applies here too.

### 1.5 ⬜ Realized-vol features from data already in the repo
EWMA (λ≈0.94) vol of daily 2Y/10Y/30Y changes, causally z-scored. Uses: shrink the ensemble toward RW when vol is elevated; inverse-variance weighting of Ridge training rows. Cheapest experiment on the list; expected small but improves tail behavior. (MOVE index itself is not freely available.)

### 1.6 ⬜ SPA test + Model Confidence Set (referee-proofing)
Hansen's SPA (via `arch.bootstrap.SPA`, block length ≥ 2h) with RW as benchmark prices in the search over alternatives — the standard answer to "you iterated on this holdout." MCS (Hansen–Lunde–Nason 2011) per horizon over {ensemble, VARX, RW, final3, AR/VAR baselines}. Pure post-processing on stored losses; zero effect on forecasts, large effect on credibility.

---

## Tier 2 — worth testing, uncertain payoff (pre-commit each variant)

- **2.1 ⬜ Regime-gated ensemble weight** — best remaining RMSE/DA shot. Compute trailing weights only over past origins whose filtered HSMM state matches today's, shrunk toward the unconditional weight (Elliott–Timmermann 2005). Caveat: unshrunk state-split weights routinely lose (the forecast-combination puzzle). Expected 0–1bp at h=20, possibly +1–2pts DA at regime turns.
- **2.2 ⬜ VIX as a nonlinear regime input only** (FRED `VIXCLS`, free). Linear VIX does *not* forecast bond returns (NY Fed SR 723); use log-VIX in HSMM emissions and stress-percentile de-weighting only. Never as a linear regressor — that's the final3 mistake repeated.
- **2.3 ⬜ Volatility-standardized targets** — fit Ridge on Δf/σₜ, rescale by today's σ. Homoskedasticizes so 2022 doesn't dominate the fit. A robustness row.
- **2.4 ⬜ Adaptive ridge penalty from trailing OOS error** + verify the change-space intercept is suppressed (so the high-penalty limit is *exactly* RW).
- **2.5 🧪 SUBSUMED by 1.2(b)** — quantile-gated trading (trade only when signal exceeds trailing interval half-width) is a variant of the tested threshold rule.
- **2.6 ⬜ Vol-targeted position sizing** — for bonds the Sharpe benefit is negligible (Harvey et al. 2018); implement as drawdown reduction, never claim more P&L.
- **2.7 ⬜ DV01-neutral 2s10s / fly overlays** — diagnostic: does the model know *shape* or just *level*? Either answer is publishable.
- **2.8 ⬜ Foreign 10Y timezone edge** (Bund/JGB close before the H.15 snapshot; free from Bundesbank/MOF). Episode-specific value at best.
- **2.9 ⬜ Elastic-net swap** — interpretability only: the sparsity pattern tells you where the h=20 edge lives.

---

## Tier 3 — tested or rejected: do not spend time here

| Idea | Verdict |
|---|---|
| 🧪 **Combine final2+final3 forecasts** | **TESTED, DEAD.** Loss correlation 0.96–0.998 with RMSE ratio ≈1 fails the Bates–Granger condition exactly as predicted. 50/50 and trailing-weight combos are all worse than final2 alone. Report the correlation as evidence final3 is spanned — it upgrades the final3 section from "it underperformed" to "and combining cannot help either." |
| ACM term premium | Vintage/leakage problem (re-estimated parameters); near-zero expected at 1–20d. |
| Fed funds futures | No free bulk history; the information channel is already spanned by the EFFR-basis result. |
| Fed SOMA / QT runoff | Pre-announced, near-deterministic; no innovation at daily origins. |
| Daily EPU index | Back-revised series; redundant with VIX in stress. |
| S&P returns as direct feature | Same spanned-information profile as final3's failures. |
| NN bond risk premia (BBT 2021) | Published corrigendum admits look-ahead; annual-horizon signal, ~0 at daily scale. |
| Gradient boosting / RF on daily changes | Documented losers vs ARIMA/RW at daily frequency (negative DM stats in head-to-heads). |
| Transformers / foundation TS models | Documented failures on daily Treasuries; ~57 effective h=20 observations is orders of magnitude below what the architecture needs. |
| Per-maturity direct models | Documented loser (De Pooter et al.): 11 noisy regressions < 3 regularized factors. |
| Continuous Bates–Granger weights | The combination puzzle: estimated optimal weights lose to crude weights. EWA (1.3) gives continuous weights the right way. |

---

## The forward-test protocol (write down before the window opens)

1. Freeze final2 + chosen combiner + full trading rule (threshold, cost vector, sizing) in a tagged commit.
2. Snapshot every input at each origin as data is released; never use revised series.
3. Pre-commit the exact tests: corrected DM, SPA, capture/NW-t at per-tenor costs.
4. State up front that 3 months ≈ only ~3 non-overlapping h=20 observations: the forward test is a sign-and-magnitude consistency check, not confirmation. Commit to reporting it regardless of outcome.

## The defensible headline for the paper

Not "we beat the random walk by 1.3bp." Rather: **"a shrinkage-toward-random-walk design with adaptive combination survives snooping-robust tests, delivers 0.67 weighted directional accuracy at the 20-day horizon, and — with conviction thresholding and turnover netting — clears realistic transaction costs."** Everything in Tiers 1–2 maximizes exactly that claim; nothing in Tier 3 plausibly changes it.
