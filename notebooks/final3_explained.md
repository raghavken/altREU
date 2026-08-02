# final3.ipynb, explained from the ground up

This document walks through every part of `final3.ipynb`: what each cell does, why it does it that way, the math behind it, and the honest caveats. It is written so you can defend any line of the notebook in a presentation or paper review without needing the code open.

**One-sentence summary:** final3 asks whether a feature set built from *what the literature says moves the yield curve* (momentum, policy drift, rate anchoring, shape mean-reversion) can out-forecast final2's feature set (factor lags, HSMM regimes, spillover networks) — under identical anti-leakage rules, on identical evaluation dates — and the answer turned out to be **no**.

*Note on versions:* the notebook was rewritten in the project's baseline code style (camelCase, explicit loops, full syntax) with the math unchanged and verified identical. The trading-strategy evaluation that briefly lived in final3 now lives in `final4.ipynb`, which owns the final model and the complete performance record; final3's job is the feature-set comparison and its DM head-to-head against final2.

---

## Table of contents

1. [Where final3 sits in the project](#1-where-final3-sits-in-the-project)
2. [The data](#2-the-data)
3. [Dynamic Nelson–Siegel: turning 11 yields into 3 factors](#3-dynamic-nelsonsiegel-turning-11-yields-into-3-factors)
4. [Why the model forecasts *changes*, not levels](#4-why-the-model-forecasts-changes-not-levels)
5. [The economics feature set, block by block](#5-the-economics-feature-set-block-by-block)
6. [The anti-leakage architecture](#6-the-anti-leakage-architecture)
7. [Two-stage training-only tuning](#7-two-stage-training-only-tuning)
8. [The three models and the adaptive ensemble](#8-the-three-models-and-the-adaptive-ensemble)
9. [Accuracy metrics, and the traps they avoid](#9-accuracy-metrics-and-the-traps-they-avoid)
10. [The Diebold–Mariano test, in plain language](#10-the-dieboldmariano-test-in-plain-language)
11. [Weighted directional accuracy: the trading backtest](#11-weighted-directional-accuracy-the-trading-backtest)
12. [The results and the pre-registered decision rule](#12-the-results-and-the-pre-registered-decision-rule)
13. [Every honest caveat in one place](#13-every-honest-caveat-in-one-place)
14. [Glossary](#14-glossary)

---

## 1. Where final3 sits in the project

| Notebook | Feature set | Role |
|---|---|---|
| `final.ipynb` | DNS factor lags + smoothed HSMM + spillover | First integrated version; HSMM probabilities are smoothed (use future data), so it's exploratory |
| `final2.ipynb` | DNS factor lags + **causal filtered** HSMM + spillover | The flagship: leakage-aware, full-holdout evaluation, adaptive ensemble |
| `final3.ipynb` | **Economics-driven features** (no HSMM) | The challenger: does economic theory beat statistical features? |

final3 deliberately reuses final2's *machinery* unchanged — change-space Ridge, two-stage tuning, adaptive trailing weight, same holdout origins — so that the **only** difference between the two notebooks is the information the model sees. If final3 had also changed the model class or the evaluation dates, you couldn't attribute any difference to the features.

The HSMM is dropped in final3 for two reasons: its filtered probabilities earned near-zero weight during final2's training-only tuning, and the policy-cycle state variables (Section 5, Block 4) capture the "what regime are we in" idea with directly observable quantities instead of a latent model.

---

## 2. The data

All three files live in `data/` and are joined **by calendar date** — never by row position, because the three sources have different holiday calendars.

| File | Contents | Frequency | Role |
|---|---|---|---|
| `FRB_H15.csv` | 11 constant-maturity Treasury yields (1M…30Y), in **percent** | Business days | The forecast target and the source of the DNS factors |
| `T5YIE.csv` | 5-year breakeven inflation (nominal minus TIPS yield) | Business days | Inflation-expectation features |
| `EFFR.csv` | Effective federal funds rate | Business days (incl. some bond holidays) | Policy-rate features |
| `data/generated/spillover_features.csv` | Diebold–Yilmaz connectedness measures | Business days | Cross-maturity transmission features (built causally by `spillover.ipynb`) |
| `data/generated/final2_results/yield_forecasts.csv` | final2's retained forecasts | — | Defines the shared evaluation origins and provides the head-to-head competitor |

A recurring subtlety: when you subtract two pandas Series with different date indexes, pandas aligns them on the **union** of dates and fills the gaps with NaN. EFFR prints on some days the bond market is closed, so `short_end_yields − EFFR` contains scattered NaNs — and a 250-day rolling statistic returns NaN if *any* day in its window is NaN. The first run of final3 lost 72% of its evaluation sample to exactly this. The fix (`.dropna()` before rolling) is commented in the feature cell; it's a good example of why every join in these notebooks is explicit about alignment.

---

## 3. Dynamic Nelson–Siegel: turning 11 yields into 3 factors

### The model

For a maturity of τ years, the Nelson–Siegel curve says the yield is:

```
y(τ) = β₁ · 1  +  β₂ · (1 − e^{−λτ})/(λτ)  +  β₃ · [(1 − e^{−λτ})/(λτ) − e^{−λτ}]
```

The three columns multiplying β₁, β₂, β₃ are called **loadings**, and they have shapes that give the betas their names:

- **Level (β₁):** its loading is the constant 1 — moving β₁ shifts *every* maturity equally.
- **Slope (β₂):** its loading is ≈1 at very short maturities and decays toward 0 — moving β₂ mostly moves the short end, i.e., it tilts the curve.
- **Curvature (β₃):** its loading is hump-shaped — near 0 at both ends, peaking in the middle — so β₃ bows the belly of the curve.

Three numbers reproduce the whole 11-point curve to within a few basis points, because yield curves are extremely smooth. This is the classic result that ~99% of curve variance is explained by three principal components (Litterman–Scheinkman 1991), and Nelson–Siegel is a parametric version of the same idea.

### How the factors are estimated each day

Because λ (the "decay") is fixed, the loadings form a constant 11×3 matrix **L**. Each day's factor vector is just an ordinary least squares fit of that day's 11 yields onto **L**:

```
factors(t) = pinv(L) @ yields(t)
```

`pinv(L)` is the Moore–Penrose pseudoinverse — the matrix that performs the OLS projection. It's computed **once** and applied to every date, which is why the notebook's factor estimation is a single matrix multiplication. Crucially, day *t*'s factors use *only* day *t*'s yields — no time-series information — so this step can never leak the future.

The **residuals** `yields(t) − L @ factors(t)` are what the 3-factor curve cannot explain (the 20Y point is the usual offender). They're kept and forecast separately (Section 8).

### Choosing the decay λ

λ controls where the curvature hump sits. The literature's classic value 0.0609 assumes maturity measured in **months**; this project measures maturity in **years**, so the equivalent is ≈0.73. Rather than trust a unit conversion, the notebook searches a grid of λ values and keeps the one minimizing curve-fit RMSE **on the first 80% of the sample only**. The holdout can't influence the choice — this is the template for every design decision in the notebook.

---

## 4. Why the model forecasts *changes*, not levels

This is the single most important modeling decision, so here is the full logic.

Ridge regression minimizes `‖y − Xβ‖² + α‖β‖²`. The penalty α shrinks coefficients toward zero. Ask: *what does the forecast become when shrinkage wins completely (β → 0)?* The prediction collapses to the intercept — the **training-sample mean of the target**.

- If the target is the future **level** of a yield factor, that mean is a years-long historical average. Yields are extremely persistent (daily AR coefficient ≈ 0.999 for level), so today's level is the best guess for next month — but the historical average can be *hundreds of basis points away* from today. A level-space Ridge is therefore biased toward an economically absurd anchor, and the bias grows with α.
- If the target is the **change** `f(t+h) − f(t)`, the training-mean change is ≈ 0. Full shrinkage now collapses to "no change" — **the random walk**, which is the best-known naive forecast for yields. Shrinkage in change space is a feature, not a bug: α becomes a dial between "pure random walk" (α large) and "trust the predictors" (α small).

The level forecast is then reconstructed as `current factors + predicted change`. Under plain OLS this reparameterization changes nothing (it's a linear transformation); under *regularization* it changes everything, because it changes what the model defaults to when the data are uninformative.

The empirical confirmation: when this project's models predicted levels, VARX lost to the random walk at 5 and 20 days and XGBoost's 20-day RMSE was 34.5bp (trees can't extrapolate levels above their training range at all). In change space both became competitive.

---

## 5. The economics feature set, block by block

Every feature dated *t* uses only data through *t* — the constructions are diffs, trailing windows, and trailing EWMAs, all of which look backward by definition. There are ~35 predictors in total.

### Block 1 — Multi-scale momentum (the best-documented signal)

```
{factor}_momentum_{k}d = factor(t) − factor(t−k)      for k ∈ {1, 5, 21, 63}
```

**Mechanism.** Yield *changes* are positively autocorrelated at roughly one-month lags: a curve that has been rising tends to keep rising. Sihvonen (2024, *Review of Finance*) shows this "yield momentum" is not spanned by the current curve shape — you cannot recover it from today's yields alone, you need the *path*. The economic driver is Fed gradualism (hiking/cutting in long same-direction campaigns) plus slow-moving capital that underreacts to news.

**Why four lookbacks?** 21d and 63d carry the documented positive autocorrelation. 1d and 5d are included but their *sign is left to the model* — the literature finds short-horizon reversal in some samples (bid-ask bounce, CMT smoothing artifacts), so imposing a positive sign there would be wrong. Ridge sees all four and weights them as the data warrant.

**Why not more factor lags instead?** A lag structure `{f(t), f(t−1), …}` and a diff structure `{f(t), Δ₁, Δ₅, …}` span similar information, but diffs are far better conditioned for Ridge: they're nearly uncorrelated with each other, whereas raw lags are 0.999-correlated with one another and force Ridge to estimate a difference of huge coefficients.

### Block 2 — Post-move policy drift

```
post_move_drift = sign(last EFFR move) · exp(−days_since_move / 50)
```

**Mechanism.** Brooks–Katz–Lustig (2018) document that longer-maturity yields *underreact* to federal-funds-rate surprises and then drift in the same direction for ~50 trading days — slow-moving capital again. The feature encodes "which way did policy last move, and how recently" with an exponential clock calibrated to that 50-day finding. (Precision note: the 50 in `exp(−days/50)` is an *e-folding* time, not a half-life — the signal halves in ≈35 trading days. The notebook constant is named `driftDecayDays` for this reason; don't call it a 50-day half-life in the paper.)

**The move detector.** A "policy move" is a day when |ΔEFFR| ≥ 7.5bp. Genuine target changes are 25bp+; day-to-day EFFR noise is a few bp (month-end funding pressure). The threshold cleanly separates the two; a rare quarter-end false positive just resets the clock on a day the market also saw a large EFFR print, which is harmless.

### Block 3 — Short-end EFFR basis

```
short_end_basis      = mean(1M, 3M, 6M, 1Y yields) − EFFR
short_end_basis_z    = 250-day trailing z-score of the basis
short_end_basis_change_5d = 5-day change of the basis
```

**Mechanism.** Money-market maturities are arbitrage-anchored to the overnight policy rate: a 1-month bill can't yield much more or less than expected overnight rates over that month. When the basis is stretched (short yields far above EFFR), it predicts either the Fed will hike (yields were right) or the basis will close back down. Either way, the *basis* is mean-reverting even though both of its components are near-random-walks — that's the classic cointegration insight, and it's the reason the z-score (how stretched is the basis versus its own recent history) is the feature rather than the raw level.

**The z-score floor.** During 2009–2015 the short end was pinned at zero and the basis barely moved, so its rolling standard deviation was ~1bp. Dividing by that would turn meaningless 1bp wiggles into "4-sigma events." The denominator is floored at 1bp (`.clip(lower=0.01)` in percent units) to prevent it.

### Block 4 — Policy-cycle state

```
last_policy_move_sign   ∈ {−1, 0, +1}
policy_run_signed       = signed count of consecutive same-direction moves (+3 = three hikes in a row)
effr_change_63d / 126d  = cumulative policy change over ~3 / ~6 months
```

**Mechanism.** Coibion–Gorodnichenko (2012): the Fed moves in runs — a hike is most likely followed by another hike. These features tell the model *where in the policy cycle we are*. Caveat honestly stated in the notebook: markets already price much of this (Rudebusch 2002), so these act more as regime conditioners than as raw alpha — they are the observable replacement for the HSMM's latent states.

### Block 5 — Shape mean-reversion z-scores (never on level)

```
slope_z, curvature_z = 250-day trailing z-scores of the DNS slope / curvature
butterfly_z          = z-score of (2·5Y − 2Y − 10Y)
```

**Mechanism.** Slope and curvature are stationary and mean-revert over weeks; a very steep curve tends to flatten, an extreme butterfly tends to un-kink. The **level** factor is a near-unit-root and deliberately gets *no* mean-reversion feature: a "level z-score" would just re-import the level-space bias that Section 4 eliminated. This asymmetry — mean-revert shape, random-walk level — is straight from Diebold–Li (2006) and the relative-value trading literature.

### Block 6 — i\*-gap: the slow nominal anchor

```
i*(t)      = EWMA(EFFR, half-life 756d) + EWMA(breakeven, half-life 756d)
istar_gap  = DNS level − i*(t)
```

**Mechanism.** Bauer–Rudebusch (2020, *AER*): yields revert not to a constant mean but to a *drifting* anchor equal to the trend real rate plus trend inflation. The 3-year-half-life EWMAs build a causal proxy for that anchor; the gap between today's level and the anchor is the (slow, 20-day-horizon) mean-reversion signal that is legitimate for level — because it reverts to a *moving* target, not a fixed historical average.

### Blocks 7–8 — Breakeven momentum and compact spillover

`breakeven_change_5d/21d`: inflation-expectation shifts propagate into nominal yields. `total_connectedness`, `long_net_transmission` and their 5-day changes: the Diebold–Yilmaz network intensity from `spillover.ipynb`, kept because final2's tuning found them mildly useful and dropping them would confound the comparison ("was final3 worse because econ features are worse, or because it lost the network block?").

---

## 6. The anti-leakage architecture

Every forecasting result in this project stands or falls on one claim: *a forecast dated t used only information available at t.* Four rules enforce it.

**Rule 1 — the training-slice bound.** To train a model for horizon *h* at origin *t*, a historical example at origin *o* may be used only if its outcome is already known: `o + h ≤ t`. The code writes this as `last_training_origin = forecast_origin − horizon`. Miss this by one day and every h=20 model would train on 19 days of future data.

**Rule 2 — tuning lives strictly in the training era.** All 40 tuning pseudo-origins end ≥ 20 trading days before the first evaluated origin, so even the longest-horizon tuning outcome is realized before evaluation begins. (final2's first version got this wrong by 10 days; final3 inherits the fix.)

**Rule 3 — design boundaries precede the holdout.** Anything *chosen* from data — the decay λ, the spillover VAR lag — is chosen on samples that end before the first evaluated origin, and the notebook computes the holdout start as the max of all such boundary dates.

**Rule 4 — the adaptive weight only sees realized outcomes.** At origin *t*, the ensemble weight is picked from forecasts made at origins *o* with `o + h ≤ t` (their outcomes are observable) — never from today's own forecast, which is recorded only *after* the weight is chosen.

One more comparability rule specific to final3: the evaluated origins are **exactly final2's origin dates** (an assertion requires ≥95% overlap and the run achieves ~100%), so the head-to-head is forecast-for-forecast on identical days — differences in results cannot come from differences in sample.

---

## 7. Two-stage training-only tuning

The grid: history window ∈ {504 days, 1000 days, expanding}, Ridge α ∈ {0.1, 1, 10, 100, 1000}, initial ensemble weight ∈ {0, 0.25, 0.5, 0.75, 1}. Forty pseudo-origins spaced a week apart score every combination by curve-wide RMSE.

**Why two stages, in this order?**

- **Stage 1** picks (window, α) minimizing the **pure-VARX** (weight = 1) tuning error. 
- **Stage 2** picks the initial weight *given* those settings.

The naive alternative — one joint search over (window, α, weight) — has a degenerate failure mode: if weight = 0 wins (the random walk is hard to beat!), then *every* (window, α) pair ties exactly, because the VARX is multiplied by zero. The "selected" VARX settings become an arbitrary tie-break. This actually happened in an early final2 run: the reported VARX got the *weakest* shrinkage (α = 0.1) by tie-break and looked absurdly bad (109bp RMSE at h=20). Two-stage selection guarantees the reported VARX is its best training-feasible self.

final3's stage-1 picked expanding windows with α ∈ {100, 1000} — very strong shrinkage, exactly what theory predicts when 20-day change targets overlap heavily (504 rows of 20-day changes contain only ~25 independent observations against ~35 predictors).

---

## 8. The three models and the adaptive ensemble

At every one of the ~1,149 origins, for each horizon:

1. **Observed Yield Random Walk** — every maturity stays at today's observed value. Note this is defined in *observed-yield* space, not "reconstruct today's DNS curve": reconstructing would add today's DNS fitting error to the benchmark and unfairly handicap it.
2. **Econ VARX** — Ridge predicts the 3-factor *change*; add current factors; multiply by the loading matrix **L** to get an 11-maturity curve; add the AR(1) residual forecast (next paragraph).
3. **Econ Adaptive Ensemble** — `w · VARX + (1−w) · RandomWalk`, with `w` re-chosen at every origin.

**The residual AR(1).** The DNS curve misfits some maturities persistently (the 20Y especially). Each maturity's fitting residual is modeled as `r(t) = c + φ·r(t−1) + ε`, re-estimated at every origin from the trailing 1,000 days; the h-step forecast decays toward the long-run mean at rate φʰ. φ is clipped to (−0.99, 0.99) so an estimated near-unit root can't explode over 20 days. This converts "factor forecast" into "curve forecast" without pretending DNS fits perfectly.

**The adaptive weight, step by step.** At origin *t*, horizon *h*:
1. Collect past origins whose outcomes are realized (`o + h ≤ t`); take the most recent 60.
2. For each candidate weight in {0, 0.25, 0.5, 0.75, 1}, compute the mean squared error that weight *would have produced* on those realized forecasts.
3. Use the winner today. If fewer than 10 realized origins exist yet, fall back to the training-only initial weight.

This mimics exactly what a live forecaster could do, and it is the mechanism that made final2 beat the random walk: a weight frozen at its 2021 (calm-period) value of 0 would never have engaged the VARX during the 2022 hiking cycle, when its trend-following information was most valuable. The notebook plots the weight path so you can see when the model trusted the VARX (weight rises) versus retreated to the random walk.

---

## 9. Accuracy metrics, and the traps they avoid

**RMSE and MAE in basis points.** Errors are `100 × (actual − predicted)` because H.15 yields are in percentage points and 1bp = 0.01pp. RMSE squares before averaging (punishes rare large misses); MAE doesn't. Both are shown because a model can look fine on one and bad on the other.

**Directional accuracy (DA), with two exclusions.** DA = share of forecasts whose predicted *sign* of change matches the realized sign. Two row types are unscorable and excluded from numerator *and* denominator:

- *Zero predicted change.* The random walk always predicts exactly zero change — it has no direction. A naive comparison scores it ~0% "accurate," making any model look good against it. It's reported as NaN instead.
- *Zero realized change.* H.15 yields are published to 0.01, so a meaningful share of realized changes (≈12% at h=1) are *exactly* zero. A nonzero prediction can never "match" a tie, so leaving ties in the denominator silently caps every model's DA below 1 − tie-share.

**The base rate column.** DA must beat the best *constant* guess ("always up" or "always down"), not 0.5. Over a hiking-cycle holdout, "always up" alone was right 57% of the time at h=20 — so a model's 60.9% is a real edge of ~4 points, not ~11.

---

## 10. The Diebold–Mariano test, in plain language

The question DM answers: "model A has lower average loss than model B — but is the difference large relative to how much the loss *difference* bounces around over time?"

Construction: let `d(t) = lossA(t) − lossB(t)` at each origin (loss = maturity-averaged squared error in bp²). The statistic is `mean(d) / se(mean(d))`. Negative favors A. The subtleties are all in the standard error:

- **Overlap.** Consecutive 20-day forecasts share 19 of their 20 days, so the `d(t)` series is heavily autocorrelated and the naive `sd/√n` wildly understates the standard error. The fix is a long-run variance summing autocovariances up to lag h−1.
- **Rectangular, not Bartlett, kernel.** Newey–West (Bartlett) weights *shrink* higher-lag autocovariances — but for overlapping forecasts those autocovariances are *positive*, so shrinking them understates the variance and overstates significance. The classic DM construction uses full (rectangular) weights; the notebook uses that, with a guard that reports NaN if the estimate goes non-positive (a known possibility with rectangular kernels).
- **Harvey–Leybourne–Newbold correction.** A small-sample multiplier and Student-t reference distribution instead of normal.
- **Benjamini–Hochberg q-values.** Several DM tests are run at once; the q-value column controls the false-discovery rate. The notebook's rule: cite q-values, not raw p-values, when claiming significance.

---

## 11. Weighted directional accuracy: the trading backtest

Plain DA counts every forecast equally. But being right on a 1bp day and wrong on a 30bp day is a terrible trade record with a fine DA. The backtest weights each call by the size of the move — which is what a strategy actually monetizes.

**Definition.** For each forecast with a nonzero predicted change, take position `sign(predicted change)` (+1 = bet yields rise). Its P&L in bp is `sign(predicted) × realized change`. Then:

```
capture     = Σ P&L / Σ |realized change|        ∈ [−1, +1]
weighted DA = (1 + capture) / 2                  ∈ [0, 1],  0.5 = breakeven
```

Capture reads as "fraction of all available basis points the strategy banked": +1 means every call right, −1 every call wrong, 0 is coin-flip money.

**The robustness protocol** (each part exists because a single number can lie):

- *Rolling windows, three lengths* (63/126/252 origins, stepped 21): a strategy that made all its money in one lucky quarter shows a low "share of profitable windows" even with a good full-sample capture. Reporting three window lengths prevents cherry-picking the one that looks best.
- *Newey–West t-statistic* on mean per-origin P&L (lag h−1, same overlap logic as the DM test).
- *Transaction costs* of 0 / 0.5 / 1.0 bp per round trip. This is the reality check: final2's h=1 signal is **unprofitable at 0.5bp cost** — the 1-day edge is real but smaller than the cost of trading it — while its h=20 signal survives 1bp comfortably.
- *Cumulative P&L plot*: steady staircase = persistent edge; one cliff = one lucky event.

**Caveats stated in the notebook:** equal notional per maturity ignores DV01 (a 1bp move on the 30Y is worth ~20× more dollars than on the 1M — a real strategy would trade duration-weighted futures); not all 11 CMT points are tradable instruments; and the holdout-conditioning disclosure (Section 13) applies to the backtest too.

---

## 12. The results and the pre-registered decision rule

The decision rule was written into the notebook **before** it ran: adopt final3 if it clearly beats final2; report it as a robustness exercise if it ties; keep final2 and say so plainly if it loses.

**It lost.** On identical origins (2021-09 → 2026-04):

| Metric (h=20) | final3 Econ Ensemble | final2 Ensemble | Random Walk |
|---|---|---|---|
| Curve RMSE | 27.0 bp | **26.7 bp** | 28.0 bp |
| Weighted DA (no cost) | 0.588 | **0.668** | — |
| Weighted DA (1bp cost) | 0.567 | **0.646** | — |
| Profitable 126-day windows | 64% | **74%** | — |

DM statistics favor final2 at all three horizons (raw p = 0.04 at h=1). The economics features do carry signal — final3 still beats the random walk at h=5/20 — but everything they know, final2's factor-lag + regime + spillover set already knew, plus more.

**Why this is still a good result for the paper.** It converts final3 from a failed experiment into a robustness finding: *the limited predictability in the curve is spanned by simple statistical features; imposing the literature's economic structure does not add to it at these horizons.* That is consistent with the consensus that daily curve movement is dominated by unforecastable news, and it strengthens (rather than undermines) final2's claim by showing the result is not an artifact of one lucky feature set.

---

## 13. Every honest caveat in one place

1. **This holdout has been looked at.** The change-space redesign (and other fixes) were adopted after seeing earlier versions fail on this same holdout. Every table is conditioned on those prior looks. The planned three-month forward test — freeze the spec, append data as released, forecast before outcomes are known — is the only unconditioned evidence.
2. **The HSMM/decay were fit on the training 80%,** and final2's tuning ran on that same training data, which can flatter regime features slightly during selection (final3 sidesteps this by not using the HSMM).
3. **Overlapping forecasts** mean ~1,149 origins contain far fewer independent observations at h=20 (~57 non-overlapping windows); the DM and NW machinery accounts for this, but intuition should too.
4. **The trading backtest is a measurement device, not a strategy claim**: no DV01 weighting, no slippage model, CMT points aren't all tradable.
5. **After Benjamini–Hochberg,** the ensemble-vs-random-walk q-values are ~0.11 — "consistently favored at every horizon" is the defensible phrasing, not "statistically significant after multiple-testing control."
6. **T5YIE starts in 2003,** so features touching breakevens limit the earliest usable dates (irrelevant for the 2021+ holdout, relevant if you extend the evaluation back).

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **bp (basis point)** | 0.01 percentage points of yield |
| **H.15** | The Fed's daily statistical release of constant-maturity Treasury yields |
| **CMT** | Constant-maturity Treasury — an interpolated par yield at a fixed maturity, not a specific bond |
| **EFFR** | Effective federal funds rate — the realized overnight policy rate |
| **Breakeven inflation** | Nominal yield minus TIPS yield: the inflation rate at which both pay the same |
| **DNS** | Dynamic Nelson–Siegel: the 3-factor (level/slope/curvature) curve model, re-fit daily |
| **Loading** | How strongly a factor moves a given maturity |
| **Decay (λ)** | The NS parameter placing the curvature hump along the maturity axis |
| **Origin** | The date a forecast is made from |
| **Horizon (h)** | Trading days between origin and target |
| **Expanding window** | Training sample grows with each origin (vs a fixed-length rolling window) |
| **Ridge / α** | Linear regression with an L2 penalty; α sets shrinkage strength |
| **Random walk** | Forecast = today's value; the benchmark to beat |
| **VARX** | Vector autoregression with exogenous inputs — here, the Ridge factor-change model |
| **Ensemble weight (w)** | Fraction of the forecast taken from the VARX vs the random walk |
| **HSMM** | Hidden semi-Markov model — latent regimes with explicit duration distributions (final2 only) |
| **GFEVD / spillover** | Generalized forecast-error variance decomposition: how much of maturity i's surprise variance traces to shocks in maturity j |
| **DM test** | Diebold–Mariano: is one forecast's average loss significantly lower than another's? |
| **HLN correction** | Harvey–Leybourne–Newbold small-sample adjustment to DM |
| **BH / q-value** | Benjamini–Hochberg false-discovery-rate adjustment for multiple tests |
| **DA / weighted DA** | Directional accuracy; weighted DA = (1 + bp-capture)/2 from the trading backtest |
| **DV01** | Dollar value of a 1bp move — proportional to duration; why equal-notional yield bets aren't a real portfolio |
| **Leakage** | Any path by which information after the origin influences the forecast or a design choice |
