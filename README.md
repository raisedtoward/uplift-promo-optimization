# Uplift-based promo assignment

Take-home case study: pick a voucher (0 / 10 / 20 / 30 / 40 %) for each eligible user for the upcoming week. 
We use a randomised experiment data from weeks 9–14 to learn optimal assignment policy while controlling for expected spend to stay within a 3,500 CU budget total.

All the conducted research is in [`notebooks/Uplift-Based Promo Assignment Research + EDA.ipynb`](notebooks/Uplift-Based%20Promo%20Assignment%20Research%20\+%20EDA.ipynb).
This README walks through it in the same order and keeps only the main ideas and findings.

| Deliverable Result                   | Location                                                                                                                                                                                                                                                             |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Decisions file                       | `outputs/decisions.csv`: `user_id`, `voucher_pct` for week 15 (not week 16, explained further)                                                                                                                                                                       |
| Code                                 | `src/` (cleaned up and reproducible pipeline based on notebook exploration logic), `atlas/` and `pulsar/` (modelling, analysis and causal inference helper libraries in a reduced form that have been maintained and reused across projects), the notebook |
| Presentation slide deck              | `docs/slidedeck.pdf`                                                                                                                                                                                                                                                 |
| Technical notes, next steps proposal | this `README.md` file                                                                                                                                                                                                                                                  |

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and `just`.

```bash
uv sync
```

Put `history.parquet` and `inference.parquet` into `data/`. 
The folder is git-ignored because the case data is confidential.

```bash
just run    # train, evaluate models, perform policy spend check on week-14, perform assignments on week 15, write outputs
just lab    # jupyter lab for the notebook
```

## 1. Data and primary sanity checks

- The same `user_id` is assumed to be the same physical user in history and inference; almost all users appear in both.
- The weekly population is very stable (~60k users per week). It looks like a weekly slice using the same eligibility criteria rather than one fixed cohort.
- All 10 arms (5 vouchers × 2 free-delivery levels) are balanced in size, and each user gets exactly one treatment per week.
- A user's arm (treatment assignment) changes from week to week, so this is a sequentially randomised (micro-randomised) design. We flatten it into user-week rows, and the estimand is *the effect of this week's voucher, given history*.
- Treatment works on average: conversion grows with the discount (about 19.6% at 0% with no free delivery, up to ~34% at 40%), and free delivery adds a bit on top within every voucher level.
- **Budget:** 3,500 CU is ambiguous (per week or for both weeks?) and an order of magnitude below historical weekly voucher cost (35–42k CU). We treat it as 3,500 CU across two weeks, so 1,750 CU per week.
- **Week 16:** its features (`orders_30d`, `days_since_last_order`, …) already reflect week 15, as if the week-15 decision had already been made and its outcome observed. This is not right, we can't know them at decision time and moreover we're trying to impact users' behavior by intervention which surely must reflect in some of the features that are already provided which directly breaks causality. It is also not how production flow would look like for exactly same reason - either all users are observed once in the earlier week and the step is 2 weeks or decisions are done weekly. That said, **we don't score week 16**: the decision file covers week 15 only.

## 2. Monetary target switch?

`ordered` is a binary flag: we know neither how many orders were placed nor their value. We first tried to naively reconstruct weekly order value from consecutive weeks as `orders_90d × aov_90d (t+1) − orders_90d × aov_90d (t)`. That didn't work, for two reasons:

- about 4% of user-weeks have no next-week observation (outflow). The rate is flat across weeks, ordering groups and arms, so some unknown eligibility rule is probably at play;
- the 90-day window is rolling, not expanding, so the difference goes negative whenever older orders leave the window.

So we model the binary `ordered` outcome. New users enter the population at a similar rate (~4.2% per week). Week 9 is dropped because it has no previous-week features (lag 1w post-treatment (wrt previous week, not current) covariates), which we used for exploration of the carry-over effect and boosting model's performance.

## 3. Things checked before modelling

- **Cities** behave the same, so `city` is kept only as an encoded feature, no significant need in additional upfront segmentation.
- **Carry-over:** in a linear probability model with week fixed effects, last week's voucher has no significant effect on this week's conversion (continuous p = 0.46; all four lagged levels p > 0.28). Its interaction with the current voucher isn't significant either. The outflow rate isn't related to treatment. So weekly exposures can be treated as independent, and a myopic one-week policy is fine. Lagged features though stay in the model as covariates. 
- **Free delivery** is fixed for us at inference. We weighed three options: pool FD out, use FD as a covariate, or model the FD assignment mechanism (might be used later based on FD assignment policy which currently is unknown to us but seems random (unexplained by known covariates). The joint Wald test on the voucher × FD interactions is borderline (p = 0.059). The conversion curves are parallel up to 10% and converge at higher discounts, so FD and vouchers act as substitutes. **FD goes into the model as a covariate**, and at inference we plug in the given FD value.
- **Discrete treatment:** five fixed levels, no inter- or extrapolation needed.
- **Randomisation checks:** a propensity model can't predict the voucher arm from features (macro ROC-AUC 0.50, log-loss 1.611 vs 1.609 for a dummy). The same holds for FD in the inference data (ROC-AUC 0.49, a 50/50 split). Assignment looks random, so we expect a week-14 out-of-time evaluation to carry over to week 15 and don't use additional inverse propensity reweighing for modelling (only using uniform 1/5 over arms during policy evaluation).

## 4. Conversion uplift modelling

Split: user-week observations are split by **user_id** 80/20 first, then accordingly into train and test samples for weeks 10-13, so no user lands in both. Full week 14 is held back as out-of-time (OOT), and some users on week 14 is a clean holdout.

Features: the 12 user aggregates, previous-week lags (`voucher_pct`, `free_delivery`, `aov_90d`, `orders_90d`, `ordered`, `voucher_used`, `voucher_cost`), the encoded city, and FD.

Univariate CATE ranking (causalml `FilterSelect`, F / KL / ED) puts `pct_orders_with_promo_90d`, `voucher_redemption_rate_90d` and `orders_90d` on top. FD ranks last: it shifts conversion, but treatment effects hardly change with it, unlike `push_opt_in`, for example.

We compared three multi-treatment learners: S-learner (LightGBM), uplift tree (UT) and uplift random forest (URF). Qini by voucher arm vs control:

| Sample                | S | UT | URF |
|-----------------------|---|---|---|
| train                 | 0.083 | 0.130 | 0.205 |
| test                  | 0.059 | 0.036 | 0.066 |
| out-of-time (week 14) | 0.055 | 0.041 | 0.054 |
| holdout (week 14)     | 0.049 | 0.049 | **0.062** |

<details><summary>Qini per voucher level (OOT / holdout)</summary>

| Treatment | S | UT | URF |
|-----------|---|---|---|
| 10%       | 0.024 / 0.023 | 0.009 / 0.016 | 0.009 / 0.024 |
| 20%       | 0.043 / 0.041 | 0.026 / 0.033 | 0.041 / 0.055 |
| 30%       | 0.066 / 0.052 | 0.056 / 0.062 | 0.069 / 0.060 |
| 40%       | 0.089 / 0.080 | 0.074 / 0.085 | 0.096 / 0.107 |

</details>

URF is calibrated and is the most stable for the higher vouchers. 10% is noisy for every learner. URF has the best macro Qini on the holdout, so **we go with URF**.

We didn't use T-, X-, R- learners due to them being based on 1-1 comparison with control and not having implementation at hand for fast experimentation (currently used version of atlas package attached is the one i maintained during previous project, but those learners currently only support binary 0-1 treatment assignments)
We also didn't optimize hyperparameters much, as well as backward feature selection based on feature importance - this library version only support it for 0-1 for all learners.
This is to be updated later and hopefully when we continue evolving this framework together :-)

Evaluation was done on 1-1 control-treatment_i basis and used macro Qini aggregation, calibration analysis through bucketed bootstrap, bucketed uplift ranking and lift curves. 

## 5. Problem formulation

For each user and voucher level we have the following inputs:

- conversion uplift `Δ(d) = P(order | d, fd, x) − P(order | 0, fd, x)` coming from URF
- `E(AOV|order next week)`, which we never observe after treatment, so we take `aov_90d` and assume vouchers don't change it significantly
- utilization `u(d)` and cost per used voucher `c(d)`
- a weekly budget `B`

**Objective: maximise incremental order value `Σ AOV_i · Δ_i(d_i)`**, with at most one voucher per user and expected total spend ≤ B.

Why we chose value rather than plain conversion: the average effect grows with the voucher, but efficiency falls. 
Conversion uplift per CU spent per user drops from 0.114 at 10% to 0.091 at 40%. 
Optimising orders alone could also buy conversions from users with low baskets.
We're still gaining incremental orders and could communicate to the stakeholders that it is very likely what we need, but re-adjustment to pure orders incrementality is not hard at all.
Problem formulation could be segmented further if we want to build ordering habits for new users vs retaining churning ones.

One small issue with this we have tried to tackle:

- **AOV for the ~19% of users with no `aov_90d`.** Leaving those users at 0 would zero out their objective. A LightGBM regressor on all features imputes values that are out of distribution for non-orderers. One trained only on 4 engagement features (`sessions_30d`, `days_since_last_session`, `push_opt_in`, `offers_tab_share_of_sessions_30d`) explains less, but its forecast for non-orderers stays close to the observed distribution. We use real `aov_90d` where it exists and that forecast elsewhere.

What we tried and eventually discarded:

- **An order-count multiplier `E[N_orders | ordered = 1]`** as a proxy for next week's value. Under a Poisson fit on `orders_30d` it averages ~1.13, and it underestimates even the binary conversion rate for heavier users. With less than one order per user per month on average, it isn't worth including.
- Naive group-based projections for `u(d)` and `c(d)` decoupled from expected `AOV`, see next section for details.


The problem is an LP problem: the assignment matrix `X ∈ [0, 1]^{N×4}` with row sums ≤ 1 and `Σ S ⊙ X ≤ B`, solved with cvxpy + HiGHS. Each row's argmax gives the assigned voucher.

## 6. Spend control and problem formulation: three iterations

We solved for B = 1,750 CU on week 14 (OOT), then measured what the selected users actually cost and how much we gain. 
We used off-policy evaluation using actual experimental data: an IPW estimate for our policy against control mode (no-voucher arm) and the randomised assignment design (actual experiment vs control mode).

| # | Expected spend per user (as constraints formulation)                                           | Users with a voucher | 10 / 20 / 30 / 40 % | Planned spend (as of optimization results) | Realised (Policy evaluation) | Realised / planned |
|---|------------------------------------------------------------------------------------------------|---|---|--------------------------------------------|------------------------------|---|
| v1 | `P(ordered \| d) · P(voucher_used \| ordered, d, fd) · E[voucher_cost \| voucher_used, d, fd]` | 8,371 (14%) | 7,828 / 365 / 138 / 40 | 1,750                                      | 3,607                        | 2.06 |
| v2 | `P(voucher_used \| d, fd) · min(d · AOV, 9)`                                                           | 7,107 (12%) | 6,196 / 684 / 152 / 75 | 1,750                                      | 5,368                        | 3.07 |
| v3 | `P̂(voucher_used \| d, fd, x) · min(d · AOV, 9)`                                                       | 5,169 (9%) | 4,231 / 346 / 135 / 457 | 1,750                                      | 2,237                        | 1.28 |

- **v1** uses arm averages. The policy slightly beats random assignment on cumulative gain, but the users it picks use vouchers far more often than the arm average (18% vs 11%), so spend is about twice the plan.
- **v2** drops the conditioning on ordering and ties cost to each user's `AOV` instead of the arm average. The policy improves, but overspend gets even worse: utilization among selected users still doesn't match the arm averages as it's much higher.
- **v3** adds a per-user utilization model (unconditioned on order fact, conditioned on treatment) - LightGBM using pre-treatment features and FD, ROC-AUC 0.78. Spend is still somewhat underestimated, but much less and gain curve has improved.

In every version most vouchers go to 10%: it is the cheapest, and it still moves conversion. In the per-cohort ATE checks, the cohorts that get a voucher show higher uplift than the cohorts that don't, and uplifts are positively correlated across voucher levels. 
The policy appears to find users for whom a bigger voucher would also work, and saves cost by giving them a smaller one.

## 7. Week 15 assignment

- Budget: 3,500 / 2 = 1,750 CU for week 15, divided by a conservative factor of 1.25 (week 14 showed 1.28) → 1,400 CU planned.
- Inputs: URF uplift, the blended AOV, the v3 spend model, and the FD given in `inference.parquet`.
- Result: 4,197 users get a voucher (3,523 at 10%, 220 at 20%, 100 at 30%, 354 at 40%). Planned spend is 1,400 CU (1,750 CU after applying the 1.25 factor), and the expected incremental order value is 13,650 CU.

## Assumptions and limitations

- Vouchers and FD don't change AOV or orders per converter. There is no post-treatment data to check this.
- The binary weekly `ordered` flag can't show extra orders from users who would have ordered anyway.
- The FD assignment we see at inference looks random given X. If production FD followed an unobserved rule (for example, giving it to price-sensitive users), the experiment couldn't reveal that, and the uplift for FD = 1 users would be off.
- Mechanics, content and voucher usage flow for each voucher level stay the same over time.
- ~4% of users leave the population each week for unknown reasons. We can't (and likely shoudln't) exclude them in advance and follow the Intent-To-Treat design.
- The spend correction (1.25) comes from a single out-of-time week.

## Next steps and data needs

- **Long-term effects.** The current design can't measure effects beyond one week, because later random vouchers mix into the outcome. We might want an experiment that tracks several weeks with no follow-up vouchers (or conditions on them).
- **Order value and order count** tracked per user-week after treatment, so we can optimise value directly instead of using `aov_90d` as a proxy.
- **Outflow:** why ~4% of users drop out each week, and whether they should be eligible at all. Does business actually want to segment further (users blocked by integrity team, completely new, etc.)?
- **Production FD logic,** to rule out FD being driven by something the experiment can't see.
- **Inference data one week at a time,** so features never reflect decisions that haven't been made yet (the week-16 issue) and we don't accumulate the error.
- **Budget definition** (per week or total) and how it relates to historical spend.
- **Modelling**: per-user order-count models (for example, negative binomial), and direct policy learning (policy trees) if FD becomes uncertain at decision time, explore other meta-learners with automated tuning and model selection.
- **Evaluation**: employ bootstrap methods for policy evaluation to account for high variance and not rely solely on point-wise estimates. 

## Next experiment design proposal

Run the optimized policy against the current random assignment mechanism in a randomised split of users, with the same budget on both sides or split accordingly to risk-appetite:

- **Split**: policy vs no-voucher control subgroup vs mixed 10-20-30-40 - reevaluate gain/cost tradeoff and compare with offline evaluation
- **Primary metrics**: incremental order value, incremental conversion   
- **Guardrails**: realised voucher spend vs plan, utilization rate, communications delivery funnel rates
- Define MDE for both metrics of interest (1-10% relative), conduct A/A test on historical data to control for FPR and understand minimal group sizes for split
- Collect the post-treatment order value and order counts listed above, margins information and delivery cost
- We still rather have some clean stream of randomized assignments even if it's small to check for directional sanity checks and to be used for re-calibration 
- Explore what other currently unavailable features might contribute to CATE identification 
