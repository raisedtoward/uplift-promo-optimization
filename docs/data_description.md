# Personalised promotions dataset

## Context

`e-commerce.co` is an online marketplace. Each row represents one eligible user and one
weekly promotion decision. A user may receive a voucher worth 0%, 10%, 20%, 30%, or 40% of
an order.

Some users also receive free delivery. This is a separate promotion that has already been
decided, so `free_delivery` must be treated as fixed when assigning vouchers.

All monetary values are expressed in currency units (CU).

## Files

- `data/history.parquet` contains eligible user-weeks from weeks 9–14. It includes features
known before the decision, the randomly assigned promotion, and the observed outcomes.
- `data/inference.parquet` contains eligible user-weeks from weeks 15–16. It includes only
information available when making the decision. You must add a voucher decision for every
row.

No additional tables need to be joined.

## Historical assignment

During weeks 9–14, eligible users were randomly assigned to one of ten combinations:

```text
voucher_pct   ∈ {0, 10, 20, 30, 40}
free_delivery ∈ {false, true}
```

Each combination was assigned to approximately 10% of eligible users within each city. This
randomised history can be used to estimate how the available decisions affect ordering and
voucher cost.

For weeks 15–16, `free_delivery` has already been assigned and appears in
`inference.parquet`. Only `voucher_pct` remains for you to choose.

## Voucher use and cost

`voucher_used` records whether the assigned voucher was used on a qualifying order.

- When `voucher_used` is `false`, `voucher_cost` is 0.
- When `voucher_used` is `true`, `voucher_cost` is the actual discount paid by the
marketplace and is greater than 0.
- A 0% voucher cannot be used and has zero cost.

Every voucher has a maximum discount of 9 CU. In other words, a 40% voucher means
“40% off, up to 9 CU.” The marketplace's voucher cost is the discount given to the
customer:

```text
voucher_cost = min(voucher_pct / 100 × order value, 9 CU)
```

## Columns

### Identifiers and decision context

- `user_id` (integer): user identifier.
- `week` (integer): 9–14 in history and 15–16 in inference.
- `city` (string): the user's city.
- `free_delivery` (boolean): whether the user receives free delivery that week.

### Previous order behaviour

- `orders_90d` (integer): orders in the previous 90 days.
- `orders_30d` (integer): orders in the previous 30 days.
- `days_since_last_order` (integer): days since the user's last order; null if none.
- `aov_90d` (float): average order value in CU over the previous 90 days.

### Previous promotion behaviour

- `pct_orders_with_promo_90d` (float): share of previous-90-day orders using a promotion.
- `avg_voucher_pct_received_90d` (float): mean voucher percentage previously assigned over
90 days.
- `voucher_redemption_rate_90d` (float): vouchers used divided by vouchers received over
90 days.
- `manual_voucher_entry_rate_90d` (float): share of previous-90-day orders for which the
user manually entered a voucher code.

### Engagement

- `sessions_30d` (integer): app sessions in the previous 30 days.
- `days_since_last_session` (integer): days since the user's last app session.
- `push_opt_in` (boolean): whether push notifications are enabled.
- `offers_tab_share_of_sessions_30d` (float): share of previous-30-day sessions that
included the Offers tab.

### Historical decision and outcomes

These columns appear only in `history.parquet`:

- `voucher_pct` (integer): randomly assigned voucher percentage.
- `ordered` (boolean): whether the user placed at least one order that week.
- `voucher_used` (boolean): whether the assigned voucher was used.
- `voucher_cost` (float): actual voucher cost paid by the marketplace in CU.

All feature values describe information available before the weekly promotion decision.
Trailing rates may be null when a user has no relevant previous observations.

## Reading the files

```python
import pandas as pd

history = pd.read_parquet("data/history.parquet")
inference = pd.read_parquet("data/inference.parquet")
```

The files require a Parquet reader such as `pyarrow`, Polars, or DuckDB.