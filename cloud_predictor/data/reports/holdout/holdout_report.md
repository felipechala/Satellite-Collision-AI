# Held-out accuracy: NASA SBM vs SBM × learned multiplier

As of 2026-10-03. 5 random splits, about 20% of bus-family groups held out each time; each split trains the estimator on the rest and predicts the held-out events. 418 held-out predictions in total (294 distinct events). Target: fragments ≥10 cm ever cataloged. Analytics only: the shipped model trains on every event.

Errors are ratios, because counts span 1 to 3,500: *typical factor off* is the median of max(pred/actual, actual/pred), so ×2 means half the events are within a factor of 2.

## Overall (pooled over splits)

| Metric | NASA SBM | SBM × model (p50) |
| --- | --- | --- |
| Typical factor off | ×30 | ×3.19 |
| Median bias | 30× too many | 1.11× too many |
| Within 2× of actual | 6% | 34% |
| Within 10× of actual | 31% | 83% |
| RMSE of ln(pred/actual) | 3.67 | 1.76 |
| Actual inside p10–p90 band | — | 84% (target 80%) |

## Per split

| Split | Train labeled | Test | Test collisions | Head | NASA factor off | Model factor off | Model within 2× |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 303 | 78 | 1 | shipped | ×13 | ×4.35 | 26% |
| 1 | 287 | 94 | 1 | shipped | ×44 | ×3.23 | 34% |
| 2 | 307 | 74 | 1 | shipped | ×44 | ×3.77 | 26% |
| 3 | 289 | 92 | 0 | shipped | ×27 | ×2.29 | 41% |
| 4 | 301 | 80 | 1 | shipped | ×28 | ×2.60 | 40% |

## By event type and cause (pooled)

| Type | Cause | n | NASA factor off | Model factor off | NASA bias | Model bias | Model coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| collision | — | 4 | ×2.78 | ×2.78 | 2.28× too few | 2.28× too few | 100% |
| explosion | battery | 28 | ×14 | ×2.78 | 14× too many | 1.69× too few | 75% |
| explosion | deliberate | 70 | ×24 | ×1.84 | 24× too many | 1.06× too few | 87% |
| explosion | propulsion | 172 | ×34 | ×3.93 | 34× too many | 1.27× too many | 83% |
| explosion | unknown | 144 | ×48 | ×2.87 | 48× too many | 1.25× too many | 87% |

## Collisions held out

| Split | Event | Actual | NASA | Model p50 | Model p10–p90 |
| --- | --- | --- | --- | --- | --- |
| 0 | discos-182 | 3536 | 1272 | 1272 | 549–4904 |
| 1 | discos-224 | 2375 | 1274 | 1274 | 550–6371 |
| 2 | discos-182 | 3536 | 1272 | 1272 | 549–4904 |
| 4 | discos-111 | 288 | 835 | 835 | 94–1150 |
