# Held-out accuracy: NASA SBM vs SBM × learned multiplier

As of 2026-10-03. 5 random splits, about 20% of bus-family groups held out each time; each split trains the estimator on the rest and predicts the held-out events. 418 held-out predictions in total (294 distinct events). Target: fragments ≥10 cm ever cataloged. Analytics only: the shipped model trains on every event.

Errors are ratios, because counts span 1 to 3,500: *typical factor off* is the median of max(pred/actual, actual/pred), so ×2 means half the events are within a factor of 2.

## Overall (pooled over splits)

| Metric | NASA SBM | SBM × model (p50) |
| --- | --- | --- |
| Typical factor off | ×30 | ×3.69 |
| Median bias | 30× too many | 1.42× too many |
| Within 2× of actual | 6% | 29% |
| Within 10× of actual | 31% | 78% |
| RMSE of ln(pred/actual) | 3.67 | 1.87 |
| Actual inside p10–p90 band | — | 81% (target 80%) |

## Per split

| Split | Train labeled | Test | Test collisions | Head | NASA factor off | Model factor off | Model within 2× |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 303 | 78 | 1 | shipped | ×13 | ×3.69 | 27% |
| 1 | 287 | 94 | 1 | shipped | ×44 | ×5.48 | 20% |
| 2 | 307 | 74 | 1 | shipped | ×44 | ×4.39 | 18% |
| 3 | 289 | 92 | 0 | shipped | ×27 | ×2.69 | 40% |
| 4 | 301 | 80 | 1 | shipped | ×28 | ×2.78 | 38% |

## By event type and cause (pooled)

| Type | Cause | n | NASA factor off | Model factor off | NASA bias | Model bias | Model coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| collision | — | 4 | ×2.78 | ×27 | 2.28× too few | 27× too few | 25% |
| explosion | battery | 28 | ×14 | ×4.42 | 14× too many | 1.31× too many | 82% |
| explosion | deliberate | 70 | ×24 | ×1.89 | 24× too many | 1.03× too few | 94% |
| explosion | propulsion | 172 | ×34 | ×4.36 | 34× too many | 1.45× too many | 80% |
| explosion | unknown | 144 | ×48 | ×3.78 | 48× too many | 2.47× too many | 76% |

## Collisions held out

| Split | Event | Actual | NASA | Model p50 | Model p10–p90 |
| --- | --- | --- | --- | --- | --- |
| 0 | discos-182 | 3536 | 1272 | 28 | 5–450 |
| 1 | discos-224 | 2375 | 1274 | 185 | 7–966 |
| 2 | discos-182 | 3536 | 1272 | 62 | 5–755 |
| 4 | discos-111 | 288 | 835 | 48 | 3–1174 |
