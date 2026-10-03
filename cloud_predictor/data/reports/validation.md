# Validation report

As of 2026-10-03. 480 events and 456 spec records read from `D:\CodeProjects\Satellite-Collision-AI\cloud_predictor\data`.

- **453** events pass `ml.data.build_training_rows`; **27** are skipped.
- 116 distinct CV groups (GroupKFold needs at least 2).

## Labeled events per head

A head is trained only with at least 30 labeled events. A/M and delta-v heads need fragments.csv (deferred to v2), so they fall back to neutral bands for now.

| Head | Labeled | Enough to train |
| --- | --- | --- |
| `n_multiplier` | 381 | yes |
| `am_mu_shift` | 0 | no |
| `am_sigma_scale` | 0 | no |
| `dv_mu_shift` | 0 | no |
| `dv_sigma_scale` | 0 | no |

## Skipped events

| Event | Reason |
| --- | --- |
| discos-355 | impactor: is required for collisions |
| discos-133 | parent_norad_id: no matching spec record |
| discos-5 | impactor.v_rel_km_s: must be a finite number |
| discos-164 | impactor: is required for collisions |
| discos-244 | impactor: is required for collisions |
| discos-207 | impactor: is required for collisions |
| discos-192 | impactor: is required for collisions |
| discos-749 | impactor: is required for collisions |
| discos-206 | impactor: is required for collisions |
| discos-163 | impactor: is required for collisions |
| discos-171 | impactor: is required for collisions |
| discos-752 | impactor: is required for collisions |
| discos-180 | impactor: is required for collisions |
| discos-748 | impactor: is required for collisions |
| discos-753 | impactor: is required for collisions |
| discos-198 | impactor: is required for collisions |
| discos-208 | impactor: is required for collisions |
| discos-231 | impactor: is required for collisions |
| discos-69 | impactor: is required for collisions |
| discos-282 | impactor: is required for collisions |
| discos-517 | parent_norad_id: no matching spec record |
| discos-650 | impactor.mass_kg: must be a finite number |
| discos-743 | impactor: is required for collisions |
| discos-784 | impactor: is required for collisions |
| discos-861 | impactor: is required for collisions |
| discos-889 | impactor: is required for collisions |
| discos-945 | impactor: is required for collisions |
