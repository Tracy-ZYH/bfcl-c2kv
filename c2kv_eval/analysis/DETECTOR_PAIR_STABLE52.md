# Detector-only stable52 development comparison

Run `bash c2kv_eval/scripts/run_detector_pair_stable52_card6.sh` from the
existing tmux. It uses NPU 6 sequentially (rule, then trained), a fresh output
root, current project source, and the existing sgl environment. The runner
sources both Ascend environment scripts. No NPU run was performed during setup.

Both variants keep ratio=4, checkpoint interval=4, temperature=0, Replace W2,
the existing recovery selector, frozen Full reference, and official BFCL scoring.
Features reuse the recovered detector_features.csv (213 segments). The target
is reference_drift; neither the score nor its training label is Task Error.

* trained: Logistic with L2 regularization (ElasticNet l1_ratio=0), three fixed
  causal features: mean_observation_anomaly, mean_argument_grounding_failure,
  max_hard_error. Equal total training weight per episode; C selected using
  episode-disjoint inner validation. L1 is omitted because it collapsed the
  three-feature model to a constant on one development fold.
* rule: the existing max_risk_score, with no fitted coefficients or fitted
  normalization. A fixed sigmoid serializes it through the existing scorer;
  this does not change its ranking. Threshold calibration still uses labels
  via official calibration scores, so this is training-free, not calibration-free.

The two variants share seed=20260905 and the same five outer folds:

| Fold | Model train | Calibration | Test |
|---|---:|---:|---:|
| 0 | 37 | 7 | 8 |
| 1 | 34 | 8 | 10 |
| 2 | 32 | 10 | 10 |
| 3 | 28 | 14 | 10 |
| 4 | 27 | 11 | 14 |

Online never-trigger shadow on calibration episodes determines achievable
score thresholds near 45/65/85 percent triggering. Ties are not randomly
broken and duplicate behaviours are not counted as separate operating points.
Only these candidates undergo closed-loop calibration. Select highest BFCL
accuracy, then closest trigger rate to calibration Oracle. No test Oracle
information selects a threshold. Every held-out episode is evaluated under
selected detector, Oracle, and never-trigger. Summary counts are weighted by
episode count rather than taking an unweighted average of unequal folds.

Primary outcome: official BFCL correct/52 and percentage-point gap to Oracle.
Read this alongside trigger rate, model calls, recovery diagnostics, and
imputation diagnostics in each variant's existing v3 reports. A recovery
reference-match rate is not a task-repair success rate. Missing selected
features in calibration shadow fail explicitly rather than silently imputing.

Both trained models and fixed-rule artifacts were built on CPU and split
disjointness was checked. End-to-end NPU validity and accuracy remain to be
measured. stable52 is a previously used Full-success-filtered development set;
this is not a fresh unbiased generalization estimate. The sparse existing
signals may not distinguish semantic failures even after calibration.

Outputs: rule/, trained/, rule.log, trained.log, summary.csv, summary.md.
Each variant includes calibration diagnostics and held-out Oracle/never-trigger
directories. Existing environment directories and recovered experiment outputs
are not modified by this workflow.
