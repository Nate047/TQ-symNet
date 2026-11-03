# Result collection (no training or inference)

Copy `paper_results.example.json` to `paper_evidence/runs.json` under the project.
All relative paths resolve against the directory containing runs.json.
Save COMPLETE platform output as `paper_evidence/online_c20_a0.txt` and
`paper_evidence/online_c20_m2_soft.txt`. Export progress logs are not scores.
Replace submission IDs and log filenames. Delete example entries you do not
have, and add real ablation logs with their actual cohorts/selected epochs.
The example 8 epochs and epoch 7 apply only if confirmed by that A0 run.

```bash
python tools/paper_visualization/collect_paper_results.py \
  --manifest paper_evidence/runs.json --out-dir paper_evidence/tables
```

Supported kinds: `online` (platform text), `val_log` (MMDet training log),
`val_metrics` / `test_metrics` (evaluate.py metrics.json). Test entries must
declare `selected_on_val: true`; this declaration is not independently proven.
Always supply the actual evaluation `annotation` for labeled datasets to record
its SHA256. `cohort` labels are explicit user declarations, not inferred from
filenames. Review `audit.json` before using tables. Rounded log values can tie;
prefer `selected_epoch` verified from the checkpoint/runner metadata.

Outputs: separate percentage-unit CSV tables per cohort/kind, all validation
curves, source hashes/audit warnings, and a Markdown evidence report. No AP is
estimated from classification sensitivity. No missing metric is filled with zero.

## Existing plots: use identical original validation annotations

```bash
python tools/paper_visualization/plot_quality_effect.py \
  --baseline paper_visualization/b1/visualization.json.gz \
  --quality paper_visualization/a0/visualization.json.gz \
  --out-dir paper_figures

python tools/paper_visualization/plot_predictions.py \
  --runs paper_visualization/b1/visualization.json.gz \
         paper_visualization/a0/visualization.json.gz \
  --out-dir paper_figures --cases 4
```

These commands require previously exported visualization.json.gz files. They
do not create missing inference data. Use export_predictions.py with the actual
annotation path to regenerate if necessary; current configs may point to a newer
120-image split. Never equate those plots with original 200-image validation.

Main text: overall architecture, paired GT selection/coverage, representative
improvement and failure examples. Supplement: full score/IoU distribution and
duplicate/localization/background FP composition. Do not discard missed GTs.
Display score thresholds affect images only, not metrics. No GT-overlay claims
for the hidden test set. Audit fixed-versus-learned score fusion in the final
checkpoint before using historical framework drawings; this collector does not
verify or rewrite their architecture labels.
