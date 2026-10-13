# Frozen M2-20e epoch17: quality ablations and final test

The detector is locked by validation AP=0.362 (epoch17). All six quality groups
use this exact checkpoint, train for 8 epochs, and select their own best epoch
using the same 120-image validation split. Final results use the fixed 200-image
test split, never test-based checkpoint selection. B0 is the detector's existing
hard-NMS output; B1 uses the same detector with Soft-NMS. A0 vs B1 isolates the
quality-network contribution from changing NMS. Execution order: B0/B1 test
first, then train each planned quality group and immediately test its VAL-best.
The six configurations are fixed before reading these test results.

Copy the complete `configs/quality_m2_20e/` folder plus `tools/preflight.py` and
`tools/evaluate.py`. Existing balanced-v2 labels/core files remain required.
The original root-level A configs retain the earlier 16-epoch detector setting.
Use the NEW folder below and fresh work directories; do not mix prior A results.

```bash
cd /home/data2/LiYingxian/TQ_SymNet
PY=/home/data2/LiYingxian/Tuberculosis-main/conda_env/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONHASHSEED=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN=work_dirs/balanced_v2_quality_m2_20e_seed0
DET=work_dirs/balanced_v2_seed0_479_120_200/m2_20e_s0/best_official_tb_bbox_mAP_epoch_17.pth

# First test the locked detector. B0 is also the M2-20e test result.
for CFG in b0_sym_hard b1_sym_soft
do
  "$PY" tools/evaluate.py "configs/${CFG}.py" "$DET" \
    --split test --mode tb --out-dir "$RUN/test_${CFG}" || exit 1
done

# Sequential single-GPU training: after EACH group, test its VAL-best checkpoint.
for CFG in a0_full a1_no_texture a4_no_detail a7_no_rank a8_direct_iou a12_direct_iou_no_rank
do
  GROUP="${CFG%%_*}"
  if [ -d "$RUN/${GROUP}_s0" ]; then
    echo "Existing run directory: $RUN/${GROUP}_s0; do not overwrite or mix runs."
    exit 1
  fi
  "$PY" tools/preflight.py --config "configs/quality_m2_20e/${CFG}.py" --server-files --build || exit 1
  "$PY" tools/train.py "configs/quality_m2_20e/${CFG}.py" \
    --seed 0 --deterministic --work-dir "$RUN/${GROUP}_s0" || exit 1
  BEST=( "$RUN/${GROUP}_s0"/best_official_tb_bbox_mAP_epoch_*.pth )
  if [ "${#BEST[@]}" -ne 1 ] || [ ! -f "${BEST[0]}" ]; then
    echo "Expected one VAL-best checkpoint in $RUN/${GROUP}_s0"
    exit 1
  fi
  "$PY" tools/evaluate.py "configs/quality_m2_20e/${CFG}.py" "${BEST[0]}" \
    --split test --mode tb --out-dir "$RUN/test_${GROUP}_tb" || exit 1
  if [ "$GROUP" = a0 ]; then
    "$PY" tools/evaluate.py "configs/quality_m2_20e/${CFG}.py" "${BEST[0]}" \
      --split test --mode subtype --out-dir "$RUN/test_a0_subtype" || exit 1
  fi
done
```

Every evaluation writes metrics.json, evaluation.log, resolved_config.py and
COCO predictions. Report official_tb_bbox_mAP/AP50/AP75 for the TB comparison;
report Active/Obsolete AP from the explicitly designated subtype output mode.
The evaluation also verifies the quality checkpoint's recorded detector SHA,
preventing accidental use of an A checkpoint trained on the 16-epoch model.
These are the local 200-image only-TB test results, not the online challenge.

This quality ablation family does not establish a matched-budget backbone
comparison between a 20-epoch M2 and previously trained 16-epoch M0/M1.
