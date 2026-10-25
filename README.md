# MaskCluster_SSL

Image-only self-supervised person retrieval for Market-1501. The central MaskCluster objectives are implemented locally with general PyTorch components. Training reads only image pixels; identity and camera metadata are used only for validation and final evaluation.

## Setup

Use Python 3.11 in an environment named `MaskCluster_SSL`. On this Windows machine:

```powershell
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -m pip install -e ".[dev]"
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -m pip check
```

The verified local CPU direct pins are in `requirements-lock-windows-cpu.txt`. Each run also records a complete installed-package inventory in `run_metadata.json`. Neither is a verified Linux/CUDA lock. On Linux, choose the official CUDA wheel pair for the assigned GPU and driver before installing the project; record the resulting environment there. Put `training.h5` and `validation.h5` under `Input/raw`, or edit their paths in the config. Keep `testing.h5` untouched until final evaluation.

## Method

`baseline` has global clustering and global ME-MAX. `no_mim` adds aligned patch clustering and patch ME-MAX. `full` adds masked RGB reconstruction. The student learns by gradients; an identically initialized EMA teacher supplies clean global targets. Shared weights, sample order, crops, photometry, masks, and corruption use separate deterministic streams across variants. The six ViT-S/16 attention heads have 64 dimensions each. Four local crops, 1,024 latent prototypes, and the clustering head's 1,024/256 hidden and bottleneck dimensions are provisional resource and design choices. Prototypes are never identity labels.

The primary `visible` attention mode follows the paper's equations 9–11: visible tokens including CLS attend to visible keys/values; masked queries receive visible context and remain positioned for reconstruction. `standard` attention is configurable. We interpret CLS as visible because the paper does not settle that detail. Intermediate layers 6/8/10/12 are summed without an extra final LayerNorm. The MLP pixel decoder simplifies the paper's convolutional decoder. Masked L1 is averaged rather than summed and uses FP32 differences/reductions; its weights remain provisional because the scale changes. Global input is 128×64 and local input 96×48. Moderate brightness, contrast, saturation, and occasional grayscale are common to all variants; teacher/student patch geometry stays aligned. These provisional values preserve identity cues more conservatively than a full DINO augmentation set. See `docs/ADAPTACION_PAPER.md`.

Position embeddings are added once before the transformer blocks. Dense visible attention reproduces the information flow but has no demonstrated speedup.

## Training and checks

The full training config is provisional. `train.py` starts training; the other commands below are bounded checks.

```powershell
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -B -m pytest tests -q -p no:cacheprovider
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -B scripts/verify_cpu_smoke.py --config configs/smoke_cpu.yaml --device cpu
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -B scripts/train.py --config configs/train.yaml --variant full
```

Run `train.py` separately for `baseline`, `no_mim`, and `full` after fixing a common pilot budget. `--resume PATH` continues at the next epoch. Checkpoints include model, optimizer, AMP scaler, counters, best scores, config, Python/NumPy/torch CPU/CUDA RNG, supplied generator states, and SHA-256 signatures of training and validation files. The schedule follows the unchanged config and global step. Resume rejects scientific config changes and unverified data changes. Output path, device, worker count, and evaluation batch/block sizes may change. Data paths may move when both file signatures match. CPU checkpoints without AMP state initialize a new scaler when AMP is enabled and log that transition. Epoch extension needs a future explicit operation.

Same-device, same-config CPU continuation is tested. Save points follow complete epochs. Per-sample streams and epoch-indexed sampling make inputs independent of worker count, including persistent workers, but a crash inside an epoch restarts it. CPU-to-GPU restoration does not promise numerical equivalence. CUDA/AMP and persistent-worker execution remain to be checked on target hardware.

## Evaluation

Each call evaluates one checkpoint encoder on original and low-resolution queries against the same original gallery. It uses L2-normalized CLS embeddings, cosine similarity, Market-1501 exclusions, mAP, Rank-1/5/10, and mAP drop in percentage points. Valid-query Rank-1 defines success or failure. Rankings include IDs, cameras, similarities, and relevance; the default export keeps the top 100 valid entries per query. Five successes and five failures per condition are selected by query index for image panels. The JSON reports actual counts if fewer exist. Human interpretation of errors is still required.

```powershell
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -B scripts/evaluate.py --checkpoint outputs/runs/RUN/best_teacher.pt --data Input/raw/validation.h5 --split validation --encoder teacher --output outputs/evaluations/RUN_teacher
& "C:\Users\daryl\anaconda3\envs\MaskCluster_SSL\python.exe" -B scripts/compare_results.py outputs/evaluations/RUN_teacher/metrics.json --output outputs/evaluations/draft.csv --allow-partial
```

Each evaluation report records the actual `--data` path, SHA-256 file and gallery signatures, group counts/shapes, and preprocessing. Pass all six completed baseline/full/no_mim teacher/student reports to `compare_results.py` for the final table. It rejects different datasets, galleries, splits, or preprocessing while allowing model variants. `--split testing` is reserved for one final evaluation after configuration and checkpoint selection are frozen. Testing must not tune preprocessing.

On a compatible GPU build, repeat the tests and bounded verifier with `--device cuda`. The verifier makes at most three total optimizer steps and checks AMP, checkpoint restoration, and dual validation. The CUDA continuation test runs only when CUDA is available. No Khipu resources, modules, paths, or partitions are assumed. No complete epoch or testing evaluation has run. The final notebook should only load saved results.
