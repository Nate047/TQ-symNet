# TQ-SymNet

This is the standalone MMDetection 2.x implementation for TQ-SymNet.
The main experiment uses the original 599/200 TB-positive detection split,
a frozen TransNeXt/CrossScale/Sym detector and a trained quality evaluator
with a learned fusion coefficient. See [PAPER_REPRODUCIBILITY.md](PAPER_REPRODUCIBILITY.md)
for required assets, one-GPU commands, evaluation and known reproduction limits.

Data, checkpoints, prediction outputs and compiled CUDA binaries are not bundled.
The historical detector checkpoint is required by the publication quality-stage
configuration and is not currently included; hashes are enforced. The independent
image-level classifier and its export instructions are documented in
[RUN_CLASSIFICATION_ORIGINAL_SYM.md](RUN_CLASSIFICATION_ORIGINAL_SYM.md).
