# UCLA ECE C147A Final Project — research overlay (v0.1)

**Source target:** [`Calvin-Pang/emg2qwerty`](https://github.com/Calvin-Pang/emg2qwerty), reviewed at commit `376923bcb4892d8cddae0ca7cd3d472ae11d93aa`. This is an **overlay**, not a fork, automatic upstream modification, or completed experiment. All official files remain unchanged; model, transforms, training entrypoint, configs and preflight are new.

## Install

1. Clone the original repository yourself and install its supported dependencies and download the authorized course data following the project instructions. The dataset and PDF are **not** included here.
2. Extract `ece147_patchkit.zip` into the clone root so that `emg2qwerty/research_lightning.py` appears alongside original `emg2qwerty/lightning.py`.
3. Run `python scripts/preflight_check.py`. Smoke tests use synthetic data only.
4. Run `pytest tests/test_research_smoke.py -q` if pytest is installed.

Use the repository's Python environment, because its dependencies include `hydra-core`, `pytorch-lightning`, `torchaudio`, `python-Levenshtein`, etc. The smoke tests do not establish that all upstream dependencies load successfully.

## Training commands (from clone root)

```bash
# A0 official reference model, original train.py. NOTE: official entrypoint runs test after fitting!
# python -m emg2qwerty.train

# A1 architecture study: reuse the ORIGINAL augmentations for a meaningful TDS comparison
python -m emg2qwerty.research_train model=research_bigru seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0

# A2 architecture study, same front-end and transforms
python -m emg2qwerty.research_train model=research_bilstm seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0

# C0: matched no-rotation/no-jitter baseline for channel ablations
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=full32 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0

# C1: single left band / C2: single right band
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=left16 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=right16 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0

# C3: 8 electrodes per band; C4: 4 electrodes per band
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=balanced16 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=balanced8 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0

# E1: true anti-aliased downsampling; window duration / hop fixed in physical time
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=fs1000 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=fs500 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=fs250 seed=42 trainer.max_epochs=40 trainer.gradient_clip_val=1.0

# Only after hyperparameters and experiments have been frozen: assess held-out test ONCE
python -m emg2qwerty.research_train model=research_bigru transforms=research_ablation +experiment=full32 train=false checkpoint=/path/to/best.ckpt +final_test=true
```

When using the official `train.py`, repeated test evaluation is easy to miss. The **new research_train.py** evaluates validation only; it calls test only with `+final_test=true`.

Run seeds 42, 123, and 2026 for your *selected* main configurations, recording mean ± SD without tuning on the test set. Report both available and failed configurations, and measure GPU wall time and latency separately on common hardware.

## Important scientific limitations

- **CTC** requires `output_frames >= target_chars + adjacent_repeated_pairs`; the default policy is fail-fast, not silently discarding data. You can opt in to `invalid_ctc_policy=zero`, but every invalid sample is logged and must be reported. There is no arbitrary 0.1% acceptable failure threshold.
- **Channel ablations** slice raw channels BEFORE STFT. Physical electrode ordering and equal-spacing assumptions should be verified with the original hardware metadata. `balanced16` denotes 16 **total** electrodes (8 per band); `balanced8` is 8 total (4 per band). All these ablation experiments disable random rotations and temporal jitter, including C0; the official A0 differs from C0.
- **Frequency ablations** keep physical STFT window 32 ms, frame hop 8 ms, and bin spacing 31.25 Hz constant; the accessible *bandwidth* and frequency-bin count change, not frequency resolution. The first MLP's input dimensions and parameter counts also change, so this is **not** an isolated causal test of bandwidth. To isolate high-frequency information, add a matched-dimension low-pass-only control at 2 kHz (future experiment).
- **Physical timing:** the spectral frame length equality test assumes windows with lengths divisible by decimation factors. Real boundary windows, jitter, filter transients, and 1800/200 raw-sample context padding require separate timestamp-alignment validation.
- **Padding:** packing prevents RNN padded frames from contributing to RNN state, but upstream `SpectrogramNorm` still computes cross-time batch statistics including padded values. Treat results as provisional until batch padding effects have been quantified or a matched masked normalization frontend is validated.
- **BiGRU:** even perfect packing does not remove unconstrained future context. The original repo's entire-session test is an **offline** metric. Do not claim online latency or real-time usability without a correctly implemented fixed-lookahead/streaming protocol; no chunked decoder is included in this patch.
- **Error analysis:** the `scripts/error_analysis.py` breakdown is a deterministic, reference-anchored diagnostic. Insertions are counted by *predicted* side separately. `B` conventionally belongs to the left hand; space and punctuation are classified as 'other'. Multi-optimal alignments make fine-grained attribution non-unique.
- **Generalization:** the original single-user split supports within-user conclusions only. Different seed runs on one validation session are not independent evidence for cross-user generalization.
- **Status:** No actual HDF5 dataset, course handout or model training results were available here. Synthetic CPU smoke tests are not proof of complete training or reproduction of 30% CER.

## Deliverables to capture for each run

- Frozen resolved Hydra configuration, seed, Git revision and dependency versions.
- Validation CER and S/D/I, trained model checkpoint, training curves, parameter counts, peak GPU memory and throughput.
- Any invalid CTC count and failures; verified sample times and frame counts.
- Final held-out test run *only once*, after selecting the locked experimental protocol.
- Full disclosures required by the course, including truthful AI-assisted code generation/review.

## Test-set isolation detail

Research model YAMLs use `ResearchEMGDataModule`, a stage-scoped subclass of the original data module. Its `setup("fit")` and `setup("validate")` construct only train/validation HDF5 datasets; test HDF5 is opened only for `setup("test")`. It also checks for duplicate session stems across the three splits. This does not prevent an investigator from accidentally tuning on a final test result; do not evaluate test until the plan is locked.
