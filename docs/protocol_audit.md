# CIFP protocol audit

Audit date: 2026-08-05 (UTC). All external roots were inspected read-only with bounded-depth
`find`; no external file was changed.

## Runtime

- Host: Ubuntu 24.04, Linux 6.8.
- uv: 0.11.18.
- System Python: 3.12.3; uv-managed project target: Python 3.11.
- GPUs: 6 × NVIDIA GeForce RTX 5090, 32,607 MiB each.
- Driver: 580.142; driver CUDA capability: 13.0; local toolkit: 12.8.
- Audit-time availability: GPUs 0–3 were saturated, GPU 5 was partly occupied, GPU 4 was free.
- Selected wheel family: PyTorch 2.11.0/cu130 and torchvision 0.26.0/cu130. This exact stack and
  Transformers 5.14.0 are installed in the independent SemTrace environment on this host.

The cu130 index is configured explicitly rather than inferred at runtime. This follows uv's
documented PyTorch source-index pattern:
https://docs.astral.sh/uv/guides/integration/pytorch/

## DINOv3 model

Resolved local directory:
`/home/wuyanzhang/.cache/modelscope/models/facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/master`

The directory contains `model.safetensors`, `config.json`, `preprocessor_config.json`, the model
card, and the DINOv3 license. Audited config values are hidden size 384, 12 blocks, patch size 16,
and 4 register tokens. Preprocessing metadata contains ImageNet mean/std. Code must still read
these values dynamically. Hugging Face documents the last hidden state ordering as one CLS token,
then `config.num_register_tokens`, then patch tokens:
https://huggingface.co/docs/transformers/model_doc/dinov3#notes

## Protocol A: ForenSynths to Self-Synthesis

### Training and validation

Root: `/data/zhy/CNNDetection/dataset`

Observed layout is `<split>/<semantic_class>/<label_dir>/<image>` with official `train` and `val`
splits. The requested names exist exactly, so aliases are identity mappings:

| Canonical | Observed | Train real/fake | Val real/fake |
|---|---|---:|---:|
| car | car | 18,003 / 18,003 | 200 / 200 |
| cat | cat | 18,003 / 18,003 | 200 / 200 |
| chair | chair | 18,003 / 18,003 | 200 / 200 |
| horse | horse | 18,003 / 18,003 | 200 / 200 |

Label aliases are `0_real -> 0` and `1_fake -> 1`. ForenSynths is treated as source/generator
`ProGAN`; real source is LSUN, following the reference protocol. The manifest builder accepts
configured class aliases such as `cars -> car`, but none are required by this installation.

### Self-Synthesis test

Root: `/data/zhy/GANGen-Detection`

Observed layout is `<generator>/{0_real,1_fake}/<image>`. Every requested generator exists with
exact spelling and has 2,000 real plus 2,000 fake files:

`AttGAN`, `BEGAN`, `CramerGAN`, `InfoMaxGAN`, `MMDGAN`, `RelGAN`, `S3GAN`, `SNGAN`, `STGAN`.

All source aliases are identity mappings. No symbolic links were found.

### Optional UFD

`/data/zhy/CNNDetection/dataset/test` contains a download script and
`CNN_synth_testset.zip`, but no extracted Guided/LDM/GLIDE/DALL-E directory tree. The observed
185 MiB file has no readable ZIP central directory and therefore appears incomplete. The optional
manifest command checks the expected sources and reports the missing extraction explicitly; it
does not unpack, repair, download, or block either main protocol.

## Protocol B: GenImage SDv1.4 cross-generator

Root: `/data/zhy/GenImage`

Observed layout is `<source>/<split>/{nature,ai}/<image>`. `nature -> real (0)` and
`ai -> fake (1)`. Training uses only `stable_diffusion_v_1_4/train`; its counts are 162,000 real
and 162,000 fake. Its official validation directories contain 6,000 real and 6,000 fake.

Explicit source aliases:

| Canonical | Observed | Validation real/fake |
|---|---|---:|
| Midjourney | Midjourney | 6,000 / 6,000 |
| SDv1.4 | stable_diffusion_v_1_4 | 6,000 / 6,000 |
| SDv1.5 | stable_diffusion_v_1_5 | 8,000 / 8,000 |
| ADM | ADM | 6,000 / 6,000 |
| GLIDE | glide | 6,000 / 6,000 |
| Wukong | wukong | 6,000 / 6,000 |
| VQDM | VQDM | 6,000 / 6,000 |
| BigGAN | BigGAN | 6,000 / 6,000 |

No symbolic links were found. Semantic class is unavailable from this directory depth and remains
an explicit empty string rather than being guessed. Real source is ImageNet.

The generated protocol-B validation manifest contains the official 12,000-row SDv1.4 val split.
The eight-source test manifest necessarily includes those same SDv1.4 val rows as its SDv1.4
per-source entry. This creates a documented validation/test overlap of 12,000 paths, while the
324,000-row SDv1.4 train manifest has zero overlap with either. Validation is used only for
checkpoint selection and test metrics remain fixed-threshold reports.

## Reference protocol alignment

The AAAI 2026 paper specifies ProGAN training on car/cat/chair/horse, SDv1.4-only GenImage
training, random 128 crop for training, center 128 crop for testing, Adam with learning rate
`2e-4`, betas `(0.9, 0.999)`, weight decay `2e-4`, 200 epochs, batch size 128, and a fixed 0.5
threshold. Source: https://ojs.aaai.org/index.php/AAAI/article/view/40927

CIFP follows those data/optimization/evaluation settings but deliberately does not implement the
paper's pixel mapping or patch shuffle.

## SemTrace independence

`/data/zhy/SemTrace` is a separate uv/Python 3.11 project. Read-only inspection was limited to
dependency/config layout and generic data, DDP, checkpoint, metric, and logging locations. CIFP
will not import SemTrace, copy its model paths, or use its normal-prediction/residual design. No
runtime path in CIFP will reference `/data/zhy/SemTrace`.

## Recorded assumptions and unresolved audit work

- Fixed random-environment ablation labels are generated once per sample with seed 42.
- External data are never filtered or deleted by audit tools.
- `docs/dataset_audit.md` contains complete path, extension, group, symlink, and hard-link counts,
  plus a deterministic 1,000-header check per root for dimensions and corruption. The CLI supports
  a full header pass when `--max-image-checks` is omitted; that pass was attempted but stopped
  because the three roots contain 3,445,286 images. This limitation is explicit in the report as
  `complete=False` rather than presenting sampled health counts as exhaustive.
- GPU smoke tests must not displace existing workloads; unavailable DDP resources are reported as
  not verified.
