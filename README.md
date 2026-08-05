# CIFP

Content-Invariant Compositional Forensic Primitive Learning（内容不变的组合式取证基元学习）
是一个用于生成图像跨生成器检测的可训练、可评测研究代码库。模型把 DINOv3 patch token
映射到无标签局部取证基元的稀疏激活，再仅以图像级 mean/max 使用统计（以及可选局部共现）
分类真假。冻结语义教师只离线构造内容伪环境，推理时完全不存在。

CIFP 不是残差方法：它不建立真实图像正常参照，不预测或重建图像/特征，不计算输入减预测、
token 减原型、距真实中心或 anomaly score。Transformer 自身的标准残差连接不被重新解释为
取证表示。

## 项目结构

- `configs/`：模型、协议和消融 YAML。
- `src/cifp/data/`：只读 manifest 数据集、协议变换、审计和 DDP batch sampler。
- `src/cifp/environments/`：离线冻结教师、可续跑 memmap 和 MiniBatchKMeans。
- `src/cifp/models/`、`losses/`：学生、基元分配、组合池化、GRL 和三项损失。
- `src/cifp/engine/`、`metrics/`：原生 DDP 训练、checkpoint、聚合评测和结果表。
- `src/cifp/analysis/`、`src/cifp/cli/`：机制分析和所有 CLI。
- `docs/`：协议、数据、方法、实施与架构审计。
- `artifacts/`、`outputs/`：项目内生成物；外部数据根始终只读。

## 从空环境到 smoke test

项目要求 Python 3.11，且只用 uv。当前锁文件选择 Linux x86-64 的 PyTorch 2.11/cu130 wheel。

```bash
uv sync --locked
export CIFP_DINOV3_PATH=/path/to/facebook--dinov3-vits16-pretrain-lvd1689m/snapshot
uv run python -m cifp.cli.inspect_data --max-image-checks 1000
uv run python -m cifp.cli.build_manifests --protocol forensynths_selfsynthesis
uv run python -m cifp.cli.build_manifests --protocol genimage_sd14
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python -m cifp.cli.train \
  --config configs/protocol/forensynths_selfsynthesis.yaml \
  --synthetic --max-steps 1 --precision fp32 --workers 0 --device cpu \
  --output outputs/synthetic_smoke
```

`CIFP_DINOV3_PATH` 必须指向包含 `config.json` 和 `model.safetensors` 的指定 DINOv3
ViT-S/16 快照。本机审计到的快照位于 ModelScope cache，但配置不硬编码用户名或 cache 路径。
若没有本地权重，请先在模型页面接受授权，再执行 `uv run hf auth login` 并把
`local_files_only` 显式改为 `false`。加载失败会报错；绝不会回退到 DINOv2、其他模型或随机权重。

## 数据与审计

默认只读根：

- ForenSynths：`/data/zhy/CNNDetection/dataset`
- GenImage：`/data/zhy/GenImage`
- Self-Synthesis：`/data/zhy/GANGen-Detection`
- SemTrace（仅审阅通用工程组织）：`/data/zhy/SemTrace`

完整审计会读取每张图像的 header，可能较慢；限制版只抽查图像，但仍统计全部路径与扩展名。

```bash
uv run python -m cifp.cli.inspect_data --output docs/dataset_audit.md
uv run python -m cifp.cli.inspect_data --max-image-checks 1000 \
  --output outputs/bounded_dataset_audit.md
uv run python -m cifp.cli.build_manifests --protocol forensynths_selfsynthesis
uv run python -m cifp.cli.build_manifests --protocol genimage_sd14
uv run python -m cifp.cli.build_manifests --protocol optional_ufd
```

Dataset 只读 parquet manifest，不在训练时重扫目录。`real=0`、`fake=1`；损坏或小于
128 的图像默认携完整路径报错，不静默跳过/resize。官方协议使用 RandomCrop/CenterCrop(128)、
DINOv3 mean/std，不启用 resize、pixel mapping、patch shuffle、颜色增强或水平翻转。

## 离线内容环境

两个协议必须分别运行，禁止合并聚类。语义特征为归一化 CLS 与归一化 mean patch 的拼接；
register token 被排除。float16 memmap 用完成位支持断点续跑，路径行号由 parquet index 固定。

```bash
uv run python -m cifp.cli.extract_semantics \
  --config configs/protocol/forensynths_selfsynthesis.yaml --device cuda
uv run python -m cifp.cli.cluster_environments \
  --config configs/protocol/forensynths_selfsynthesis.yaml

uv run python -m cifp.cli.extract_semantics \
  --config configs/protocol/genimage_sd14.yaml --device cuda
uv run python -m cifp.cli.cluster_environments \
  --config configs/protocol/genimage_sd14.yaml
```

聚类默认 `C=100`、最多平衡抽样 200,000 行、`random_state=42`。输出包括 clusterer、
fit index、更新后的训练 manifest 和环境分布问题报告。`--random` 仅用于固定随机环境消融。

## 训练

论文对齐配置为 Adam、`lr=2e-4`、`weight_decay=2e-4`、200 epochs、bf16 和 global
batch 128。不会自动启动正式训练。4 GPU 正式命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cifp.cli.train --config configs/protocol/forensynths_selfsynthesis.yaml \
  --run-id forensynths_cifp_seed42

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cifp.cli.train --config configs/protocol/genimage_sd14.yaml \
  --run-id genimage_sd14_cifp_seed42
```

1 GPU debug（最多 20 steps）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m cifp.cli.train \
  --config configs/protocol/forensynths_selfsynthesis.yaml \
  --max-steps 20 --workers 2 --device cuda --output outputs/real_debug
```

6 GPU 在默认 per-GPU batch 32 下实际 global batch 为 192，运行元数据会标记
`non_protocol_batch=true`，不可作为 batch-128 对齐实验。恢复完整状态：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m cifp.cli.train --config configs/protocol/forensynths_selfsynthesis.yaml \
  --resume outputs/forensynths_cifp_seed42/checkpoints/last.pt \
  --output outputs/forensynths_cifp_seed42
```

每个 run 保存解析配置、Git/uv/GPU 运行元数据、JSONL/CSV/TensorBoard 日志、`last.pt`、
`best_validation_ap.pt` 和 `best_validation_accuracy.pt`。bf16 不可用时必须显式传
`--precision fp32`；不会静默切 fp16。

## 评测

固定 `sigmoid(fake_logit)` 与 threshold 0.5；测试集不重新选阈值。结果含逐样本 CSV、
per-source Acc/AP/AUROC/FPR/Recall/Precision/confusion matrix、macro mAcc/mAP、overall、
worst source，以及 Markdown/LaTeX 表。

Self-Synthesis：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m cifp.cli.evaluate \
  --config configs/protocol/forensynths_selfsynthesis.yaml \
  --checkpoint outputs/forensynths_cifp_seed42/checkpoints/best_validation_ap.pt \
  --output outputs/forensynths_cifp_seed42/evaluation/self_synthesis
```

GenImage 八源：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m cifp.cli.evaluate \
  --config configs/protocol/genimage_sd14.yaml \
  --checkpoint outputs/genimage_sd14_cifp_seed42/checkpoints/best_validation_ap.pt \
  --output outputs/genimage_sd14_cifp_seed42/evaluation/genimage_cross_generator
```

DDP 评测可把 `python` 换为 `torchrun --standalone --nproc_per_node=N`；sampler 不填充，
rank 聚合后会检查重复和漏样本。

## 消融与基线

`configs/ablation/` 包含 patch mean+MLP、CLS+MLP、随机固定字典、K=1、dense assignment、
无组合正则、无内容对抗、固定随机环境、全冻结、最后两 block、mean/max 和共现配置。
它们继承主协议，命令只需替换 `--config`。`random_content_env.yaml` 会在内存中用 seed 42
按路径稳定地替换环境且不覆写主 manifest；`--random-content-env` 是等价的短 smoke 开关。

## 基元机制分析

所有结果应放在 `outputs/<run_id>/analysis/`：

```bash
RUN=outputs/forensynths_cifp_seed42
uv run python -m cifp.cli.extract_analysis_features \
  --config configs/protocol/forensynths_selfsynthesis.yaml \
  --checkpoint "$RUN/checkpoints/best_validation_ap.pt" \
  --manifest artifacts/manifests/forensynths_selfsynthesis/test.parquet \
  --output "$RUN/analysis/test_features.npz"
uv run python -m cifp.cli.primitive_usage \
  --features "$RUN/analysis/test_features.npz" --output "$RUN/analysis/usage.json"
uv run python -m cifp.cli.primitive_generator_mi \
  --features "$RUN/analysis/test_features.npz" --output "$RUN/analysis/generator_mi.json"
uv run python -m cifp.cli.primitive_content_mi \
  --features "$RUN/analysis/test_features.npz" --output "$RUN/analysis/content_mi.json"
uv run python -m cifp.cli.primitive_masking \
  --config configs/protocol/forensynths_selfsynthesis.yaml \
  --checkpoint "$RUN/checkpoints/best_validation_ap.pt" \
  --features "$RUN/analysis/test_features.npz" --output "$RUN/analysis/masking.json"
uv run python -m cifp.cli.primitive_coverage \
  --train-features "$RUN/analysis/train_features.npz" \
  --unknown-features "$RUN/analysis/test_features.npz" \
  --output "$RUN/analysis/coverage.json"
uv run python -m cifp.cli.visualize_activations \
  --features "$RUN/analysis/test_features.npz" \
  --output-dir "$RUN/analysis/activation_maps"
uv run python -m cifp.cli.linear_probe \
  --features "$RUN/analysis/test_features.npz" --output "$RUN/analysis/linear_probes.json"
```

热图仅表示模型激活，不自动等同于人类可解释的取证痕迹。

## 输出结构

```text
artifacts/manifests/<protocol>/
  train.parquet  validation.parquet  test.parquet
  semantic_features/  clusterer.pkl  cluster_fit_indices.npy
  environment_config.json  environment_report.json
outputs/<run_id>/
  resolved_config.yaml  run_metadata.json  metrics.{jsonl,csv}  tensorboard/
  checkpoints/  validation/  evaluation/  analysis/
```

## 常见问题

- `content_env=-1`：先完成语义提取和聚类；不会把未分配环境当成有效类别。
- 小图或坏图：查看完整异常路径和 `docs/dataset_audit.md`，由配置决定修复策略；不删除样本。
- CUDA/cuDNN 动态库冲突：若宿主 `LD_LIBRARY_PATH` 注入了与 uv wheel 不同的 cuDNN，
  使用干净 shell 或对单个命令执行 `env -u LD_LIBRARY_PATH ...`；不要改 wheel 来掩盖问题。
- 如何证明推理不需教师：`CIFP` 位于 `cifp.models`，不含 teacher 成员；
  `model.inference(images)` 只返回 fake logits 且不调用环境头。可运行
  `uv run pytest -q tests/test_frozen_teacher.py tests/test_architecture.py`，也可搜索 checkpoint
  key，只有 forensic student、基元、组合器、真假头和训练期环境头，没有语义教师参数。

详细工程假设、实际目录别名与结构边界见 `docs/protocol_audit.md`、
`docs/method_spec.md` 和 `docs/architecture_audit.md`。
