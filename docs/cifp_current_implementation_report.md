# CIFP 当前真实实现方法与实验报告

> 审计基准：当前仓库 `HEAD=0d090fed3b0065e7936a9fd33f6ed2537d684cf8`，指定运行目录
> `outputs/20260807T104523`。本报告以实际代码为最高优先级，并以该运行保存的
> `resolved_config.yaml`、checkpoint 内嵌配置、训练日志、指定 evaluation 和 analysis 产物
> 交叉核验。当前工作区中的 `configs/protocol/genimage_sd14.yaml` 有未提交修改；指定运行的
> 事实均取自运行目录保存的解析配置，而不是用当前文件覆盖。

## 1. 项目概况

### 1.1 项目目标

CIFP（Content-Invariant Compositional Forensic Primitive Learning）面向生成图像检测中的
跨生成器泛化问题。其工作假设不是“每个生成器都对应一个完全独立的整体指纹”，而是不同
生成器可能共享一组局部取证模式，并通过不同的激活强度、空间位置和组合方式形成图像级判别
证据。模型因此学习无人工语义的局部取证基元字典，将每个图像 Patch 表示为少量基元的软激活，
再仅从这些激活的组合统计构造真假分类表示。

内容语义可能成为检测捷径：若训练集中的真实/生成标签与物体类别、场景或来源分布相关，模型
可能依靠语义而不是生成过程相关证据。当前项目通过训练前离线构造内容伪环境，并在训练时用
梯度反转环境分类器约束最终取证表示，尝试降低这种内容环境可预测性。现有实现和实验只能支持
“采用了该约束并观察到环境预测接近低准确率”的事实，不能据此宣称已经完全去除语义。

### 1.2 当前实现状态

当前仓库已经具备：src-layout Python 包、uv 锁定环境、两套主数据协议的 Parquet manifest、
严格图像读取和预处理、离线 DINOv3 内容环境构造、CIFP 主模型、原生 PyTorch DDP、bf16/FP32
训练、checkpoint 恢复、逐来源评测、论文表格导出、基元使用/互信息/遮蔽/覆盖/热图/线性探针
分析代码，以及69项当前通过的单元与集成测试。`scripts/` 目录当前不存在；命令行入口位于
`src/cifp/cli/`，`tools/` 提供对应包装脚本。

指定 GenImage 运行已完成200 epochs训练、三个checkpoint保存、完整10万张八来源评测及主要
机制分析。该运行使用单张 NVIDIA GeForce RTX 5090、bf16 和 batch 4096，属于项目显式记录的
`reflect_pad_filtered_non_protocol` 变体，不等同于项目文档所述 global batch 128 对齐配置。

【代码依据】`src/cifp/`、`tests/`、`pyproject.toml`、`uv.lock`
【运行依据】`outputs/20260807T104523/resolved_config.yaml`、`run_metadata.json`

### 1.3 核心方法概述

当前主模型的数据流为：DINOv3 ViT-S/16 最后一层真实 Patch Token，经局部 MLP 投影并 L2
归一化后，与端到端学习的归一化字典计算温度缩放余弦相似度；每个 Patch 仅保留 top-4 logits，
经 Softmax 得到稀疏激活（`top_k=4`）。模型对全部 Patch 的激活计算逐基元均值和最大值，
拼接成64维统计量，
经组合投影器形成128维 `z_for`，真假分类器只接收 `z_for`。指定最佳实验没有启用空间共现扩展。

## 2. CIFP 整体方法

### 2.1 核心科学假设

1. 局部生成痕迹可以由有限数量、可跨图像复用的潜在基元描述。
2. 未见生成器不一定要求全新的基元；它可能复用训练期形成的基元，并改变强度和组合。
3. 图像级真假判别可建立在局部基元激活的组合统计上，而无需构造真实图像的正常参照。
4. 内容语义对真假标签的偶然相关会损害跨域泛化，因此应把内容环境视为训练期干扰因素。

这些是假设和工程化建模选择。当前实验支持模型确实使用多个基元、不同生成器产生不同且部分
相近的使用分布；尚无已执行的 train-vs-unknown coverage 结果直接检验“新组合”比例。

### 2.2 总体网络结构

对于输入 `x∈R^{B×3×128×128}`，实际 forward 为：

```text
x
→ DINOv3 forensic student
→ 最后一层 true Patch Tokens F ∈ R[B,64,384]
→ LocalForensicProjector
→ 归一化局部特征 H ∈ R[B,64,256]
→ SparsePrimitiveDictionary（D ∈ R[32,256]）
→ top-4 激活 A ∈ R[B,64,32]
→ concat(mean_patch(A), max_patch(A)) ∈ R[B,64]
→ CompositionPooler MLP
→ z_for ∈ R[B,128]
→ FakeClassifier(Linear(128,1))
→ fake_logit ∈ R[B]
```

训练时还有独立支路：

```text
z_for → Gradient Reversal → EnvironmentClassifier → logits_env ∈ R[B,100]
```

环境支路不向真假分类器提供输入；它只通过反向梯度约束 `z_for`。

### 2.3 训练数据流

训练数据由项目内 manifest 读取。指定运行中，每张图像经 RGB 转换、必要时反射填充、随机
128×128裁剪和 DINOv3 mean/std 归一化后进入学生网络。训练 batch 同时提供 `label` 和已离线
分配的 `content_env`：`label` 用于 BCE 真假损失，`content_env` 仅用于环境交叉熵。

学生最后两个 Transformer block（其中 LayerNorm 仍冻结）、局部投影器、字典、组合投影器、
真假头和环境头共同参与训练。更早的学生参数冻结；离线语义教师完全不在训练图中。

### 2.4 推理数据流

推理调用 `CIFP.inference(images)`：

```text
图像 → 学生 Patch Token → 局部投影 → 字典稀疏激活
     → mean/max组合 → z_for → fake logit → sigmoid → fake probability
```

推理不需要语义教师、语义 embedding、聚类器、`content_env`、生成器标签、来源标签、多视图或
test-time optimization。环境分类器在 `grl_lambda=None` 时不执行。真假分类器的唯一直接输入是
`z_for`。

【代码依据】`src/cifp/models/cifp.py`、`src/cifp/models/composition.py`

## 3. 取证特征编码

### 3.1 视觉基础模型

主模型和离线教师均加载明确指定的
`facebook/dinov3-vits16-pretrain-lvd1689m`，不允许回退到 DINOv2、其他变体或随机权重。
本机实际快照配置为：

| 项目 | 实际值 |
|---|---:|
| 模型类型 | `dinov3_vit` |
| hidden size | 384 |
| Transformer blocks | 12 |
| attention heads | 6 |
| patch size | 16 |
| register tokens | 4 |
| 预训练配置 nominal image size | 224 |
| CIFP实际输入 | 128×128 |

模型结构参数从所加载 config 动态读取，Token 切片没有硬编码384、4或64。

【代码依据】`src/cifp/models/backbone.py`
【模型依据】本地 DINOv3 `config.json`、`preprocessor_config.json`

### 3.2 Patch Token 提取

DINOv3 `last_hidden_state` 被按如下顺序校验和切分：

```text
[CLS] [4 register tokens] [true patch tokens]
```

128可被patch size 16整除，因此运行时网格为8×8，共64个真实 Patch Token。CLS 被返回给基线
接口但在 CIFP 主 forward 中以 `_cls_token` 丢弃；4个 register token 也不会进入投影器或分类器。
当前模型只使用最后一层 `last_hidden_state`，没有多层拼接。

### 3.3 局部取证特征投影

局部投影器为：

```text
LayerNorm(384)
→ Linear(384,256)
→ GELU
→ Linear(256,256)
→ L2 normalize
```

无额外 dropout。输出 `H∈R^{B×64×256}` 是为字典余弦匹配学习的局部表征。代码没有把它
直接定义为纯生成痕迹，也没有用其与任何正常特征做差。

### 3.4 参数冻结与微调策略

指定运行设置 `train_last_n_blocks=2`、`train_norm=false`。实现先冻结整个学生，再开放最后两个
block；随后遍历所有 LayerNorm 并按 `train_norm=false` 再次冻结，因此 block 10和11中的
LayerNorm参数也不训练。Patch embedding、CLS/register参数和前10个block冻结。

运行日志记录：

| 模块 | 总参数 | 可训练参数 |
|---|---:|---:|
| 全模型 | 21,824,357 | 3,774,437 |
| DINOv3学生 | 21,596,544 | 3,546,624 |
| 局部投影器 | 165,120 | 165,120 |
| 基元字典 | 8,192 | 8,192 |
| 组合池化器 | 24,960 | 24,960 |
| 真假分类器 | 129 | 129 |
| 环境分类器 | 29,412 | 29,412 |

## 4. 组合式取证基元学习

### 4.1 取证基元定义

“取证基元”在当前代码中是256维可学习方向。它没有人工痕迹标签，不预先命名为纹理、边缘、
高频或压缩伪影，也不绑定某个生成器、真假类别或真实图像中心。其语义由端到端检测、组合正则
和内容环境对抗共同塑造，因此机制分析只能讨论激活分布和预测关联，不能直接赋予人类物理含义。

### 4.2 可学习基元字典

指定运行字典为：

\[
D\in\mathbb{R}^{K\times d},\quad K=32,\ d=256.
\]

原始参数用均值0、标准差 `d^{-1/2}=1/16` 的正态分布初始化；每次 forward 计算
`L2-normalize(D, dim=-1)`。该参数在主实验中 `requires_grad=True`，由总损失端到端更新。

最佳AP checkpoint 的归一化字典审计结果为：off-diagonal cosine平方均值0.01244、绝对值均值
0.08415，最大正余弦0.48899、最小余弦−0.46326，没有绝对余弦≥0.9的基元对；参与率定义的
有效秩约23.09。该结果不显示字典方向整体坍缩为相同向量，但也不表示32个方向完全正交。

### 4.3 稀疏基元分配

局部特征和字典均已归一化，因此矩阵乘法等价于余弦相似度。对第 `i` 个Patch与第 `k` 个
基元：

\[
\ell_{ik}=\frac{h_i^\top \hat d_k}{\tau},\qquad \tau=0.1.
\]

每行取最大的4个logit，其他位置置为 `-∞`，再执行 Softmax：

\[
a_{ik}=\operatorname{softmax}_k(\operatorname{TopKMask}(\ell_i)),\qquad top\_k=4.
\]

所以 `a_ik≥0`、每行和为1，且每个Patch最多4个非零项。一个Patch可同时激活多个基元；同一
基元可被不同Patch、图像和生成器复用。当前项目没有 Entmax 后端；`dense` Softmax 仅作为显式
消融配置存在。

### 4.4 基元激活矩阵

128输入下，激活矩阵为 `A∈R^{B×64×32}`。它不是 Token 与字典重建值的差，也不产生
reconstructed token。训练输出保留 `A` 以计算组合正则和机制统计；推理只向外返回真假logit。

### 4.5 图像级组合式取证表示

当前主实验实际使用：

\[
p_{mean}=\frac{1}{N}\sum_{i=1}^{N}a_i,\qquad
p_{max,k}=\max_i a_{ik},
\]

\[
p=[p_{mean};p_{max}]\in\mathbb{R}^{64}.
\]

组合投影器为：

```text
LayerNorm(64)
→ Linear(64,128)
→ GELU
→ Dropout(0.1)
→ Linear(128,128)
```

得到 `z_for∈R^{B×128}`。最终头是 `Linear(128,1)`，没有把 CLS、Patch Token、语义 embedding、
环境标签或生成器标签拼接给分类器。

### 4.6 基元共现机制（已实现扩展，主实验未启用）

代码已经实现动态网格的水平/垂直邻接共现。每条无向边 `(i,j)` 计算：

\[
R_{ij}=\frac{1}{2}(a_i a_j^\top+a_j a_i^\top),
\]

对全部边平均得到 `R∈R^{B×32×32}`，展平后经 `Linear(1024,128)→GELU→Dropout`，再与
mean/max统计拼接。指定运行配置和checkpoint都显示 `cooccurrence.enabled=false`，checkpoint
没有共现投影器参数，因此本报告不把它写成当前主方法结果。

## 5. 内容语义去偏机制

### 5.1 内容语义教师

离线教师与学生使用同一指定 DINOv3 ViT-S/16，但对象完全独立。教师构造时冻结全部参数、固定
eval模式，并在 `torch.inference_mode()` 下提取：

\[
s=\left[\operatorname{L2}(CLS);\operatorname{L2}\left(\frac1N\sum_i Patch_i\right)\right]
\in\mathbb{R}^{768}.
\]

mean Patch 排除了CLS和4个register token。教师只存在于 `cifp.cli.extract_semantics`，不属于
`CIFP` 模型。最佳checkpoint的230个模型键中没有 teacher/semantic 参数。

### 5.2 内容伪环境构造

指定 GenImage 运行先审计原324,000行SDv1.4训练manifest，排除3个空文件；3,858张小图保留，
由 `reflect_pad` 处理，得到323,997行训练manifest。离线提取完成323,997个float16语义向量，
路径索引与memmap行号固定对应。

随后按 `(label, semantic_class, source)` 分层轮转抽样最多200,000行，以
`MiniBatchKMeans(n_clusters=100, batch_size=4096, random_state=42, n_init="auto")` 聚类，并为
全部训练样本赋予 `content_env∈{0,…,99}`。GenImage的 `semantic_class` 为空且训练source只有
SDv1.4，因此该分层在此运行中主要体现真假平衡。实际100个环境均非空、均含两类标签；环境大小
1,807至5,164，报告标记环境5、23、62为极端标签不平衡。

【代码依据】`src/cifp/environments/teacher.py`、`clustering.py`
【产物依据】`artifacts/manifests/genimage_sd14_filtered/environment_config.json`、
`environment_report.json`

### 5.3 梯度反转与环境对抗

训练时：

```text
z_for
→ GRL
→ Linear(128,128)
→ GELU
→ Dropout(0.1)
→ Linear(128,100)
→ content_env logits
```

GRL forward恒等；反向传给 `z_for` 的梯度乘 `-λ_grl`，环境分类器自身仍沿最小化交叉熵方向
更新。代码按 fractional epoch `e` 使用：

\[
\lambda_{grl}(e)=
\begin{cases}
0,&e\le5\\
(e-5)/15,&5<e<20\\
1,&e\ge20.
\end{cases}
\]

由于 `e=epoch+batch_index/len(loader)`，索引0至4的epoch完全为0；索引5（第6个epoch）从0开始
在epoch内部逐渐增大，索引20起固定为1。即使GRL为0，环境头和 `L_nui` 仍计算，环境头自身仍
学习；只是该损失当时不给 `z_for` 反向信号。

最佳AP epoch 151的训练日志中，环境准确率按batch平均约2.20%，环境CE约4.538；100类均匀
随机准确率为1%、CE为 `ln(100)≈4.605`。这说明当时环境预测较弱，但环境规模不均衡，且该值
来自训练batch，不能单独证明内容信息已被完全消除。

### 5.4 训练与推理阶段差异

| 模块/数据 | 训练 | 推理 |
|---|---|---|
| DINOv3学生 | 是 | 是 |
| 局部投影、字典、组合器、真假头 | 是 | 是 |
| 离线语义教师 | 否，仅训练前工具 | 否 |
| 聚类器/语义memmap | 否，仅训练前准备 | 否 |
| `content_env` | 环境CE标签 | 不需要 |
| GRL与环境分类器 | 是 | 不执行 |
| generator/source标签 | 不进入模型 | 不需要 |

## 6. 损失函数

指定主实验的总损失为：

\[
L=L_{det}+\lambda_{comp}L_{comp}+\lambda_{nui}L_{nui},
\]

其中所有下列项在指定运行中均实际启用。

### 6.1 真假检测损失

\[
L_{det}=\operatorname{BCEWithLogits}(fake\_logit,y),\quad y_{real}=0,\ y_{fake}=1.
\]

`pos_weight=null`，因为训练sampler尽量构造真假各半batch。权重为1。

### 6.2 激活稀疏/尖锐性损失

\[
L_{sparse}=\frac{1}{BN}\sum_{b,i}-\sum_k a_{bik}\log(a_{bik}+10^{-8}).
\]

Top-k提供结构稀疏性，最小化熵使保留的4个激活进一步集中。内部权重 `w_sparse=0.1`。

### 6.3 基元使用均衡损失

令 `q_k=mean_{b,i}(a_bik)`：

\[
L_{balance}=\sum_k q_k\log(q_kK+10^{-8}).
\]

它等价于相对均匀分布的KL形式，抑制长期只使用少量基元。内部权重 `w_balance=1.0`。

### 6.4 字典多样性损失

对归一化字典 `G=DD^T`：

\[
L_{diversity}=mean_{i\ne j}(G_{ij}^2).
\]

它惩罚基元方向重合。内部权重 `w_diversity=0.1`。`lambda_comp=0.1`，所以三个正则对总损失
的有效系数分别为0.01、0.1和0.01。

### 6.5 内容环境对抗损失

\[
L_{nui}=CE(environment\_logits,content\_env),\qquad \lambda_{nui}=0.1.
\]

环境头沿普通CE方向训练；`z_for`方向再由GRL反转并按schedule缩放。

### 6.6 当前不存在的损失

代码中没有 reconstruction loss、像素/特征残差损失、正常中心距离、prototype distance
classification、异常分数损失、标签平滑、mixup或cutmix损失。

【代码依据】`src/cifp/losses/composition.py`、`src/cifp/losses/total.py`

## 7. 训练方法与工程实现

### 7.1 数据预处理

指定运行实际预处理为：

- PIL解码并转RGB；损坏图像报完整路径，不静默跳过。
- 训练：不足128时反射填充，再 RandomCrop(128)。
- 验证/测试：不足128时反射填充，再 CenterCrop(128)。
- 不Resize、不Pixel Mapping、不Patch Shuffle、不颜色增强。
- `horizontal_flip=false`。
- 像素缩放至 `[0,1]`，使用 mean `(0.485,0.456,0.406)`、std
  `(0.229,0.224,0.225)`。

这与仓库严格配置的 `small_image_policy=error` 不同；指定运行明确保存为
`reflect_pad_filtered_non_protocol`。

### 7.2 优化器与超参数

| 参数 | 指定运行实际值 |
|---|---|
| Optimizer | `torch.optim.Adam` |
| Learning rate | `2e-4`，恒定 |
| Betas | `(0.9,0.999)` |
| Weight decay | `2e-4` |
| Scheduler | 无 |
| Epochs | 200 |
| 每GPU batch | 4096 |
| GPU数 | 1 |
| 实际global batch | 4096 |
| Gradient accumulation | 1 |
| Precision | bf16 |
| Seed | 42 |
| Worker | 8 |
| 每epoch optimizer steps | 79 |
| 总optimizer steps | 15,800 |

训练sampler尽量在每个batch中平衡real/fake，并轮转覆盖多个 `content_env`。单GPU每epoch仅生成
79个完整batch，尾部不足一个batch的样本不进入该epoch。日志中 `sampler_fallbacks=0`。

训练从解析配置写入到 `last.pt` 的文件时间间隔约27小时27分钟；该时间包含每epoch对10万张
validation manifest的评测和结果写出，不是纯forward/backward计时。日志未记录峰值显存，不能
从当前产物补全。

### 7.3 分布式训练

项目实现原生 `torch.distributed` + NCCL + `DistributedDataParallel`。自定义训练sampler按rank
切分样本并支持 `set_epoch`；评测sampler不填充，因此不会为对齐rank数复制样本，聚合后还检查
重复和漏样本。指定运行metadata只记录一张RTX 5090，因此本次主实验实际为单GPU，不应写成
DDP训练。

### 7.4 checkpoint 与模型选择

checkpoint保存模型、Adam状态、epoch、global step、解析配置、运行metadata以及Python/NumPy/
PyTorch/CUDA随机状态。每epoch保存 `last.pt`；overall AP或固定0.5 overall Accuracy创新高时分别
保存 `best_validation_ap.pt` 和 `best_validation_accuracy.pt`。

指定运行将 `artifacts/manifests/genimage_sd14/test.parquet` 同时配置为 validation 和 test，
所以每个epoch都在同一10万张八来源集合上计算模型选择指标。结果为：

| checkpoint | 保存的epoch索引 | step | 当epoch overall Acc@0.5 | AP |
|---|---:|---:|---:|---:|
| best validation Accuracy | 78 | 6,241 | 81.756% | 95.023% |
| best validation AP | 151 | 12,008 | 80.622% | 95.182% |
| last | 199 | 15,800 | 76.517% | 94.521% |

指定 evaluation 和 analysis 的连续指标与epoch 151完全一致，表明它们对应best-AP参数或与其
数值等价的模型；evaluation产物本身没有另存checkpoint路径或哈希。

### 7.5 推理与评测实现

评测使用 `sigmoid(fake_logit)` 作为fake probability。默认阈值为配置的0.5；CLI额外提供
`--test-oracle-threshold`。启用时，它读取整个评测manifest的真实标签，在全部连续分数上选择
使overall Accuracy最高的一个全局阈值；并非为每个来源单独选阈值。AP和AUROC始终使用连续
分数。输出包含逐样本CSV、JSON/CSV指标、Markdown/LaTeX表格及逐来源、macro、overall和
worst-source统计。

## 8. 数据集与实验协议

### 8.1 ForenSynths / Self-Synthesis

已实现协议使用 `/data/zhy/CNNDetection/dataset` 的ProGAN训练数据，仅选择 car、cat、chair、
horse四类：训练144,024张（72,012 real、72,012 fake），官方validation 1,600张（800/800）。
real来源标为LSUN，fake生成器标为ProGAN。

Self-Synthesis测试manifest位于 `/data/zhy/GANGen-Detection`，包含9个来源：AttGAN、BEGAN、
CramerGAN、InfoMaxGAN、MMDGAN、RelGAN、S3GAN、SNGAN、STGAN。每个来源4,000张（2,000 real、
2,000 fake），合计36,000张。严格协议训练RandomCrop(128)，测试CenterCrop(128)，默认小图报错，
默认阈值0.5，逐来源报告Acc/AP。根据本报告的结果来源限制，不在此混入其他run的实验数字。

### 8.2 GenImage

项目实现SDv1.4单生成器训练和八来源测试。原始SDv1.4训练集为324,000张；指定运行审计后排除
3个空文件，实际训练323,997张（162,000 real ImageNet、161,997 SDv1.4 fake）。3,858张小图
未删除，而由反射填充处理。

测试manifest共100,000张、真假各50,000：

| 来源 | real | fake | 合计 |
|---|---:|---:|---:|
| Midjourney | 6,000 | 6,000 | 12,000 |
| SDv1.4 | 6,000 | 6,000 | 12,000 |
| SDv1.5 | 8,000 | 8,000 | 16,000 |
| ADM | 6,000 | 6,000 | 12,000 |
| GLIDE | 6,000 | 6,000 | 12,000 |
| Wukong | 6,000 | 6,000 | 12,000 |
| VQDM | 6,000 | 6,000 | 12,000 |
| BigGAN | 6,000 | 6,000 | 12,000 |

仓库文档将数据、裁剪、优化器和评测组织定位为参考 *Beyond Semantic Features: Pixel-level
Mapping for Generalized AI-Generated Image Detection* 的协议。代码搜索和实际forward表明，CIFP
没有实现或调用该论文的Pixel-level Mapping和Patch Shuffle；它们不属于CIFP方法。

### 8.3 评价指标

实现指标包括Accuracy、Average Precision、AUROC、FPR、Recall、Precision、2×2 confusion
matrix、逐来源结果、macro mAcc/mAP、样本加权overall Acc/AP，以及worst-source Acc/AP。
`real=0`、`fake=1`，fake为正类。

## 9. GenImage 主实验结果

本节唯一结果来源为
`outputs/20260807T104523/evaluation/genimage_test_oracle`。该目录记录
`threshold_selection=test_oracle_accuracy`，在完整10万张test manifest上选择的单一全局阈值为
`0.008913927711546421`。同时，如7.4所述，该test manifest也用于逐epoch checkpoint选择。以下
数值是这一明确选择流程下的结果。

### 9.1 每生成器结果

| 测试来源 | Acc | AP | AUROC | FPR | Recall |
|---|---:|---:|---:|---:|---:|
| ADM | 74.40% | 84.33% | 84.82% | 11.08% | 59.88% |
| BigGAN | 82.72% | 91.01% | 92.05% | 9.63% | 75.07% |
| GLIDE | 88.78% | 94.96% | 94.83% | 10.27% | 87.82% |
| Midjourney | 86.35% | 92.79% | 91.75% | 10.83% | 83.53% |
| SDv1.4 | 94.68% | 99.86% | 99.91% | 10.60% | 99.97% |
| SDv1.5 | 94.53% | 99.86% | 99.85% | 10.80% | 99.86% |
| VQDM | 84.02% | 91.95% | 91.86% | 10.48% | 78.52% |
| Wukong | 94.13% | 99.24% | 99.20% | 10.55% | 98.80% |

### 9.2 平均结果

| 汇总口径 | Acc | AP | AUROC | FPR | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Macro（八来源等权） | 87.45% | 94.25% | — | — | — | — |
| Overall（10万样本） | 87.73% | 95.18% | 94.47% | 10.54% | 86.01% | 89.08% |

worst-source Accuracy和AP均来自ADM，分别为74.40%和84.33%。overall confusion matrix为：

```text
TN=44,729  FP=5,271
FN=6,996   TP=43,004
```

### 9.3 跨生成器泛化分析

模型仅用SDv1.4训练。选择后结果中，相关版本SDv1.5的Acc/AP达到94.53%/99.86%；未见的Wukong
达到94.13%/99.24%，GLIDE达到88.78%/94.96%。Midjourney、VQDM和BigGAN保持84%上下或更高
Accuracy以及90%以上AP。ADM最弱，表明当前表示对不同生成范式的迁移程度并不一致。

AP普遍高于Accuracy所反映的绝对决策表现，且最终全局阈值显著低于0.5，说明样本排序能力强于
概率刻度的一致性。这个阈值是全test manifest标签参与选择后的结果，不能表述为预先固定的
0.5协议结果。

### 9.4 当前实验能够支持的结论

现有选择后结果表明：SDv1.4单生成器训练得到的组合式表示可迁移到多个未见生成器，尤其对
SDv1.5、Wukong和GLIDE保持较高表现；同时ADM显示仍有明显薄弱来源。由于同一test manifest
参与checkpoint和阈值选择，本结果适合作为当前实现能力与上界的事实记录；若用于严格独立测试
结论，需要固定独立validation上的checkpoint和阈值后再评测一次。

【实验依据】`evaluation/genimage_test_oracle/metrics.json`、`metrics.csv`、`predictions.csv`

## 10. CIFP 机制分析

本节主要来源为 `outputs/20260807T104523/analysis`，分析归档覆盖同一10万张GenImage测试图。

### 10.1 基元使用与字典有效性

32个基元的总体使用率均超过 `1e-4`，`effective_primitive_count=32`。最高使用率基元为8
（7.02%）、13（5.69%）、26（5.56%）、4（4.88%）和5（4.85%），前5项合计28.00%。平均
图像级使用熵为1.661 nat，相当于 `exp(1.661)=5.27` 个均匀基元；结合checkpoint的字典Gram
统计，不显示“所有字典方向相同”或“只有一个有效基元”的坍缩。

但分布并非完全均匀，而且单图可以高度集中。因此准确表述是：字典在数据集总体上32项均被
使用，单张图像通常只突出其中较少一部分。

真假分布存在明显差异。fake富集最明显的是基元8、1、24、21、3，fake-real使用率差分别为
6.18、5.24、5.11、5.01、4.89个百分点；real富集最明显的是基元13、14、4、5、26，差值为
7.58、7.22、7.17、6.11、5.85个百分点。这是统计关联，不把任何基元定义为真实中心。

### 10.2 不同生成器的基元组合

逐生成器usage显示不同来源具有不同峰值组合。例如Midjourney对基元29、24、21的平均使用较高，
BigGAN对基元1和8较高，GLIDE对基元8、24、30较高。不同来源又共享大量相同字典项。

在32维平均使用分布上，SDv1.4与SDv1.5的L1距离仅0.0362，是所有fake生成器对中最接近的一对；
BigGAN–VQDM和ADM–VQDM约为0.343。这一结果支持“不同生成器形成不同但有共享成分的组合”这一
初步解释。

基元激活与九类目标（8个fake生成器加real）的平均互信息为0.07863 nat；最高的是基元13
（0.1965）、12（0.1609）、19（0.1586）、21（0.1490）。实现先按每个基元使用率的中位数将其
二值化，再计算互信息。由于目标中 `real` 占一半，该结果混合了真假关联与fake生成器身份关联，
不能单独证明或否定基元—生成器一一对应；当前没有fake-only MI输出。

项目虽然实现了 `primitive_coverage`，但指定analysis目录没有 `train_features.npz` 或
`coverage.json`，因此当前没有足够依据报告top-r训练/未知重叠率和组合新颖度。架构上未知样本
只能激活训练形成的固定字典，逐生成器usage也确认它们实际激活多个字典项；更强的“未知组合
由已见基元构成”定量结论仍待coverage实验。

### 10.4 基元遮蔽分析

遮蔽分析直接使用缓存的 `A`：将指定基元列置0、对每个Patch剩余激活重新归一化，再通过原组合
池化器和真假头重分类；它不重新运行图像、backbone或字典。基线使用固定阈值0.5，overall
Acc/AP为80.622%/95.182%。

| 被遮蔽基元 | Acc下降 | AP下降 | 解释边界 |
|---:|---:|---:|---|
| 1 | 2.831 pp | 0.833 pp | 当前最稳定的重要基元 |
| 21 | 1.627 pp | 0.266 pp | 对预测有补充贡献 |
| 25 | 1.181 pp | 0.087 pp | 对阈值结果影响较明显 |
| 12 | 1.022 pp | 0.113 pp | 有独立贡献 |
| 28 | 0.909 pp | 0.167 pp | 有独立贡献 |

使用最多的基元8被遮蔽后Acc/AP只下降0.417/0.089个百分点，说明总体频率不等于分类重要性。
随机种子42的一次随机遮蔽选中基元2，Acc/AP下降0.506/0.177个百分点；只有一次随机样本，不能
作为稳定统计。遮蔽任一单基元后的性能仍远高于随机，且多个不同基元会造成下降，支持预测不是
由唯一基元承担；但本次 `mask_count=1`，不能据此量化多基元协同因果效应。

基元29、30被单独遮蔽后Acc分别提高1.837和0.981个百分点，AP也略升0.081和0.088个百分点；
这可能来自冗余、误导性证据或分数边界变化，不能仅凭该结果直接删除基元。

### 10.5 内容语义去偏分析

`content_mi.json` 中 semantic class和content environment MI均为0，但测试归档的10万项
`semantic_class` 全为空字符串、`content_env` 全为−1。单一类别的互信息必然为0，所以该输出
不能证明内容不变。

`z_for`线性探针在同一测试归档内按80/20分层随机切分，使用StandardScaler和LogisticRegression：

| 目标 | Probe Accuracy | 有效解释 |
|---|---:|---|
| true/fake | 86.99% | `z_for`含明显真假线性信息 |
| generator（含real类） | 60.03% | 含一定来源信息，但多数类real基线为50% |
| real_source | 86.99% | 本数据中等价于真假标签，不是独立证据 |
| semantic_class | 无法训练 | 全部为空 |
| content_env | 无法训练 | 全部为−1 |

因此内容去偏的当前有效证据主要是训练机制和环境头训练统计；指定测试analysis缺少可用内容标签，
不能完成跨内容MI/probe验证。

### 10.6 单图基元激活分析

使用现有 `extract_analysis_features` 和 `visualize_activations`，以指定运行的解析配置和
`best_validation_ap.pt` 对以下固定图像执行单图推理：

```text
/data/zhy/GenImage/stable_diffusion_v_1_4/train/ai/658_sdv4_00138.png
```

输出位于 `outputs/20260807T104523/analysis/example_658_sdv4_00138/`。为保留完整画面，实际分析先使用
Pillow 12.3.0的bicubic resize将原始512×512图像整体缩放到128×128，没有执行空间裁剪；
缩放图和处理记录分别保存为 `input_full_image_resized_128.png` 和 `preprocessing.json`。
因输入已恰好为128×128，后续transform的全幅crop操作是恒等映射，不会丢失画面。模型仍产生
8×8 Patch网格；数值结果同时写入 `analysis_summary.json`。

这种全图缩放是为完整观看空间激活所做的单图分析预处理，不是主实验的
CenterCrop(128)评测协议。模型输出fake logit −0.4216、fake probability 0.396135；
在固定0.5阈值下该已知fake样本被预测为real。

| Top基元 | 图像平均使用率 | 最大Patch激活 | 非零Patch数 | 最强区域（row,col） |
|---:|---:|---:|---:|---|
| 8 | 0.16427 | 0.94189 | 39/64 | (4,7)、(6,0)、(7,3)、(2,4) |
| 3 | 0.12699 | 0.62939 | 35/64 | (5,6)、(2,3)、(2,1)、(4,6) |
| 30 | 0.10459 | 0.70801 | 25/64 | (5,5)、(3,2)、(4,5)、(7,0) |

全图有22个基元至少在一个Patch中非零，top-3平均使用率合计39.58%，没有出现单一或两个
基元支配整张图像的情况。每个Patch恰有4个非零激活，平均Patch熵1.091 nat，低于4项
均匀分布的 `ln(4)=1.386`；这说明top-4内有一定集中性，但集中程度远弱于先前的中心裁剪输入。

基元8的最强点分布在右边缘、左下和下部Patch；基元3同时在面部、右侧和左上区域响应；
基元30在躯干/尾部附近、面部与底部Patch上有较强值。三张热图均保留完整猫体和背景，
显示不同基元在空间上存在重叠又不完全相同的响应。

该单图支持“每个Patch遵守top-4结构稀疏性”和“多个基元存在不同空间响应”两个有限结论；
图像级使用分布则是多基元分散组合，不是少数基元绝对主导的个例。热图只是模型分配可视化，
不能据此把基元解释为某种确定纹理、猫的语义部位或人类可命名生成伪影。由于完整图像缩放改变了
相对于正式CenterCrop协议的局部尺度，且该fake样本在0.5阈值下未被检出，这一个例不能作为单图
检测能力证据；它的用途仅是观察完整画面上的基元分配。

原analysis目录已有的48张热图来自归档前16项，而这16项全部是Midjourney目录中的真实图，
不适合直接做真假或跨生成器视觉对照；本次固定图像分析补充了一个完整画面的SDv1.4 fake个例，
但仍只是单样本观察。旧中心裁剪产物仅保留在 `center_crop_backup/` 中，不用于本报告的当前结论。

## 11. CIFP 与残差型方法的本质区别

SemTrace的定位是对真实正常特征进行预测，再通过预测值与实际特征之间的残差恢复生成痕迹。
CIFP当前代码没有对应路径：它不训练正常预测器，不重建图像或特征，不计算
`actual feature - expected feature`，也不以距真实中心或最近真实原型的距离分类。

CIFP直接将局部Patch特征映射为共享字典上的稀疏激活，以激活的mean/max组合形成 `z_for`，再
直接判别真假。二者的痕迹定义、训练目标和推理数据流不同。CIFP的Transformer内部保留标准
残差连接，但这些是DINOv3骨干结构，不被当作取证残差。仓库依赖和源码均无SemTrace import；
运行时也不引用 `/data/zhy/SemTrace`。

## 12. 当前已经实现的完整功能清单

| 类别 | 功能 | 是否实现 | 是否用于主实验 | 代码位置 |
|---|---|---:|---:|---|
| 模型 | DINOv3 ViT-S/16学生加载与动态Token切分 | 是 | 是 | `models/backbone.py` |
| 模型 | 最后2个block微调、LayerNorm独立冻结 | 是 | 是 | `models/backbone.py` |
| 模型 | 384→256局部投影与L2归一化 | 是 | 是 | `models/primitives.py` |
| 模型 | 32×256可学习归一化字典 | 是 | 是 | `models/primitives.py` |
| 模型 | 温度0.1、top-4 masked Softmax | 是 | 是 | `models/primitives.py` |
| 模型 | mean/max组合与128维 `z_for` | 是 | 是 | `models/composition.py` |
| 模型 | 真假Linear head仅访问 `z_for` | 是 | 是 | `models/composition.py` |
| 扩展 | 邻接对称共现投影 | 是 | 否 | `models/composition.py` |
| 内容环境 | 冻结DINOv3离线语义教师 | 是 | 是，训练前 | `environments/teacher.py` |
| 内容环境 | float16 memmap断点续跑 | 是 | 是 | `environments/store.py` |
| 内容环境 | MiniBatchKMeans 100环境 | 是 | 是 | `environments/clustering.py` |
| 内容环境 | GRL和100类环境头 | 是 | 是，仅训练 | `models/adversarial.py` |
| 损失 | BCE、稀疏、均衡、多样性、环境CE | 是 | 是 | `losses/` |
| 数据 | manifest-only严格Dataset | 是 | 是 | `data/dataset.py` |
| 数据 | ForenSynth/Self-Synthesis builder | 是 | 非本指定结果 | `data/builders.py` |
| 数据 | GenImage SDv1.4/八来源 builder | 是 | 是 | `data/builders.py` |
| 数据 | Random/CenterCrop与reflect-pad变体 | 是 | 是 | `data/transforms.py` |
| 训练 | 环境/标签平衡batch sampler | 是 | 是 | `data/sampler.py` |
| 训练 | Adam、bf16、梯度累积 | 是 | 是 | `cli/train.py`、`engine/trainer.py` |
| 训练 | 原生NCCL DDP | 是 | 否，主run单GPU | `engine/distributed.py` |
| 工程 | last/best AP/best Acc与完整resume状态 | 是 | 是 | `engine/checkpoint.py` |
| 评测 | 固定阈值和可选test-oracle全局阈值 | 是 | oracle用于指定结果 | `engine/evaluator.py` |
| 评测 | per-source/macro/overall/worst-source | 是 | 是 | `metrics/binary.py` |
| 分析 | usage、MI、masking、linear probe、热图 | 是 | 是 | `analysis/`、`cli/` |
| 分析 | primitive coverage | 是 | 未执行 | `analysis/statistics.py` |
| 基线 | Patch mean、CLS | 是 | 未见指定run结果 | `models/baselines.py` |
| 消融 | random dictionary、K=1、dense、no-comp、no-nuisance、random env、frozen | 是，配置/路径 | 未见指定run结果 | `configs/ablation/` |
| 可选后端 | Entmax15 | 否 | 否 | 当前项目中未找到实现 |
| 外部日志 | W&B | 否 | 否 | 当前项目中未找到实现 |

## 13. 当前已经完成的实验与分析清单

| 实验/分析 | 状态 | 输出目录 | 主要结论 |
|---|---|---|---|
| GenImage SDv1.4完整训练 | 已完成，200 epochs | `outputs/20260807T104523/` | 生成best AP/Acc和last checkpoint |
| 八来源test-oracle评测 | 已完成 | `evaluation/genimage_test_oracle/` | mAcc 87.45%，mAP 94.25% |
| 基元使用分析 | 已完成 | `analysis/usage.json` | 32基元均有效使用，平均熵1.661 |
| 字典方向审计 | 已完成，本报告只读计算 | best-AP checkpoint | 无绝对余弦≥0.9的基元对 |
| 生成器MI | 已完成 | `analysis/generator_mi.json` | 平均MI 0.07863，含real类别混杂 |
| 内容MI | 已执行但标签不足 | `analysis/content_mi.json` | 空semantic和单一−1环境，不能验证内容不变 |
| 线性探针 | 已完成 | `analysis/linear_probes.json` | 真假86.99%，含real的generator为60.03% |
| 单基元遮蔽 | 已完成 | `analysis/masking.json` | 基元1影响最大；频率不等于重要性 |
| 测试集热图 | 已完成 | `analysis/activation_maps/` | 48图，但样本均为前16张真实Midjourney图 |
| 指定SDv1.4 fake单图分析 | 本次已完成 | `analysis/example_658_sdv4_00138/` | 全图bicubic缩放、无空间裁剪；top基元8/3/30 |
| Train-vs-unknown coverage | 未执行 | 当前无 `coverage.json` | 当前项目中未找到足够结果依据 |
| 共现增强实验 | 未见指定run | 无指定输出 | 不能写成主实验已验证模块 |
| 其他基线/消融训练 | 未见指定run | 无指定输出 | 配置存在，不报告性能 |

## 14. 当前方法的技术特点总结

1. **直接组合判别而非残差判别。** 模型从局部特征到字典激活再到组合表示，没有正常参照、
   重建和差分路径。
2. **结构化分类访问边界。** 真假头只访问128维 `z_for`，而 `z_for` 只由基元mean/max统计构造；
   语义教师、CLS、register、环境标签和来源元数据不进入真假头。
3. **局部共享与稀疏激活。** 每个Patch在32个基元中最多激活4个，允许多基元软组合，同时
   字典在全数据集上保持32项有效使用。
4. **训练期内容环境约束。** 离线DINOv3 embedding和MiniBatchKMeans构造100个环境，GRL使
   `z_for`承受反向环境梯度；推理完全移除教师和环境支路。
5. **跨来源结果已有但选择过程需明确。** 指定结果在八来源上达到选择后mAcc 87.45%、mAP
   94.25%；同一test manifest参与逐epoch checkpoint选择和全局阈值选择，后续使用时必须保留
   这一实验过程说明。
6. **机制证据是初步而非完备。** usage、字典Gram、generator分布、masking和单图热图支持
   多基元及组合使用；内容标签和coverage产物不足，尚不能完整验证内容不变与新组合假设。

## 15. 供后续开题报告写作提取的事实摘要

### 15.1 主要研究内容可用事实

- CIFP研究SDv1.4等有限训练生成器条件下，如何通过局部可复用取证基元提高未知生成器检测。
- 主模型将DINOv3最后一层Patch Token投影到256维局部空间，与32项可学习字典进行top-4稀疏
  匹配，再从mean/max激活统计构造128维组合式取证表示。
- 内容去偏由离线DINOv3内容伪环境和训练期GRL环境头实现；语义信息不作为真假分类输入。
- 当前指定运行已完成GenImage SDv1.4单生成器训练、八来源评测和多项机制分析。

### 15.2 研究方法可用事实

- 视觉基础模型：`facebook/dinov3-vits16-pretrain-lvd1689m`，实际hidden 384、patch 16、4个
  register token；只使用64个真实Patch Token，不使用CLS做CIFP分类。
- 基元字典：`D∈R^{32×256}`，正态初始化、每次forward归一化、端到端学习、无人工标签。
- 稀疏激活：归一化Token与字典的温度0.1余弦相似度，top-4 mask后Softmax。
- 组合聚合：逐基元mean和max拼接为64维，经MLP形成128维 `z_for`；主实验未启用共现。
- 对抗去偏：768维离线语义embedding、100类MiniBatchKMeans环境、GRL和环境CE。
- 损失：BCE + 0.1×组合正则 + 0.1×环境CE；组合内部稀疏/均衡/多样性权重0.1/1/0.1。

### 15.3 技术路线可用事实

训练前：

```text
训练manifest
→ 冻结DINOv3教师
→ concat(L2(CLS), L2(mean Patch))
→ float16 memmap
→ 平衡抽样200,000
→ MiniBatchKMeans(C=100)
→ 为全部训练样本写入content_env
```

训练：

```text
图像
→ DINOv3学生true Patch Tokens
→ 256维局部投影
→ 32项字典top-4激活
→ mean/max组合
→ z_for
├→ 真假头 + BCE
└→ GRL + 环境头 + CE
同时对激活熵、总体使用均衡和字典方向多样性正则化
```

推理：

```text
图像 → 学生 → Patch局部特征 → 基元稀疏激活
     → mean/max → z_for → fake logit → sigmoid概率
```

### 15.4 实验方案可用事实

- ForenSynth/Self-Synthesis协议代码：ProGAN car/cat/chair/horse训练，9个Self-Synthesis来源测试。
- GenImage协议代码：SDv1.4训练，Midjourney、SDv1.4、SDv1.5、ADM、GLIDE、Wukong、VQDM、
  BigGAN测试。
- 指定运行实际采用过滤3个空文件并对小图反射填充的非严格预处理变体，单GPU batch 4096。
- 评测实现Acc、AP、AUROC、FPR、Recall、Precision、macro/overall/worst-source和逐样本结果。
- 已有分析包括usage、字典方向、generator MI、content MI、linear probe、单基元masking和热图；
  coverage代码存在但指定run没有结果，共现和其他消融没有指定实验输出。
- 指定主结果使用完整test manifest选择checkpoint，并选择overall Accuracy最优的单一全局阈值
  0.0089139；mAcc 87.45%、mAP 94.25%。

### 15.5 可行性分析可用事实

- 完整训练、评测、checkpoint和机制分析链路已在真实GenImage数据上运行完成。
- 指定训练使用一张RTX 5090、bf16、batch 4096、200 epochs、15,800 optimizer steps；从配置
  写入到last checkpoint约27小时27分钟，但其中包含每epoch完整10万张评测和落盘。
- 完整测试集一次test-oracle评测日志约3分02秒；10万张analysis特征提取日志约2分48秒。
- 项目使用PyTorch 2.11.0+cu130、torchvision 0.26.0、Transformers 5.14.0和Python 3.11，
  依赖由uv.lock固定。
- 运行metadata记录cuDNN编译9.19与运行时9.8不一致的查询警告；训练和评测产物已完成，但后续
  复现应使用干净动态库环境。当前日志未提供峰值显存。

### 15.6 已有科研基础可用事实

- 已完成DINOv3 Patch提取、可控微调、局部投影、基元字典、稀疏分配、组合池化、真假头。
- 已完成离线语义特征断点存储、内容聚类、环境审计、GRL与环境分类。
- 已完成两套数据manifest协议、严格图像异常处理、环境平衡sampler和DDP代码。
- 已完成Adam/bf16训练、三类checkpoint、随机状态恢复、多格式日志和逐epoch评测。
- 已完成指定GenImage 200-epoch训练、10万张八来源选择后评测及主要基元机制分析。
- 已完成指定SDv1.4生成图像的全图缩放、无空间裁剪单图top基元热图和数值分析。
- 当前测试结果为69 passed、1 skipped；跳过项是需显式设置环境变量并提供至少两张可见GPU的
  DDP smoke test，本次未启动该额外多GPU进程。
