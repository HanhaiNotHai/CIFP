# CIFP method specification

## Scientific scope

CIFP tests whether unseen generators can be represented by new strengths and spatial combinations
of a finite set of reusable local forensic primitives. The primitives are learned without trace,
class, or generator labels. Content environments are training nuisances constructed offline, not
inference features.

## Offline semantic environments

For one protocol's training manifest, the frozen, evaluation-mode DINOv3 teacher returns the last
hidden state in config-defined order: CLS, register tokens, patch tokens. The semantic row is

`s = concat(L2(cls), L2(mean(patch_tokens)))`.

The extractor runs in `torch.inference_mode`, stores float16 rows in a resumable memmap, and writes
an immutable `row,path` index. A seed-42 round-robin sample over `(label, semantic_class, source)`
fits MiniBatchKMeans on at most 200,000 rows. Every training row is then assigned one of 100
environments. ForenSynths and GenImage have independent stores and clusterers.

## Forensic model

The DINOv3 student reads its hidden size, patch size, register-token count, and block count from its
config. Only true patch tokens enter CIFP; CLS is discarded. With a divisible input `(H,W)`, the
runtime grid is `(H/patch_h, W/patch_w)`. Earlier blocks are frozen; the default trains only the
last two blocks, while layer-normalization trainability is independently configured.

For patch token `F`, the local projector is

`H = L2(Linear(GELU(Linear(LayerNorm(F)))))`, with default output dimension 256.

The `K x d` learned dictionary is L2-normalized at every forward pass. Assignment logits are cosine
similarities divided by temperature 0.1. Masked top-4 softmax gives non-negative rows summing to
one with at most four nonzero entries. Dense softmax exists only as an explicit ablation.

The mandatory composition is `p = concat(mean_patch(A), max_patch(A))`. A LayerNorm/MLP maps `p`
to the 128-dimensional `z_for`; a single linear head maps only `z_for` to one fake logit. When
enabled, horizontal/vertical patch neighbors form the symmetric mean relation
`0.5 * (a_i outer a_j + a_j outer a_i)`. Its projected form is concatenated with mean/max before
forming `z_for`.

During training only, `z_for` also enters a gradient-reversal layer and environment classifier.
The GRL forward is identity and its backward multiplies gradients into `z_for` by `-lambda_grl`;
environment-head parameters receive the ordinary CE-minimizing gradient. GRL is zero before epoch
5, linearly reaches 1.0 at epoch 20, then remains one.

## Objective

`L = L_det + lambda_comp L_comp + lambda_nui L_nui`, where detection is BCEWithLogits with
`real=0`, `fake=1`, and environment nuisance is cross entropy. The composition term is

`L_comp = w_sparse L_sparse + w_balance L_balance + w_diversity L_diversity`.

- `L_sparse` is mean assignment entropy and sharpens the retained top-k entries.
- For global use `q`, `L_balance = sum(q log(q K + eps))` discourages primitive starvation.
- `L_diversity` is mean squared off-diagonal cosine Gram value of normalized dictionary rows.

Defaults are `lambda_comp=0.1`, `lambda_nui=0.1`, and weights `0.1/1.0/0.1`. They are initial CIFP
engineering settings, not claimed as parameters from another paper.

## Explicit exclusions

No component predicts a normal image or feature, reconstructs an input, subtracts a token and a
primitive, measures a distance to a real center, emits an anomaly score, or fuses semantic and
forensic features. There is no reconstruction loss. Standard internal Transformer skip connections
remain an implementation detail of DINOv3 and are not the CIFP representation.

## Reproducibility assumptions

- Seeds for Python, NumPy, PyTorch, manifests, validation decisions, clustering, and fixed-random
  environments are 42 unless explicitly overridden.
- Official validation splits are used; no 5% engineering split is needed for the audited roots.
- Evaluation threshold is always 0.5 and AP uses continuous fake probability.
- A formal aligned run is four GPUs x 32 samples with no accumulation. Six GPUs x 32 is recorded
  as a non-protocol global batch of 192.
