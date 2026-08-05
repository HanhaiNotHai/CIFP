# CIFP implementation plan

## Completion contract

CIFP will be complete when the repository can be installed with uv, both manifests can be
built from the audited read-only roots, the model and all requested losses pass deterministic
tests, a synthetic and a real-data training step succeed, and evaluation emits per-source and
aggregate reports. Full 200-epoch training is deliberately not part of this implementation run.

## Phases

1. **Environment and protocol audit** — record hardware, dependencies, model metadata, dataset
   layouts, aliases, counts, and SemTrace independence.
2. **Project foundation** — add the Python 3.11 uv project, structured YAML configuration,
   package/CLI skeleton, logging, and test configuration.
3. **Data protocol** — implement bounded audits, Parquet manifests, strict image loading and
   transforms, and the deterministic environment-balanced sampler.
4. **CIFP model** — test then implement dynamic DINOv3 patch extraction, projector, dictionary,
   sparse assignment, compositional pooler, classifier, cooccurrence, GRL, and losses.
5. **Offline environments** — implement resumable float16 feature shards, balanced fitting
   selection, MiniBatchKMeans, full assignment, reports, and fixed random-environment ablation.
6. **Training and evaluation** — implement native DDP, bf16 checks, optimizer/checkpoint resume,
   four logging sinks, distributed prediction gathering, metrics, and tables.
7. **Baselines and analysis** — expose the requested config-driven baselines/ablations and seven
   analysis CLIs without adding another training framework.
8. **Verification** — run focused tests incrementally, then uv sync, ruff, pytest, data audit,
   manifest builds, synthetic smoke, one real batch, and a GPU smoke when resources permit.
9. **Final audit** — inspect the diff, document tensor access and prohibited-path absence, and
   report every passed, failed, or unexecuted command exactly.

## Smallest implementation choices

- Use dataclasses plus PyYAML instead of a configuration framework.
- Use pandas/pyarrow for manifests and reports, scikit-learn for metrics/clustering, and native
  PyTorch for model, DDP, logging, and checkpoints.
- Use masked top-k softmax only; optional entmax15 and W&B are excluded.
- Keep semantic-teacher tooling in `cifp.environments`; the `CIFP` model receives only a forensic
  student backbone.
- Reuse no SemTrace code or imports. Its read-only inspection only informs generic layout checks.

## Verification ladder

Each behavior follows red/green/refactor: focused unit test, minimal implementation, focused test,
then the next slice. Expensive work stops at 20 optimizer steps; no dataset or model download is
authorized.

## Implementation status

Phases 1–9 are implemented. Both main parquet protocols, actual DINOv3 patch/teacher checks, a
real-data bf16 optimizer step, a two-GPU NCCL smoke, checkpoint resume, and a 72-row per-source
evaluation smoke have completed. The formal audit contains complete directory/file counts and a
1,000-header sample per root; the optional exhaustive 3.45-million-header pass was stopped for
cost. No 200-epoch training or full-dataset evaluation is part of this run.

## Authoritative sources

- DINOv3 token ordering and dynamic configuration:
  https://huggingface.co/docs/transformers/model_doc/dinov3#notes
- uv PyTorch index configuration:
  https://docs.astral.sh/uv/guides/integration/pytorch/
- PyTorch distributed initialization:
  https://docs.pytorch.org/docs/stable/distributed.html#initialization
- PyTorch DDP:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- MiniBatchKMeans:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
- Reference evaluation protocol:
  https://ojs.aaai.org/index.php/AAAI/article/view/40927
