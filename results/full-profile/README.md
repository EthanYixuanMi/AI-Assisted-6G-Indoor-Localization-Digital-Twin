# Full-profile result package

[English](#english) · [中文](#中文)

## English

### Source run

| Field | Value |
|---|---|
| Profile | `full` |
| Source directory | `qa/residual_optimization/full_scale050_cap090_20260727` |
| Success | `true` |
| Duration | 1,965.82 seconds |
| Root seed | 42 |
| Evaluation seeds | 42, 123, 2026, 31415, 27182 |
| Python | 3.12.4 |
| Platform | Windows 11, CPU-only |

The source run passed its internal 50-file output contract. Its original
`manifest.json` is retained here as provenance, even though this GitHub
package intentionally excludes several large, regenerable artifacts.

This run uses the optimized Residual AI policy shipped in the current source:
the learned correction is scaled by `0.5`, and its norm is capped at the
training-residual `0.90` quantile. Each seed learns its cap from its own
training split; the spatial holdout is not used for this threshold.

For privacy and portability, the four machine-specific `_meta` paths in each
packaged `config_resolved.yaml` were rewritten as repository-relative paths.
All scientific parameters are unchanged. The config hash recorded in the
immutable source-run visual manifest therefore refers to the original private
copy, while the public copy is the readable configuration shipped here.

### Included

- `figures/`: selected paper-facing PDF/PNG/SVG visuals;
- `screenshots/`: five saved Dashboard scenarios;
- `tables/`: compact report CSV tables;
- `latex/`: complete LaTeX table environments and the generated result snippet;
- `provenance/visual_asset_manifest.json`: plotting protocol and source hashes;
- core configuration, environment, seeds, metrics, robustness, ablation,
  runtime, training log, result summaries, and source-run manifest.

### Deliberately excluded

| Artifact | Reason |
|---|---|
| `models/**/*.joblib` | Full Residual-AI checkpoints exceed 200 MB each and cross GitHub's 100 MB per-file limit. |
| Full `per_sample_predictions.csv` | Approximately 59 MB and reproducible from the saved configuration/seeds. |
| `data/splits/*` and dataset arrays | Large intermediate training/evaluation data. |
| Per-seed model directories | Large and not required to inspect the reported numbers. |
| Large plotting-source CSVs | Figures, compact tables, protocol metadata, and regeneration code are included. |
| `los_nlos_comparison.*` | Omitted because its saved panels use different evaluation protocols; the underlying rows remain in the CSVs. |
| `robustness_anchor_failure.*` | Omitted to avoid presenting pooled random/fixed-critical failures without the protocol caveat; use the CSV or the explicitly captioned combined figure. |

The repository-level `outputs/latest/per_sample_predictions.csv` is a compact
4,500-row Dashboard replay assembled from the five saved scenario trajectories.
It is for interactive demonstration, not for recomputing the aggregate tables.

### Evidence boundaries

- `tables/main_results.csv` aggregates five independently generated and trained
  spatial-holdout runs.
- `figures/error_cdf.*` uses the root-run seed-42 spatial-holdout realization.
  The independently regenerated five-seed aggregation uses separate
  measurement realizations, so the CDF is neither pooled nor a row from that
  aggregate.
- Robustness sweeps keep the root-seed model fixed and vary measurement seeds.
- The combined anchor-failure panel pools random and fixed-critical protocols
  at overlapping failure counts; use `robustness_results.csv` for separate rows.
- LoS/NLoS labels come from simulated geometric wall/obstacle intersections,
  not measured radio-link ground truth.

### Regenerate the omitted artifacts

```bash
python scripts/run_pipeline.py --profile full
```

## 中文

### 来源运行

| 字段 | 数值 |
|---|---|
| Profile | `full` |
| 来源目录 | `qa/residual_optimization/full_scale050_cap090_20260727` |
| 成功状态 | `true` |
| 耗时 | 1,965.82 秒 |
| Root seed | 42 |
| Evaluation seeds | 42、123、2026、31415、27182 |
| Python | 3.12.4 |
| 平台 | Windows 11、纯 CPU |

来源运行通过了内部 50 文件输出契约。这里保留原始 `manifest.json` 作为追溯，
但 GitHub 发布包有意排除了可重新生成的大文件。

本次运行使用当前源码中的优化 Residual AI 策略：学习到的修正量乘以 `0.5`，
并将修正向量范数限制在训练残差的 `0.90` 分位数以内。每个 seed 都只使用
自己的训练集计算 cap，不使用空间留出测试集。

为保护隐私并提高可移植性，两份公开 `config_resolved.yaml` 中各有四个本机
`_meta` 路径被改写为仓库相对路径，所有科学参数均未变化。因此，原始可视化
manifest 中记录的配置哈希对应未公开的源运行副本；本目录提供的是可阅读的
脱敏公开副本。

### 已包含

- `figures/`：精选论文 PDF/PNG/SVG 图；
- `screenshots/`：五种 Dashboard 场景；
- `tables/`：紧凑的报告 CSV；
- `latex/`：完整 LaTeX 表格环境和生成的结果片段；
- `provenance/visual_asset_manifest.json`：绘图协议和源文件哈希；
- 核心配置、环境、seed、指标、鲁棒性、消融、运行时间、训练日志、结果摘要
  和来源运行 manifest。

### 有意排除

| 文件 | 原因 |
|---|---|
| `models/**/*.joblib` | Full Residual-AI checkpoint 单文件超过 200 MB，超过 GitHub 100 MB 限制。 |
| 完整 `per_sample_predictions.csv` | 约 59 MB，可由配置和 seed 重新生成。 |
| `data/splits/*` 和数组 | 大型中间训练/评估数据。 |
| 各 seed 模型目录 | 文件很大，查看报告结果时不需要。 |
| 大型绘图源 CSV | 已保留图片、紧凑表格、协议 metadata 和重生成代码。 |
| `los_nlos_comparison.*` | 保存图中的不同面板使用了不同评估协议，因此不直接发布；底层结果仍保存在 CSV 中。 |
| `robustness_anchor_failure.*` | 为避免在缺少协议说明时混合 random 与 fixed-critical 失效而不发布；请读取 CSV 或使用已明确标注的组合图。 |

仓库根目录的 `outputs/latest/per_sample_predictions.csv` 是由五种场景保存轨迹
整理出的 4,500 行轻量回放，只用于交互演示，不用于重新计算聚合表。

### 证据边界

- `tables/main_results.csv` 聚合五次独立数据生成和训练后的空间留出结果；
- `figures/error_cdf.*` 使用 root run 的 seed-42 空间留出 realization；
  五-seed 聚合使用独立重生成的 measurement realization，因此该 CDF 既不是
  pooled CDF，也不对应聚合表中的某一行；
- 鲁棒性扫描固定 root-seed 模型，只改变 measurement seed；
- 组合锚点失效图在相同 failure count 处合并 random 和 fixed-critical，
  精确分离结果应读取 `robustness_results.csv`；
- LoS/NLoS 来自模拟几何穿墙规则，不是实测无线链路标签。

### 重新生成被排除的文件

```bash
python scripts/run_pipeline.py --profile full
```
