<p align="right">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<h1 align="center">AI 辅助的 6G 室内定位数字孪生</h1>

<p align="center">
  一个纯 CPU、以仿真为核心的室内定位框架，支持可复现实验、鲁棒性评估和交互式结果回放。
</p>

<p align="center">
  <img alt="Python 3.10–3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white">
  <img alt="CPU only" src="https://img.shields.io/badge/运行方式-纯%20CPU-2ea44f">
  <img alt="Tests" src="https://img.shields.io/badge/pytest-32%20项测试-brightgreen">
  <img alt="Full profile" src="https://img.shields.io/badge/结果-5%20seed%20Full%20Profile-f59e0b">
</p>

<p align="center">
  <img src="results/full-profile/figures/dashboard_overview.png"
       alt="AI 辅助室内定位数字孪生 Dashboard"
       width="100%">
</p>

## 项目解决什么问题？

墙体、障碍物、类似多径的结构化偏差、测量噪声和锚点掉线都会破坏传统几何定位
方法的理想假设。本项目将这些因素放进一个可配置的二维数字孪生中，并在完全
相同的保存场景下比较：

- **几何最小二乘（Geometric LS）**：透明、可解释的模型驱动基线；
- **KNN 指纹定位**：经典的场地相关方法；
- **Direct AI**：从信号特征直接预测坐标；
- **Residual AI**：学习对几何估计的残差修正；
- **Kalman 滤波**：单独用于连续轨迹后处理。

仓库把环境生成、RSS 类测量仿真、模型训练、空间留出评估、鲁棒性扫描、
论文图表和 Streamlit 结果回放连接成一条可复现流水线。

> **范围说明：**仓库中的所有数字和图片都来自纯软件仿真。本项目不声称已经
> 实现真实 6G 部署、标准化 3GPP 信道或真实建筑定位精度。

## 一览

| 项目 | 已打包 Full Profile 设置 |
|---|---|
| 环境 | 30 m × 20 m 二维室内地图 |
| 基础设施 | 6 个固定锚点 |
| 场景 | Normal、High Noise、Strong Blockage、Anchor Failure、Domain Shift |
| 主评估协议 | Normal spatial holdout |
| Full Profile seeds | 42、123、2026、31415、27182 |
| 每个 seed 的数据量 | 20,000 训练 / 4,000 验证 / 6,000 域内测试 / 4,000 空间留出 |
| 执行环境 | Python 3.12.4、Windows 11、纯 CPU |

## 系统架构

<p align="center">
  <img src="results/full-profile/figures/system_architecture.svg"
       alt="数字孪生系统架构"
       width="94%">
</p>

系统由四个可审计层次构成：

1. **环境孪生**：通过 YAML 定义锚点、墙体、障碍物、语义区域和轨迹；
2. **无线测量孪生**：生成距离、穿墙状态、LoS/NLoS、RSS、结构化偏差、
   噪声、硬件偏差和锚点可用性；
3. **定位与滤波**：四种点定位方法，以及只用于轨迹的 Kalman 滤波；
4. **评估与可视化**：误差指标、压力测试、CSV/JSON 追溯、论文图和 Dashboard。

## 五种可回放场景

<table>
  <tr>
    <td width="50%" align="center">
      <img src="results/full-profile/screenshots/dashboard_normal.png" alt="Normal 场景"><br>
      <strong>Normal</strong><br>
      标称传播条件和较低的随机掉线率。
    </td>
    <td width="50%" align="center">
      <img src="results/full-profile/screenshots/dashboard_high_noise.png" alt="High Noise 场景"><br>
      <strong>High Noise</strong><br>
      更强的 RSS 测量不确定性。
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="results/full-profile/screenshots/dashboard_strong_blockage.png" alt="Strong Blockage 场景"><br>
      <strong>Strong Blockage</strong><br>
      更大的墙损和 NLoS 偏差。
    </td>
    <td width="50%" align="center">
      <img src="results/full-profile/screenshots/dashboard_anchor_failure.png" alt="Anchor Failure 场景"><br>
      <strong>Anchor Failure</strong><br>
      锚点失效和可用性掩码。
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="results/full-profile/screenshots/dashboard_domain_shift.png" alt="Domain Shift 场景"><br>
      <strong>Domain Shift</strong><br>
      改变传播参数和硬件偏差。
    </td>
  </tr>
</table>

## Residual AI 优化更新

当前源码为 Residual AI 加入了保守的残差信任域：最终预测使用学习修正量
的 50%，同时将异常大的修正限制在训练集残差范数的第 90 百分位以内。
该上限只由训练集计算，不读取空间留出测试集。

使用 Quick profile 的 3 个 seed 进行独立数据生成和独立训练后，评估结果如下：

| 方法 | 跨 seed 平均误差 (m) | 平均 P90 (m) | 平均 NLoS 误差 (m) |
|---|---:|---:|---:|
| Geometric LS | 5.304 | 9.583 | 5.796 |
| Direct AI | 4.304 | **7.303** | 4.392 |
| 旧策略 Residual AI | 7.372 | 10.630 | 7.554 |
| **信任域 Residual AI** | **3.969** | 8.210 | **4.234** |

在这一 Quick-profile 协议下，优化后的 Residual AI 平均误差相对旧策略降低
**46.2%**，相对 Geometric LS 降低 **25.2%**，相对 Direct AI 降低
**7.8%**。每个 seed 的原始数值见
[`quick_three_seed_results.csv`](results/residual-optimization/quick_three_seed_results.csv)。
seed 42 和 123 用作开发诊断；未参与策略选择的 seed 2026 也从 7.088 m
降至 3.884 m，改善 **45.2%**。

## 优化后 Residual AI 的 Full Profile 结果

下表聚合 5 次独立数据生成和模型训练，每个 seed 使用 4,000 个空间留出样本。
Residual AI 使用当前的 50% 修正比例，以及仅由训练集确定的第 90 百分位上限。

| 方法 | 平均误差 (m) | RMSE (m) | 中位数 (m) | P90 (m) | 单行预热推理 (ms) |
|---|---:|---:|---:|---:|---:|
| Geometric LS | 5.36 | 6.16 | 4.81 | 9.58 | 5.19 |
| KNN | 7.43 | 7.92 | 7.11 | 11.08 | **1.61** |
| Direct AI | 4.27 | **4.74** | 4.08 | **6.82** | 3.77 |
| **Residual AI** | **3.90** | 4.88 | **3.29** | 7.97 | 25.50 |

优化后的 Residual AI 相对 Geometric LS 将平均误差降低 **27.1%**，
相对 Direct AI 降低 **8.5%**；它在 5 个 seed 上取得最优的平均误差、
中位数误差、LoS 误差和 NLoS 误差。Direct AI 仍具有最优 RMSE 和 P90，
且速度更快、模型更小，因此该精度提升伴随明确的部署成本。

<table>
  <tr>
    <td width="50%" align="center">
      <img src="results/full-profile/figures/error_cdf.png" alt="定位误差 CDF"><br>
      <strong>误差 CDF</strong><br>
      root-seed 空间留出诊断图；它既不是五个 seed 合并后的 CDF，也不对应
      独立重生成的五-seed 聚合表中的某一行。
    </td>
    <td width="50%" align="center">
      <img src="results/full-profile/figures/robustness_results.png" alt="鲁棒性结果"><br>
      <strong>鲁棒性</strong><br>
      噪声与锚点失效扫描。锚点面板在相同失效数量处合并了 random 和
      fixed-critical 两种协议。
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="results/full-profile/figures/ablation_results.png" alt="Residual 模型消融"><br>
      <strong>消融实验</strong><br>
      LoS/NLoS 特征对优化后的 residual 模型有帮助；在这一 seed 42
      消融中，移除 spatial-bias training 后误差略有降低。
    </td>
    <td width="50%" align="center">
      <img src="results/full-profile/figures/runtime_comparison.png" alt="运行时间比较"><br>
      <strong>运行时间</strong><br>
      本机 CPU 在线推理和离线训练测量。
    </td>
  </tr>
</table>

详细 CSV、LaTeX 表格、图片、截图、配置、seed 和原始运行 manifest 位于
[`results/full-profile/`](results/full-profile/README.md)。

## 三分钟启动 Dashboard

仓库在 `outputs/latest/` 中附带了一个 4,500 行的轻量回放集。它包含五个
场景中五种方法各 180 个轨迹帧，因此不下载模型、不重新训练也能直接浏览。
其中 `manifest.json` 记录精简文件的实际大小和哈希，
`source_run_manifest.json` 则保留原始 Full 运行记录。

创建 GitHub 仓库后，请将下面两条 clone 命令中的
`YOUR_GITHUB_USERNAME` 替换为仓库所有者的账号名。

### Windows PowerShell

```powershell
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Assisted-6G-Indoor-Localization-Digital-Twin.git
cd AI-Assisted-6G-Indoor-Localization-Digital-Twin

python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Assisted-6G-Indoor-Localization-Digital-Twin.git
cd AI-Assisted-6G-Indoor-Localization-Digital-Twin

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m streamlit run app.py
```

页面只会筛选和回放保存的仿真记录。切换 Dashboard 控件不会重新训练模型，
也不会静默生成新的测量值。

## 复现实验

所有命令都应在仓库根目录执行。

```bash
# 较快的端到端流程
python scripts/run_pipeline.py --profile quick

# 生成本仓库 Full Profile 表格所使用的五 seed 流程
python scripts/run_pipeline.py --profile full

# 从已有完整结果重新导出论文图和 Dashboard 截图
python scripts/export_report_assets.py \
  --results-dir outputs/latest \
  --output-dir report_assets \
  --include-dashboard
```

Quick profile 适合开发和流程检查。Full profile 会训练更大的模型并执行更广的
多 seed 评估，CPU 上通常需要约 15–40 分钟，具体取决于处理器。

可编辑安装后，也可以通过命令行入口运行：

```bash
localization-twin --profile quick
```

## 测试

```bash
python -m pytest -q --basetemp .test-tmp/pytest
```

32 项测试覆盖几何、传播、确定性数据生成、空间划分、全部定位器、Kalman
滤波、指标、可视化契约和一个微型端到端流水线。

## 仓库结构

```text
.
├── app.py                         # Streamlit 保存结果回放入口
├── config/                        # 默认、Quick 和 Full YAML
├── src/localization_twin/         # 仿真、模型、评估与报告导出
├── scripts/                       # 流水线和导出入口
├── tests/                         # 单元、可视化和 smoke tests
├── outputs/latest/                # GitHub 友好的轻量 Dashboard 回放
├── results/full-profile/
│   ├── figures/                   # PDF/PNG/SVG 论文图
│   ├── screenshots/               # 五种场景预览
│   ├── tables/                    # 结果 CSV
│   ├── latex/                     # 可直接引用的 LaTeX 表格
│   └── provenance/                # 图表协议与哈希
└── docs/                          # 实验设计与架构源文件
```

大型模型、完整数据划分和 59 MB 的全量逐样本预测没有放进 Git。它们由流水线
重新生成；轻量回放和所有报告级结果则直接包含在仓库中。精确边界参见
[结果包说明](results/full-profile/README.md)。

## 可复现性说明

- 所有随机过程由最终 YAML 和保存的 seed 控制；
- 主空间留出结果会针对每个 evaluation seed 独立生成数据并重新训练；
- 鲁棒性扫描固定 root-seed 模型，只改变 measurement seed，CSV 中明确保存
  两类字段；
- scaler、imputer 和模型选择仅使用 training/validation 数据；
- `manifest.json` 记录 profile、平台、包版本、样本量、算法和生成文件；
- 所有论文结论都必须限定在保存的仿真配置下。

## 局限性

这是一个轻量二维 RSS 数字孪生，不包含完整 3GPP Indoor Factory 信道、CIR、
CSI、天线阵列、同步误差、三维多径或动态人体遮挡，也没有物理系统的双向同步。
Domain Shift 只是受控的仿真压力测试，不能作为真实建筑迁移证据。

## 项目成员

Xin Bao · Chenghao Li · Yuhang Li · Yixuan Mi · Qihan Wu · Yuhan Wang ·
Chuchen Xu · Tingting Yang
