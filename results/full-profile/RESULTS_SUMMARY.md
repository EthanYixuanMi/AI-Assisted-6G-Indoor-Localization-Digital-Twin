# 结果摘要

本摘要只解释本地软件仿真的实测输出，不代表真实 6G 网络或真实建筑部署。

## 最重要的发现

- 独立仿真与训练 seed 的 Normal 空间留出结果中，平均误差最低的是 **Direct AI**：
  **4.27 m**，P90 为 **6.82 m**。
- Residual AI 相对 Geometric LS 的平均误差恶化为 36.0%；该结论仅适用于保存的配置和空间留出协议。
- LoS 条件下最优方法为 **Geometric LS**
  （3.67 m），NLoS 条件下最优方法为
  **Direct AI**（4.25 m）。
- 推理最快的是 **Direct AI**，本机预热后一行输入时间为
  **1.343 ms**。

## 稳定性与限制

噪声、墙体偏差、锚点掉线和 domain shift 会改变方法排序，因此不能从
Normal 场景推导“某方法始终最好”。模型是在同一类简化 RSS 数字孪生中训练
和测试的；空间留出可以降低位置泄漏，却不能消除 simulation-to-reality gap。

## 可以写进报告的结论

可以报告各方法在本次仿真中的绝对误差、相对 Geometric LS 的变化、LoS/NLoS
差异、鲁棒性曲线以及本机 CPU 延迟。必须同时写明配置、seed、样本量和仿真
边界。

## 不能写的结论

不能声称已实现真实 6G 系统、真实厘米级定位、标准 3GPP 信道验证或真实建筑
泛化；本项目没有相应硬件和实测证据。

## 展示建议

先用 `dashboard_normal.png` 解释数字孪生，再展示
`trajectory_comparison.png` 与 `error_cdf.png`，随后切换
`dashboard_strong_blockage.png` 和 `dashboard_anchor_failure.png`，最后用
`robustness_results.png` 与 `runtime_comparison.png` 收束结论。
