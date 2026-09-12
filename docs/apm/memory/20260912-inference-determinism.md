---
date: 2026-09-12 15:00
title: 推理确定性：eval 模式确定、MC dropout 故意不确定，打分需固定种子
keywords: 确定性, 可复现, MC dropout, 随机种子, cuDNN, batch size, u_threshold, 边界翻转
abstract: 同一张图每次预测是否相同取决于模式：标准 eval（dropout 关、随机增强关）下模型是确定性函数，同一权重+同一硬件配置输出逐位相同，但需注意 GPU 非确定性算法（cudnn.deterministic）与 batch size 差异两个小坑；MC dropout 模式每次前向都不同（这正是认知不确定性的来源，取 T 次均值仍是抽样估计）。打分协议：MC dropout 前固定随机种子保证可复现，T 取够大降方差，否则边界图跨 run 翻标签；checkpoint meta 的 u_threshold/percentiles 依赖可复现打分。
---

user:
对于同一张图片，模型每次预测都是相同的吗？

assistant:
取决于推理模式，两种情形都要说清：
- 标准 eval 模式：是。model.eval() 关闭 dropout（用缩放恒等替代）、BN 用运行统计量，前向传播是权重与输入的确定性函数。同一权重 + 同一预处理 + 同一硬件/配置 → 输出逐位相同。两个小坑：①GPU 默认可能选非确定性 cuDNN 算法，需 torch.backends.cudnn.deterministic=True 且 benchmark=False；②不同 batch size 走不同 kernel/归约顺序，结果可能有微小数值差（~1e-6 级），对阈值附近的 tag 判定理论上有影响。跨硬件（GPU vs CPU、不同架构）不保证逐位一致。
- MC dropout 模式：否，且是故意的。rank 代码已有 MC dropout：测试时保持 dropout 激活，每次前向采样不同的子网络，T 次前向取均值/方差——均值是认知不确定性的估计（模型不确定性），方差本身就是产出物。同图两次打分，均值会有蒙特卡洛波动，波动大小 ≈ 不确定性大小，正好可以当作置信信号。
- 对打分协议的影响：u_threshold/percentiles 进 checkpoint meta 的约定依赖可复现打分——MC dropout 前对每张图固定随机种子（同图同种子）即可做到跨 run 可复现；T 取 20~50 控制方差；概率贴着阈值的边界图天然可能跨 run 翻标签，这类图应结合 MC 方差标记为"不确定"而不是硬判。
