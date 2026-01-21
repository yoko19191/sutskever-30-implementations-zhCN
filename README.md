# Sutskever 30 - 完整实现套件

**Ilya Sutskever 推荐的 30 篇基础论文的综合教学实现**

[![Implementations](https://img.shields.io/badge/实现-30%2F30-brightgreen)](https://github.com/yoko19191/sutskever-30-implementations-zhCN)
[![Coverage](https://img.shields.io/badge/覆盖率-100%25-blue)](https://github.com/yoko19191/sutskever-30-implementations-zhCN)
[![Python](https://img.shields.io/badge/Python-仅NumPy-yellow)](https://numpy.org/)

## 概述

本仓库包含了 Ilya Sutskever 著名阅读列表中论文的详细教学实现——他告诉 John Carmack，这个收藏将教会你深度学习中"90% 重要的内容"。

### 关于阅读列表

**出处**：这个阅读列表最初由 Ilya Sutskever（OpenAI 联合创始人兼首席科学家）推荐给 John Carmack。该列表在社区中广泛流传，被认为是深度学习领域的核心阅读材料。

**论文时间范围**：阅读列表中的论文涵盖了从 2012 年（AlexNet）到 2023 年（Lost in the Middle）的深度学习发展历程，反映了该领域在过去十余年中的关键进展。

**论文更新截止时间**：本阅读列表的论文更新截止至 **2023 年**。最新的论文是论文 30：Lost in the Middle (2023)，该论文揭示了语言模型在长上下文中的位置偏差问题。

**参考资源**：

- 原始阅读列表整理：[Ilya Sutskever 的阅读列表 (GitHub)](https://github.com/dzyim/ilya-sutskever-recommended-reading)
- 相关讨论和解读：[Aman 的 AI 期刊 - Sutskever 30 入门](https://aman.ai/primers/ai/top-30-papers/)

每个实现：

- ✅ 引用原文出处
- ✅ 仅使用 NumPy（无深度学习框架）以确保教学清晰度
- ✅ 包含合成/引导数据以便立即执行
- ✅ 提供丰富的可视化和解释
- ✅ 展示每篇论文的核心概念
- ✅ 在 Jupyter notebooks 中运行以实现交互式学习

## 快速开始

### 1. 安装 uv (如果尚未安装)

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 运行项目

```bash
# 克隆仓库
git clone git@github.com:yoko19191/sutskever-30-implementations-zhCN.git

# 进入目录
cd sutskever-30-implementations-zhCN

# 使用 uv 同步依赖
uv sync

# 运行任何 notebook
uv run jupyter notebook 02_char_rnn_karpathy.ipynb
```

## Sutskever 30 篇论文

### 基础概念（论文 1-5）

| # | 论文                                                     | Notebook                              | 核心概念                             |
| - | -------------------------------------------------------- | ------------------------------------- | ------------------------------------ |
| 1 | 复杂性动力学第一定律 (The First Law of Complexodynamics) | ✅`01_complexity_dynamics.ipynb`    | 熵 (Entropy)、复杂性增长、细胞自动机 |
| 2 | RNN 的惊人有效性                                         | ✅`02_char_rnn_karpathy.ipynb`      | 字符级模型、RNN 基础、文本生成       |
| 3 | 理解 LSTM 网络                                           | ✅`03_lstm_understanding.ipynb`     | 门控 (Gates)、长期记忆、梯度流       |
| 4 | RNN 正则化                                               | ✅`04_rnn_regularization.ipynb`     | 序列丢弃 (Dropout)、变分丢弃         |
| 5 | 保持神经网络简单                                         | ✅`05_neural_network_pruning.ipynb` | MDL 原则、权重剪枝、90%+ 稀疏性      |

### 架构与机制（论文 6-15）

| #  | 论文                                                           | Notebook                                 | 核心概念                                         |
| -- | -------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------ |
| 6  | 指针网络 (Pointer Networks)                                    | ✅`06_pointer_networks.ipynb`          | 注意力作为指针、组合优化问题                     |
| 7  | ImageNet/AlexNet                                               | ✅`07_alexnet_cnn.ipynb`               | CNN、卷积、数据增强                              |
| 8  | 顺序很重要：集合的序列到序列 (Order Matters: Seq2Seq for Sets) | ✅`08_seq2seq_for_sets.ipynb`          | 集合编码、排列不变性、注意力池化                 |
| 9  | GPipe                                                          | ✅`09_gpipe.ipynb`                     | 流水线并行、微批次、重计算                       |
| 10 | 深度残差学习 (ResNet)                                          | ✅`10_resnet_deep_residual.ipynb`      | 跳跃连接、梯度高速公路                           |
| 11 | 扩张卷积                                                       | ✅`11_dilated_convolutions.ipynb`      | 感受野、多尺度                                   |
| 12 | 神经消息传递 (GNNs)                                            | ✅`12_graph_neural_networks.ipynb`     | 图网络、消息传递                                 |
| 13 | **Attention Is All You Need**                            | ✅`13_attention_is_all_you_need.ipynb` | Transformers (Transformer)、自注意力、多头注意力 |
| 14 | 神经机器翻译                                                   | ✅`14_bahdanau_attention.ipynb`        | 序列到序列 (Seq2seq)、Bahdanau 注意力            |
| 15 | ResNet 中的恒等映射                                            | ✅`15_identity_mappings_resnet.ipynb`  | 预激活、梯度流                                   |

### 高级主题（论文 16-22）

| #  | 论文                              | Notebook                               | 核心概念                                        |
| -- | --------------------------------- | -------------------------------------- | ----------------------------------------------- |
| 16 | 关系推理 (Relational Reasoning)   | ✅`16_relational_reasoning.ipynb`    | 关系网络、成对函数                              |
| 17 | **变分有损自编码器**        | ✅`17_variational_autoencoder.ipynb` | VAE、ELBO、重参数化技巧                         |
| 18 | **关系 RNN**                | ✅`18_relational_rnn.ipynb`          | 关系记忆、多头自注意力、手动反向传播 (~1100 行) |
| 19 | 咖啡自动机 (The Coffee Automaton) | ✅`19_coffee_automaton.ipynb`        | 不可逆性、熵、时间箭头、Landauer 原理           |
| 20 | **神经图灵机**              | ✅`20_neural_turing_machine.ipynb`   | 外部记忆、可微分寻址                            |
| 21 | Deep Speech 2 (CTC)               | ✅`21_ctc_speech.ipynb`              | CTC 损失、语音识别                              |
| 22 | Scaling Law                       | ✅`22_scaling_laws.ipynb`            | 幂律、计算最优训练                              |

### 理论与元学习（论文 23-30）

| #  | 论文                                          | Notebook                                  | 核心概念                                                    |
| -- | --------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| 23 | MDL 原则                                      | ✅`23_mdl_principle.ipynb`              | 信息论、模型选择、压缩                                      |
| 24 | **机器超级智能**                        | ✅`24_machine_super_intelligence.ipynb` | 通用 AI、AIXI、Solomonoff 归纳、智能度量、自我改进          |
| 25 | Kolmogorov 复杂性                             | ✅`25_kolmogorov_complexity.ipynb`      | 压缩、算法随机性、通用先验                                  |
| 26 | **CS231n: CNN 视觉识别**                | ✅`26_cs231n_cnn_fundamentals.ipynb`    | 图像分类流程、kNN/线性/NN/CNN、反向传播、优化、调优神经网络 |
| 27 | 多令牌预测 (Multi-token Prediction)           | ✅`27_multi_token_prediction.ipynb`     | 多个未来令牌、样本效率、快 2-3 倍                           |
| 28 | 密集段落检索                                  | ✅`28_dense_passage_retrieval.ipynb`    | 双编码器、MIPS、批次内负样本                                |
| 29 | 检索增强生成 (Retrieval-Augmented Generation) | ✅`29_rag.ipynb`                        | RAG-Sequence、RAG-Token、知识检索                           |
| 30 | 迷失在中间 (Lost in the Middle)               | ✅`30_lost_in_middle.ipynb`             | 位置偏差、长上下文、U 型曲线                                |

## 精选实现

### 🌟 必读 Notebooks

这些实现涵盖了最有影响力的论文并展示了核心深度学习概念：

#### 基础

1. **`02_char_rnn_karpathy.ipynb`** - 字符级 RNN

   - 从零构建 RNN
   - 理解通过时间的反向传播
   - 生成文本
2. **`03_lstm_understanding.ipynb`** - LSTM 网络

   - 实现遗忘/输入/输出门
   - 可视化门控激活
   - 与普通 RNN 比较
3. **`04_rnn_regularization.ipynb`** - RNN 正则化

   - RNN 的变分丢弃
   - 正确的丢弃放置
   - 训练改进
4. **`05_neural_network_pruning.ipynb`** - 网络剪枝与 MDL

   - 基于幅度的剪枝
   - 迭代剪枝与微调
   - 90%+ 稀疏性且损失最小
   - 最小描述长度原则

#### 计算机视觉

5. **`07_alexnet_cnn.ipynb`** - CNN 与 AlexNet

   - 从零实现卷积层
   - 最大池化和 ReLU
   - 数据增强技术
6. **`10_resnet_deep_residual.ipynb`** - ResNet

   - 跳跃连接解决退化问题
   - 梯度流可视化
   - 恒等映射直觉
7. **`15_identity_mappings_resnet.ipynb`** - 预激活 ResNet

   - 预激活 vs 后激活
   - 更好的梯度流
   - 训练 1000+ 层网络
8. **`11_dilated_convolutions.ipynb`** - 扩张卷积

   - 多尺度感受野
   - 无需池化
   - 语义分割

#### 注意力机制与 Transformers

9. **`14_bahdanau_attention.ipynb`** - 神经机器翻译

   - 原始注意力机制
   - 带对齐的序列到序列 (Seq2seq)
   - 注意力可视化
10. **`13_attention_is_all_you_need.ipynb`** - Transformers

    - 缩放点积注意力
    - 多头注意力
    - 位置编码
    - 现代 LLM 的基础
11. **`06_pointer_networks.ipynb`** - 指针网络

    - 注意力作为选择
    - 组合优化
    - 可变输出大小
12. **`08_seq2seq_for_sets.ipynb`** - 集合的序列到序列

    - 排列不变集合编码器
    - 读取-处理-写入架构
    - 无序元素的注意力
    - 排序和集合操作
    - 比较：顺序敏感 vs 顺序不变
13. **`09_gpipe.ipynb`** - GPipe 流水线并行

    - 跨设备的模型分区
    - 流水线利用的微批次
    - 先前向后调度 (全部前向，全部反向)
    - 重计算 (梯度检查点)
    - 气泡时间分析
    - 训练超过单设备内存的模型

#### 高级主题

14. **`12_graph_neural_networks.ipynb`** - 图神经网络

    - 消息传递框架
    - 图卷积
    - 分子属性预测
15. **`16_relational_reasoning.ipynb`** - 关系网络

    - 成对关系推理
    - 视觉问答 (Visual QA)
    - 排列不变性
16. **`18_relational_rnn.ipynb`** - 关系 RNN

    - 带关系记忆的 LSTM
    - 跨记忆槽的多头自注意力
    - 架构演示（前向传播）
    - 序列推理任务
    - **第 11 节：手动反向传播实现 (~1100 行)**
    - 所有组件的完整梯度计算
    - 数值验证的梯度检查
17. **`20_neural_turing_machine.ipynb`** - 记忆增强网络

    - 内容和位置寻址
    - 可微分读/写
    - 外部记忆
18. **`21_ctc_speech.ipynb`** - CTC 损失与语音识别

    - 联结时序分类
    - 无对齐训练
    - 前向算法

#### 生成模型

19. **`17_variational_autoencoder.ipynb`** - VAE
    - 生成建模
    - ELBO 损失
    - 潜空间可视化

#### 现代应用

20. **`27_multi_token_prediction.ipynb`** - 多令牌预测

    - 预测多个未来令牌
    - 2-3 倍样本效率
    - 投机解码
    - 更快的训练和推理
21. **`28_dense_passage_retrieval.ipynb`** - 密集检索

    - 双编码器架构
    - 批次内负样本
    - 语义搜索
22. **`29_rag.ipynb`** - 检索增强生成

    - RAG-Sequence vs RAG-Token
    - 结合检索 + 生成
    - 知识驱动输出
23. **`30_lost_in_middle.ipynb`** - 长上下文分析

    - LLM 中的位置偏差
    - U 型性能曲线
    - 文档排序策略

#### 缩放与理论

24. **`22_scaling_laws.ipynb`** - 缩放定律

    - 幂律关系
    - 计算最优训练
    - 性能预测
25. **`23_mdl_principle.ipynb`** - 最小描述长度

    - 信息论模型选择
    - 压缩 = 理解
    - MDL vs AIC/BIC 比较
    - 神经网络架构选择
    - 基于 MDL 的剪枝（连接到论文 5）
    - Kolmogorov 复杂性预览
26. **`25_kolmogorov_complexity.ipynb`** - Kolmogorov 复杂性

    - K(x) = 生成 x 的最短程序
    - 随机性 = 不可压缩性
    - 算法概率 (Solomonoff)
    - 归纳的通用先验
    - 与 Shannon 熵的连接
    - 奥卡姆剃刀的形式化
    - ML 的理论基础
27. **`24_machine_super_intelligence.ipynb`** - 通用人工智能

    - **智能的形式理论 (Legg & Hutter)**
    - 心理测量 g 因子和通用智能 Υ(π)
    - 序列预测的 Solomonoff 归纳
    - AIXI：理论最优的 RL 智能体
    - 蒙特卡洛 AIXI (MC-AIXI) 近似
    - Kolmogorov 复杂性估计
    - 跨环境的智能测量
    - 递归自我改进动态
    - 智能爆炸场景
    - **6 个章节：从心理测量到超级智能**
    - 连接论文 #23 (MDL)、#25 (Kolmogorov)、#8 (DQN)
28. **`01_complexity_dynamics.ipynb`** - 复杂性与熵

    - 细胞自动机 (Rule 30)
    - 熵增长
    - 不可逆性（基础介绍）
29. **`19_coffee_automaton.ipynb`** - 咖啡自动机（深度探索）

    - **不可逆性的全面探索**
    - 咖啡混合和扩散过程
    - 熵增长和粗粒化
    - 相空间和 Liouville 定理
    - Poincaré 回归定理（e^N 时间后会重新混合！）
    - Maxwell 妖和 Landauer 原理
    - 计算不可逆性（单向函数、哈希）
    - 机器学习中的信息瓶颈
    - 生物不可逆性（生命和第二定律）
    - 时间箭头：基本 vs 涌现
    - **10 个全面章节探索所有尺度的不可逆性**
30. **`26_cs231n_cnn_fundamentals.ipynb`** - CS231n：从第一性原理的视觉

    - **纯 NumPy 的完整视觉流程**
    - k 近邻基线
    - 线性分类器 (SVM 和 Softmax)
    - 优化 (SGD、Momentum、Adam、学习率调度)
    - 带反向传播的 2 层神经网络
    - 卷积层 (conv、pool、ReLU)
    - 完整的 CNN 架构 (Mini-AlexNet)
    - 可视化技术（滤波器、显著性图）
    - 迁移学习原则
    - 调优技巧（健全性检查、超参数调优、监控）
    - **10 个章节覆盖整个 CS231n 课程**
    - 连接论文 #7 (AlexNet)、#10 (ResNet)、#11 (扩张卷积)

## 仓库结构

```
sutskever-30-implementations-zhCN/
├── README.md                           # 本文件
├── PROGRESS.md                         # 实现进度跟踪
├── IMPLEMENTATION_TRACKS.md            # 所有 30 篇论文的详细跟踪
│
├── 01_complexity_dynamics.ipynb        # 熵与复杂性
├── 02_char_rnn_karpathy.ipynb         # 基础 RNN
├── 03_lstm_understanding.ipynb         # LSTM 门控
├── 04_rnn_regularization.ipynb         # RNN 的丢弃
├── 05_neural_network_pruning.ipynb     # 剪枝与 MDL
├── 06_pointer_networks.ipynb           # 注意力指针
├── 07_alexnet_cnn.ipynb               # CNN 与 AlexNet
├── 08_seq2seq_for_sets.ipynb          # 排列不变集合
├── 09_gpipe.ipynb                     # 流水线并行
├── 10_resnet_deep_residual.ipynb      # 残差连接
├── 11_dilated_convolutions.ipynb       # 多尺度卷积
├── 12_graph_neural_networks.ipynb      # 消息传递 GNN
├── 13_attention_is_all_you_need.ipynb # Transformer 架构
├── 14_bahdanau_attention.ipynb         # 原始注意力
├── 15_identity_mappings_resnet.ipynb   # 预激活 ResNet
├── 16_relational_reasoning.ipynb       # 关系网络
├── 17_variational_autoencoder.ipynb   # VAE
├── 18_relational_rnn.ipynb             # 关系 RNN
├── 19_coffee_automaton.ipynb           # 不可逆性深度探索
├── 20_neural_turing_machine.ipynb     # 外部记忆
├── 21_ctc_speech.ipynb                # CTC 损失
├── 22_scaling_laws.ipynb              # 经验缩放
├── 23_mdl_principle.ipynb             # MDL 与压缩
├── 24_machine_super_intelligence.ipynb # 通用 AI 与 AIXI
├── 25_kolmogorov_complexity.ipynb     # K(x) 与随机性
├── 26_cs231n_cnn_fundamentals.ipynb    # 从第一性原理的视觉
├── 27_multi_token_prediction.ipynb     # 多令牌预测
├── 28_dense_passage_retrieval.ipynb    # 密集检索
├── 29_rag.ipynb                       # RAG 架构
└── 30_lost_in_middle.ipynb            # 长上下文分析
```

**所有 30 篇论文已实现！（100% 完成！）🎉**

## 学习路径

### 初学者路径（从这里开始！）

1. **字符级 RNN** (`02_char_rnn_karpathy.ipynb`) - 学习基础 RNN
2. **LSTM** (`03_lstm_understanding.ipynb`) - 理解门控机制
3. **CNN** (`07_alexnet_cnn.ipynb`) - 计算机视觉基础
4. **ResNet** (`10_resnet_deep_residual.ipynb`) - 跳跃连接
5. **VAE** (`17_variational_autoencoder.ipynb`) - 生成模型

### 中级路径

6. **RNN 正则化** (`04_rnn_regularization.ipynb`) - 更好的训练
7. **Bahdanau 注意力** (`14_bahdanau_attention.ipynb`) - 注意力基础
8. **指针网络** (`06_pointer_networks.ipynb`) - 注意力作为选择
9. **集合的序列到序列** (`08_seq2seq_for_sets.ipynb`) - 排列不变性
10. **CS231n** (`26_cs231n_cnn_fundamentals.ipynb`) - 完整视觉流程（kNN → CNN）
11. **GPipe** (`09_gpipe.ipynb`) - 大模型的流水线并行
12. **Transformer** (`13_attention_is_all_you_need.ipynb`) - 现代架构
13. **扩张卷积** (`11_dilated_convolutions.ipynb`) - 感受野
14. **缩放定律** (`22_scaling_laws.ipynb`) - 理解缩放

### 高级路径

15. **预激活 ResNet** (`15_identity_mappings_resnet.ipynb`) - 架构细节
16. **图神经网络** (`12_graph_neural_networks.ipynb`) - 图学习
17. **关系网络** (`16_relational_reasoning.ipynb`) - 关系推理
18. **神经图灵机** (`20_neural_turing_machine.ipynb`) - 外部记忆
19. **CTC 损失** (`21_ctc_speech.ipynb`) - 语音识别
20. **密集检索** (`28_dense_passage_retrieval.ipynb`) - 语义搜索
21. **RAG** (`29_rag.ipynb`) - 检索增强生成
22. **迷失在中间** (`30_lost_in_middle.ipynb`) - 长上下文分析

### 理论与基础

23. **MDL 原则** (`23_mdl_principle.ipynb`) - 通过压缩进行模型选择
24. **Kolmogorov 复杂性** (`25_kolmogorov_complexity.ipynb`) - 随机性与信息
25. **复杂性动力学** (`01_complexity_dynamics.ipynb`) - 熵与涌现
26. **咖啡自动机** (`19_coffee_automaton.ipynb`) - 不可逆性深度探索

## Sutskever 30 的关键洞察

### 架构演进

- **RNN → LSTM**：门控解决梯度消失问题
- **普通网络 → ResNet**：跳跃连接实现深度
- **RNN → Transformer**：注意力实现并行化
- **固定词汇表 → 指针**：输出可以引用输入

### 基本机制

- **注意力**：可微分的选择机制
- **残差连接**：梯度高速公路
- **门控**：学习的信息流控制
- **外部记忆**：将存储与计算分离

### 训练洞察

- **缩放定律**：性能随规模可预测地提升
- **正则化**：丢弃、权重衰减、数据增强
- **优化**：梯度裁剪、学习率调度
- **计算最优**：平衡模型大小和训练数据

### 理论基础

- **信息论**：压缩、熵、MDL
- **复杂性**：Kolmogorov 复杂性、幂律
- **生成建模**：VAE、ELBO、潜在空间
- **记忆**：可微分数据结构

## 实现理念

### 为什么仅使用 NumPy？

这些实现有意避免使用 PyTorch/TensorFlow 以：

- **加深理解**：看清楚框架抽象的内容
- **教学清晰度**：无魔法，每项操作都显式
- **核心概念**：专注于算法而非框架 API
- **可迁移知识**：原则适用于任何框架

### 合成数据方法

每个 notebook 生成自己的数据以：

- **立即执行**：无需下载数据集
- **受控实验**：理解简单情况下的行为
- **概念专注**：数据不会模糊算法
- **快速迭代**：立即修改和重新运行

## 扩展与后续步骤

### 基于这些实现构建

理解核心概念后，尝试：

1. **扩展**：在 PyTorch/JAX 中实现以处理真实数据集
2. **组合技术**：例如，ResNet + Attention
3. **现代变体**：
   - RNN → GRU → Transformer
   - VAE → β-VAE → VQ-VAE
   - ResNet → ResNeXt → EfficientNet
4. **应用**：应用于实际问题

### 研究方向

Sutskever 30 指向：

- 缩放（更大的模型、更多数据）
- 效率（稀疏模型、量化）
- 能力（推理、多模态）
- 理解（可解释性、理论）

## 资源

### 原始论文

完整引用和链接见 `IMPLEMENTATION_TRACKS.md`

### 延伸阅读

- [Ilya Sutskever 的阅读列表 (GitHub)](https://github.com/dzyim/ilya-sutskever-recommended-reading)
- [Aman 的 AI 期刊 - Sutskever 30 入门](https://aman.ai/primers/ai/top-30-papers/)
- [带注释的 Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Andrej Karpathy 的博客](http://karpathy.github.io/)

### 课程

- Stanford CS231n：卷积神经网络
- Stanford CS224n：深度学习自然语言处理
- MIT 6.S191：深度学习导论

## 贡献

这些实现是教育性的，可以改进！考虑：

- 添加更多可视化
- 实现缺失的论文
- 改进解释
- 发现错误
- 添加与框架实现的比较

## 许可证

教育用途。原始研究引用请参阅各论文。

## 致谢

- **Ilya Sutskever**：策划了这个基本阅读列表
- **论文作者**：他们的基础性贡献
- **社区**：让这些想法变得易于理解

---

## 最新添加（2025 年 12 月）

### 最近实现（21 篇新论文！）

- ✅ **论文 4**：RNN 正则化（变分丢弃）
- ✅ **论文 5**：神经网络剪枝（MDL、90%+ 稀疏性）
- ✅ **论文 7**：AlexNet（从零实现 CNN）
- ✅ **论文 8**：集合的序列到序列（排列不变性、注意力池化）
- ✅ **论文 9**：GPipe（流水线并行、微批次、重计算）
- ✅ **论文 19**：咖啡自动机（深入探讨不可逆性、熵、Landauer 原理）
- ✅ **论文 26**：CS231n（完整视觉流程：kNN → CNN，全部使用 NumPy）
- ✅ **论文 11**：扩张卷积（多尺度）
- ✅ **论文 12**：图神经网络（消息传递）
- ✅ **论文 14**：Bahdanau 注意力（原始注意力）
- ✅ **论文 15**：ResNet 恒等映射（预激活）
- ✅ **论文 16**：关系推理（关系网络）
- ✅ **论文 18**：关系 RNN（关系记忆 + 第 11 节：手动反向传播 ~1100 行）
- ✅ **论文 21**：Deep Speech 2（CTC 损失）
- ✅ **论文 23**：MDL 原则（压缩、模型选择，连接到论文 5 和 25）
- ✅ **论文 24**：机器超级智能（通用 AI、AIXI、Solomonoff 归纳、智能度量、递归自我改进）
- ✅ **论文 25**：Kolmogorov 复杂性（随机性、算法概率、理论基础）
- ✅ **论文 27**：多令牌预测（2-3 倍样本效率）
- ✅ **论文 28**：密集段落检索（双编码器）
- ✅ **论文 29**：RAG（检索增强生成）
- ✅ **论文 30**：迷失在中间（长上下文）

## 快速参考：实现复杂度

### 可以在一个下午实现

- ✅ 字符级 RNN
- ✅ LSTM
- ✅ ResNet
- ✅ 简单的 VAE
- ✅ 扩张卷积

### 周末项目

- ✅ Transformer
- ✅ 指针网络
- ✅ 图神经网络
- ✅ 关系网络
- ✅ 神经图灵机
- ✅ CTC 损失
- ✅ 密集检索

### 一周深度探索

- ✅ 完整的 RAG 系统
- ⚠️ 大规模实验
- ⚠️ 超参数优化


# 分享：第一人称读论文 Prompt

我用这个 Prompt 帮助我读了很多篇文章，希望对你也有帮助。

使用方法就是在任意 Chat 界面上传论文，然后贴上这段 Prompt 

```plaintext
请你根据这篇文章，请你完全代入研究团队的视角。用第一人称的方法解释这篇文章的逻辑。研究的背景是什么？核心痛点是什么？这一步需要做什么？为什么要这么做？下一步又应该做什么？需要考虑到研发团队当时可能有的疑问和解决思路。输出一篇第一人称的文章，文章包含整篇文章的所有细节。
```


---

**"如果你真的学会了所有这些，你将了解今天 90% 重要的内容。"** - Ilya Sutskever

祝学习愉快！🚀
