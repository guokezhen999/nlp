
# 🚀 我的机器学习与深度学习项目集

欢迎来到我的个人机器学习与深度学习项目作品集！这个仓库汇集了我近期在深度学习、自然语言处理和强化学习等领域的一些学习实践和项目。旨在展示我在构建、训练和评估复杂模型方面的技能，并体现我的代码组织和工程实践能力。

---

## 📚 目录

1.  [🛠️ `mydl` 自定义深度学习库](#-mydl-自定义深度学习库)
2.  [📧 垃圾邮件分类 (BERT + CNN)](#-垃圾邮件分类-bert--cnn)
3.  [🧠 RLHF (人类反馈强化学习) 核心组件示例](#-rlhf-人类反馈强化学习-核心组件示例)


---

## 🛠️ `mydl` 自定义深度学习库

**路径:** [`./mydl/`](./mydl/)

`mydl` 是我个人从零开始构建的一个轻量级深度学习工具包，灵感来源于 D2L.ai 的学习经验。它旨在封装常用的神经网络模块、数据处理工具和训练实用程序，以提高代码的复用性和开发效率。

这个库展示了我对深度学习基本原理的深入理解以及将理论付诸实践的能力。它包含了：

*   **`models.py`**: 定义了多种基础神经网络模型架构，例如用于情感分类的双向 LSTM (BiRNN)，以及构建 Transformer 模型所需的关键组件（如点积/多头注意力、位置编码、FFN、残差连接和层规范化），并实现了完整的 Transformer 编码器和解码器。此外，还提供了计算模型参数总数的实用函数。
*   **`nlp.py`**: 提供了一系列自然语言处理相关的辅助工具，包括文本分词、词汇表构建和管理（支持词频统计和未知词处理）、标签映射、预训练词嵌入的加载，以及文本序列截断与填充等数据预处理功能。
*   **`training.py`**: 封装了模型训练过程中的常用工具，例如用于记录和统计训练指标和时间的累加器和计时器、计算模型准确率的函数、在 Jupyter Notebook 中绘制动态图表的 `Animator`，以及通用的单设备模型训练循环。

---

## 📧 垃圾邮件分类 (BERT + CNN)

**项目路径:** [`./kaggle-projects/spam-email-classification/`](./kaggle-projects/spam-email-classification/)
**Kaggle Notebook 链接:** [https://www.kaggle.com/code/guokezhen/spam-email-classification-bert-cnn-99-46ac](https://www.kaggle.com/code/guokezhen/spam-email-classification-bert-cnn-99-46ac)

这是一个经典的机器学习项目，旨在利用先进的自然语言处理技术对电子邮件进行垃圾邮件分类。

该项目结合了强大的 **BERT 预训练模型**作为文本特征提取器，能够捕捉文本的深层语义信息。在此基础上，构建了一个 **TextCNN**（文本卷积神经网络）架构，通过多尺度的卷积核（3, 4, 5）有效捕获文本中的局部模式，从而进行高效的分类。整个模型和训练流程均使用 **PyTorch** 框架实现。

项目高效地加载、划分和预处理了包含数千封邮件的数据集。在仅仅 2 个训练周期后，模型在测试集上取得了**高达 99.2% 的分类准确率**，展示了模型卓越的性能和泛化能力，并能对新的邮件文本进行实时预测。该项目突显了我在处理真实世界文本分类问题、应用现代深度学习模型和优化模型性能方面的实践经验。

---

## 🧠 RLHF (人类反馈强化学习) 核心组件示例

**项目路径:** [`./rlhf/`](./rlhf/)

这个 Jupyter Notebook (`rlhf.ipynb`) 旨在理解和实现基于人类反馈的强化学习 (RLHF) 的核心组件和工作流程。RLHF 是训练大型语言模型 (LLMs) 以更好地与人类偏好对齐的关键技术。此项目通过 PyTorch 和 Hugging Face Transformers 库，演示了从奖励模型到策略优化的关键步骤。

项目中的核心组件包括：

1.  **奖励模型 (Reward Model - RM)：** 使用预训练的 `OpenAssistant/reward-model-deberta-v3-base` 模型，用于评估 LLM 生成文本的质量和与人类偏好的一致性，能够对 `(prompt, response)` 对进行评分。
2.  **Actor-Critic 网络 (Policy Model)：** 基于 `Qwen/Qwen3-0.6B` 预训练语言模型，通过添加 `PolicyWithValueHeadWrapper`，集成了一个 Value Head。此模型不仅能生成文本，还能预测生成文本的预期奖励。
3.  **数据收集 (Rollout)：** 演示了从 Prompt 数据集中获取批量 Prompts，并使用 Policy Model 生成 Responses 的过程，同时收集了生成过程中每个 Token 的 `log_probs` 和对应的 `values`。
4.  **广义优势估计 (Generalized Advantage Estimation - GAE)：** 实现 GAE 算法以更稳定地估计强化学习中的优势函数，用于加速训练。
5.  **PPO (Proximal Policy Optimization) 损失计算：** 实现了 PPO 算法的核心损失函数，包括策略损失、Value 损失和熵损失，并展示了如何通过这些损失进行梯度更新。
6.  **训练循环：** 模拟了 RLHF 的迭代训练过程，包括数据收集、奖励计算（包含 KL 散度惩罚）和 PPO 优化。




