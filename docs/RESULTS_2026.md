# 2026 方法：实际复现结果

执行日期：2026-11-03（Asia/Singapore）。两个正式作业均已完成，CQ 验证输出后完整取回，本地再次从逐笔分数复算指标通过。以下数值为实际运行结果。

## 数据与协议

沿用原有 ULB/Worldline 50,000 条均匀样本；去重后 49,961 条、88 条欺诈。数据逻辑哈希与经典方法实验完全一致。按时间约 60/20/20 切分，源分段为 29,976 / 9,991 / 9,994 条，欺诈数为 60 / 10 / 18。Time 只用于排序和切分。

EBM 使用分段中的全部交易；STTN-CP 每段独立建立 8 步窗口，端点数为 29,969 / 9,984 / 9,987，欺诈数仍为 60 / 10 / 18。窗口是均匀样本内相邻的匿名全局交易。所有预处理、模型搜索与特征排序都使用训练集；最终报警阈值仅用验证集选择。

## 新增方法正式指标

| 方法 | 测试交易数 | AP | 梯形 PR-AUC | ROC-AUC | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Optimized EBM 2026 | 9994 | 0.7583 | 0.7571 | 0.9499 | 0.9167 | 0.6111 | 0.7333 | 11 / 1 / 7 |
| STTN-CP 2026 | 9987 | 0.7105 | 0.7089 | 0.8600 | 0.9167 | 0.6111 | 0.7333 | 11 / 1 / 7 |

- STTN-CP：859,330 个参数、完整宽度 128 / 3 个时空块。训练 25 轮后按既定耐心值 10 早停，最佳 checkpoint 为第 15 轮，验证 F1@0.5=0.8。选定 checkpoint 后，再用验证集选择最终阈值 0.8276261091。训练与评分约 24 分钟。
- Optimized EBM：25 个 L25 缩放候选、9 个 L9 参数候选，每个候选 3 折，共 102 次 CV 拟合，加特征排序和重新训练共 104 次拟合。训练与评分约 31 分钟。
- 本次 EBM 内层 CV 选择：MinMax → Standard → Yeo–Johnson Power → Normal Quantile；interactions=0、max_bins=256、learning_rate=0.1、max_rounds=50。0 个交互是本次训练内搜索的实际选择，论文报告的 20 个交互配置也在候选中。
- 训练集选中的前 18 个特征：`V12, V14, V4, V16, V13, V3, V7, V8, V11, V9, V6, V24, V19, V15, V21, V1, V10, V20`。它们不是通过测试集排序得到。

EBM 的原始 OA 行与水平表未公开，本仓库重建这些细节；STTN 使用注明版本的作者维护版及公共数据协议。两者的完整来源、补全规则和与论文实验的差异见 [方法说明](METHODS_2026.md)。

## 与 10 个经典方法对齐

下表仅使用所有方法共同覆盖的 **9,987 笔测试交易、18 笔欺诈**。行号、标签、数据哈希、seed 和分段模式均已检查一致；每个方法保留原验证集冻结的阈值，没有用测试集重新调参。

| 方法 | AP | ROC-AUC | F1 |
|---|---:|---:|---:|
| logistic_regression | 0.6641 | 0.9642 | 0.6857 |
| gaussian_nb | 0.0832 | 0.9264 | 0.1832 |
| knn | 0.7782 | 0.8888 | 0.8485 |
| svm_rbf | 0.7809 | 0.9491 | 0.8000 |
| decision_tree | 0.5288 | 0.7732 | 0.6286 |
| random_forest | 0.7658 | 0.8815 | 0.8387 |
| adaboost | 0.5846 | 0.9603 | 0.5600 |
| gradient_boosting | 0.4314 | 0.7884 | 0.5333 |
| xgboost | 0.7626 | 0.9816 | 0.7742 |
| mlp | 0.7686 | 0.9445 | 0.7879 |
| optimized_ebm_2026 | 0.7583 | 0.9499 | 0.7333 |
| sttn_cp_2026 | 0.7105 | 0.8600 | 0.7333 |

这只是一个数据集、一个 seed 和一个时间划分的描述性比较。验证集只有 10 笔欺诈，测试集只有 18 笔，无法据此给出普遍优劣或统计显著性结论；也不能将这些数值冒称为论文原表格的复现值。

## 可复核产物

- [EBM 正式结果、权重和搜索记录](../results/recent2026/optimized_ebm_2026/)。
- [STTN 正式结果、权重和 25 轮记录](../results/recent2026/sttn_cp_2026/)。
- [共同测试交易上的 12 方法指标](../results/recent2026/aligned_comparison/metrics.csv)与 [对齐检查](../results/recent2026/aligned_comparison/alignment.json)。
- [CQ 作业与固定代码版本](../results/recent2026/CQ_RUNS.json)。正式训练均固定 `e761d4e376a550127c09292237509a20f9d6d10e`，远端 Python 3.13.16，torch 2.8.0、interpret-core 0.7.8、sklearn 1.7.2；各目录附完整环境记录。
- 两个方法均通过独立指标复算、验证阈值最优性、预测端点与标签对照、时间边界及集合隔离检查。原有经典方法亦通过加强后的审计。
- 正式权重在 macOS CPU 上另抽查 16 笔（8 欺诈、8 个高分正常样本）：EBM 分数完全一致；STTN 最大分数差异约 0.00194，冻结阈值下 16 笔分类均一致。该抽查不等于完整跨设备等价证明，正式指标仍以原始 CQ 分数为准。[检查记录](../results/recent2026/formal_checkpoint_reload.json)。
- 本地完整测试 11 项通过；Linux CI 的经典协议和 2026 方法测试均通过。原始作者架构 Git blob 校验一致，许可证和 NOTICE 已保留。
- 使用 CQ 管理的免费 Kaggle CPU/GPU，未租赁付费实例。Kaggle 结果与模型已完整取回，并随本仓库保存。

## 直接使用已训练权重

```bash
python -m pip install -r requirements-2026.txt
python -m pip install --no-deps -e .
mkdir -p runs
python -m financial_supervised_ad.recent2026.predict \
  --model-dir results/recent2026/sttn_cp_2026 \
  --csv data/new_transactions.csv --output runs/sttn_predictions.csv
```

EBM 将 model-dir 改为 `results/recent2026/optimized_ebm_2026`。CSV 需要同名的 V1–V28 和 Amount 特征；STTN 还需要 Time，并对最初 7 条标记为历史不足。无需 Class 标签。训练、预测、独立审计与同样本比较命令详见方法说明和 scripts。
