# 2026 年方法扩展

核验日期：2026-11-03（Asia/Singapore）。新增两个有 2026 年公开论文依据的方法。复现包含可运行训练、冻结验证阈值、测试逐笔预测、权重保存和独立指标审计。论文数值不作为本仓库实测结果。

## 1. STTN-CP

Jayabalan & Radhakrishnan，2026-04-15：[论文及日期](https://zenodo.org/records/19593993)，[作者代码](https://github.com/kathiresan-jayabalan/sttn-cp-ccfd)。采用固定作者版本 `e5d3657d826ca2e42d6b2489e0577afc45f773df` 的模型与 InfoNCE 实现，原文件完整保留，仅添加来源注释；[Apache-2.0 许可证和 NOTICE](../third_party/sttn_cp/) 一并保留。

模型在交易特征维度及时间维度依次做注意力，使用分类损失与两种噪声视图的对比损失。正式设置沿用该作者版本：窗口 8、宽度 128、3 个时空块、4 个头、FFN 256、投影 64、batch 128、Adam 0.001、对比温度 0.07、损失权重 0.5；最多 50 epoch，验证 F1@0.5 连续 10 轮不提升则停止。

复现边界：作者维护版与其早期论文实验已有差别；本仓库复现的是注明版本的维护版架构和联合训练目标，不宣称重现原论文的 SMOTE 实验数值。本地适配进一步采用公共 60/20/20 时间划分、去除重复行、排除 Time 特征，并在选定 checkpoint 后用验证集选择最终 F1 阈值。两者的差异都有据可查。

## 2. Optimized EBM

Fazel、Bakhtiary、Bigdeli，2026-02-06：[论文 arXiv:2602.06955](https://arxiv.org/abs/2602.06955)。使用 [InterpretML 官方 EBM](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html) 实现加性主效应与交互项，并重建论文的缩放搜索、参数搜索和前 18 个特征筛选流程。

初始模型采用论文表 3 配置：20 个交互、256 bins、学习率 0.05、100 rounds。后续在训练集内部用 L25 搜索缩放流程，再用 L9 搜索参数，训练模型给特征排序，取前 18 个重新训练。外部验证/测试标签均不参与这些搜索和排名。

**公开资料缺口：**论文未公开原始 OA 行、完整三水平参数表和最终缩放顺序。本仓库没有将补全内容冒称为作者原配置。下列是明确披露、可重复的重建选择：

| 项目 | 本仓库补全规则 |
|---|---|
| L25 | GF(5) 生成的 25 行、5 列、强度 2 的正交表；5 列代表缩放位置 |
| 缩放水平 | MinMax、Standard、Yeo–Johnson Power、Normal Quantile、Robust；同一行的重复缩放器视为恒等步骤并跳过 |
| L9 | GF(3) 生成的 9 行、4 列、强度 2 的正交表 |
| interactions | 0 / 10 / 20 |
| max_bins | 64 / 128 / 256 |
| learning_rate | 0.01 / 0.05 / 0.1 |
| max_rounds | 100 / 50 / 200 |
| 内层选择 | 训练集内 3 折分层 CV 的平均 ROC-AUC；并列保留第一项 |
| 特征重要性 | 训练模型主效应重要性，加上等分给参与特征的交互重要性；取前 18 |
| 其他 EBM 参数 | `interpret-core==0.7.8` 默认值；正式模式 outer_bags=14 |

因此这是公开方法流程的独立重建，不是缺失作者代码的逐字复刻。L9 包含论文表 3 的报告配置，但数据与划分变化后它不一定被选中。全部候选、各折结果、最终参数和特征列表均保存。

## 公共评估与可比性

- 沿用经典实验的 seed=42、ULB 标签无关 50k 均匀样本、去重及时间切分，数值预处理仅拟合训练集。
- EBM 使用 29 个原始候选特征，再筛到 18；STTN-CP 使用全部 29 个候选特征。
- STTN 每个集合独立建窗口，丢弃该集合最初 7 个无足够历史的端点，不填入未来数据或跨集合历史。它与逐笔经典方法的测试行数不同，原始指标不得直接当作完全同一测试样本的排名；比较时应按保存的 row_id 对齐。
- 窗口是 50k 均匀样本内的相邻全局交易，不是完整原始交易流的连续 8 笔，也不能解释为同一账户的交易序列；ULB 没有可用的账户标识。
- 最终阈值统一由验证集最大 F1 确定；逐笔分数可通过 `scripts/audit_results.py` 独立复核。任何训练过程都不以测试集指标选 checkpoint。
- 只验证一个稀疏欺诈数据集、一个 seed；不能据此声称复现论文中的数值或普遍优于其他方法。

## 运行

Python 3.12 环境中安装可选依赖，不改变经典方法运行入口：

```bash
python -m pip install -r requirements-2026.txt
python -m pip install --no-deps -e .

# 离线 CPU 烟雾测试（缩小模型与搜索，不代表正式实验）
financial-ad-2026 --method optimized_ebm_2026 --quick --output runs/ebm_smoke
financial-ad-2026 --method sttn_cp_2026 --quick --device cpu --output runs/sttn_smoke

# 正式参数
financial-ad-2026 --method optimized_ebm_2026 --csv data/creditcard_50k.csv.gz \
  --seed 42 --n-jobs 4 --output runs/ebm_2026
financial-ad-2026 --method sttn_cp_2026 --csv data/creditcard_50k.csv.gz \
  --seed 42 --n-jobs 4 --device cuda --output runs/sttn_2026

python scripts/audit_results.py runs/ebm_2026
python scripts/audit_results.py runs/sttn_2026

# 载入已训练模型，使用冻结阈值预测；仅载入可信权重
python -m financial_supervised_ad.recent2026.predict --model-dir runs/sttn_2026 \
  --csv data/new_transactions.csv --output runs/sttn_predictions.csv
```

纯 CPU Linux 环境可先从 PyTorch 的 CPU wheel 源安装 `torch==2.8.0`，然后安装其余依赖。GPU 或较长训练按实验室要求通过 CQ 调度。Kaggle 路径使用 `jobs/kaggle_2026.sh`，所有输出写入 `CQ_OUTPUT_DIR`；每次运行记录环境版本与固定源代码 commit。

STTN 预测需要 Time 列排序，每个独立输入文件最初 7 条输出 `eligible_endpoint=false` 且预测为空；其余行输出恢复到输入行顺序。EBM 预测不需要时间列。真实交易特征必须与训练时一致，目标标签不作为输入。

除公共 metrics / predictions / splits / audit 外，STTN 输出 `best_model.pt`、`preprocessing.joblib`、`history.jsonl`；EBM 输出 `model.joblib`、`search_history.json`；两者均输出 `inference.json` 中的特征 schema 与阈值。

GPU 烟雾测试权重在本机 CPU 上抽查 16 笔，冻结阈值下分类一致，分数最大差异约 `3.02e-6`；不宣称跨设备逐位相同。正式评估以保存的原始预测分数为准。[验证记录](../results/recent2026/qualification/cuda_to_cpu_checkpoint.json)
