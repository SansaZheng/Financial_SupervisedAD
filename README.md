# Financial Supervised Anomaly Detection

金融交易监督异常检测的 **10 个经典方法复现**：Logistic Regression、Gaussian Naive Bayes、kNN、RBF SVM、CART Decision Tree、Random Forest、AdaBoost、Gradient Boosting、XGBoost、MLP。

任务：使用带标签的交易训练二分类器，`0=正常，1=欺诈/异常`。提供统一训练、验证阈值选择、测试评估、模型保存及预测入口。基于 scikit-learn / XGBoost 参考实现，复现经典方法与统一评估流程；不宣称逐项重现某篇论文的结果。

方法检索、原始文献、原理、参数及实验边界见 [方法说明](docs/METHODS.md)。实际执行结果见 [实验报告](docs/RESULTS.md)。

新增 **2026 年方法扩展**：[STTN-CP 与 Optimized EBM](docs/METHODS_2026.md)。两者有独立的 `financial-ad-2026` 入口和可选依赖，保留原有 10 个经典方法的用法。

## 安装

建议 Python 3.12。已在 macOS ARM64 / Python 3.12 验证；Linux CI 配置见工作流。

```bash
git clone https://github.com/SansaZheng/Financial_SupervisedAD.git
cd Financial_SupervisedAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
pytest -q
```

`requirements.txt` 固定直接科学计算依赖；实际安装的全部版本另存于 `results/environment-freeze.txt`。安装 XGBoost 若提示缺少 OpenMP，需要按 [官方安装文档](https://xgboost.readthedocs.io/en/stable/install.html)配置本机 OpenMP 运行库。

本次 macOS 环境未安装 Homebrew，验证时使用了 scikit-learn wheel 自带的兼容 OpenMP 库。在同类环境中，激活虚拟环境后可先设置：

```bash
export DYLD_LIBRARY_PATH="$(python -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

这只影响当前 shell，不修改系统库。Linux 环境无需此设置。

## 离线验证：全部 10 个方法

```bash
financial-ad --n-samples 5000 --quick --seed 42 --n-jobs 2 \
  --output runs/synthetic_quick_seed42
```

无 `--csv` 时明确使用 `make_classification` 合成数据，仅验证软件和评估流程，不能视为金融实证结果。`--quick` 减少树数量和 MLP 迭代上限；正式实验请去掉。

## 公开真实交易数据

ULB/Worldline 数据含 284,807 笔交易和 492 笔欺诈。下载入口为 [TensorFlow 官方教程](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)使用的公开镜像，不需 Kaggle 凭据。

```bash
# 读取完整远端流，仅保存标签无关的均匀 50k 样本及校验元数据。
python scripts/download_creditcard.py \
  --output data/creditcard_50k.csv.gz --sample-size 50000 --seed 42

# 全部 10 个方法，正式参数，按时间约 60/20/20 留出。
financial-ad --csv data/creditcard_50k.csv.gz --split temporal \
  --seed 42 --n-jobs 2 --output runs/creditcard_50k_temporal_seed42
```

不会上传原始交易数据。完整数据模式、数据限制及自有 CSV 格式见 [data/README.md](data/README.md)。大规模数据与长时间实验应放在授权的 `/data` 或 `/date` 工作区，按实验室要求通过 CQ 调度；本项目不自动申请付费云资源。

## 单独运行、权重及保存模型

```bash
financial-ad --csv data/creditcard_50k.csv.gz --methods random_forest xgboost \
  --balance balanced --threshold fpr --max-fpr 0.01 --save-models \
  --output runs/weighted_rf_xgb

python scripts/predict.py --model runs/weighted_rf_xgb/random_forest.joblib \
  --csv data/new_transactions.csv --output runs/new_predictions.csv
```

预测 CSV 需要训练时的全部特征列，不需要标签或时间列；额外列忽略。只载入可信的 joblib 文件。新输出路径必须不存在。

自有数据示例：

```bash
financial-ad --csv /data/transactions.csv --label is_fraud --time-column timestamp \
  --drop-columns transaction_id account_id investigation_outcome \
  --split temporal --output /data/financial-ad/run01
```

## 结果文件

每次运行保存：

- `config.json`：参数、数据哈希、数据来源、切分规模、依赖版本、代码 commit。
- `splits.csv`：清理后样本索引、标签、所属集合及可用的时间列。
- `metrics.csv`：全部方法的独立测试指标，包含 AP 和单独定义的梯形 PR-AUC。
- `<method>.json`：完整模型参数、阈值、指标和收敛警告。
- `<method>_predictions.csv.gz`：验证集和测试集逐笔分数、标签、预测，可重算全部指标。
- `<method>.joblib`：仅在指定 `--save-models` 时保存训练后 pipeline、特征 schema 及冻结阈值。
- `summary.json` / `progress.txt`：完成状态与运行进度。

可独立从逐笔分数复算指标与验证集阈值：

```bash
python scripts/audit_results.py runs/creditcard_50k_temporal_seed42
```

非空输出目录拒绝覆写。固定参数在测试之前定义，阈值只用验证标签选择。测试 AP 排名仅为描述性结果；不要据此反复调参。

`SVM` 输出 margin，其余方法输出正类概率。类别不平衡下主要观察 AP/PR 曲线与实际误报成本；Accuracy 不是本项目的主指标。
