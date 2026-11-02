# 10 个经典监督方法：检索依据与复现规格

检索日期：2026-11-03（Asia/Singapore）。这里将“金融交易监督异常检测”解释为：利用带标签交易学习二分类，正常为 0，欺诈/异常为 1。异常分数越大表示越可疑。它不等同于金融时间序列预测残差检测。

## 选择依据

- Niu, Wang & Yang (2019), [A Comparison Study of Credit Card Fraud Detection: Supervised versus Unsupervised](https://arxiv.org/abs/1904.10604)：研究中的监督组明确包含 LR、kNN、SVM、DT、RF、XGBoost。该研究采用五折交叉验证，本仓库采用独立验证集和时间留出，因此不能直接比较论文数值。
- [Credit Card Fraud Detection Using Machine Learning: A Comparative Study of Ensemble Learning Algorithms](https://ieeexplore.ieee.org/abstract/document/10335075) (2023), DOI 10.1109/ICSCC59169.2023.10335075：将 NB、AdaBoost 和 Gradient Boosting 纳入信用卡欺诈比较。
- [A machine learning based credit card fraud detection using the GA algorithm for feature selection](https://link.springer.com/article/10.1186/s40537-022-00573-8) (2022)：包括 ANN 与多种经典分类器。本仓库只实现监督 MLP 基线，不复现其 GA 特征选择流程。
- [TensorFlow 官方不平衡分类教程](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)：提供 ULB 数据公开镜像，说明极低欺诈率下训练/验证/测试、类别权重和阈值的重要性。本仓库不使用 TensorFlow，MLP 由 scikit-learn 实现。

“复现”范围是使用维护中的参考库实现这 10 类方法，在同一协议下训练、评分和评估。并非重写原论文底层求解器，也不宣称精确重现上述论文的表格、超参数、五折实验或原作者代码。

## 方法与固定参数

下列为正式模式，`--quick` 只用于烟雾测试。正式结果不使用 quick。所有随机算法固定传入 seed。

| 方法 / CLI 名称 | 原理与交易适用性 | 本仓库参考实现与主要参数 | 一手来源 |
|---|---|---|---|
| Logistic Regression / `logistic_regression` | 以线性 log-odds 学习交易欺诈概率，作为可解释基线 | `LogisticRegression(C=1, max_iter=2000)`，L2 正则 | [官方数学说明与实现](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) |
| Gaussian Naive Bayes / `gaussian_nb` | 条件独立的高斯类条件密度与贝叶斯后验，计算成本低 | `GaussianNB(var_smoothing=1e-9)` | [官方高斯 NB 说明](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes) |
| k-Nearest Neighbors / `knn` | 邻近交易标签投票；依赖标准化后的距离，适合局部模式 | `KNeighborsClassifier(k=5, weights='distance')` | [官方 kNN 说明与参考文献](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification) |
| RBF Support Vector Machine / `svm_rbf` | 通过核最大间隔捕捉非线性交易边界 | `SVC(C=1, kernel='rbf', gamma='scale', probability=False)` | [Cortes & Vapnik, Support-vector networks (1995)](https://doi.org/10.1007/BF00994018)；[官方说明](https://scikit-learn.org/stable/modules/svm.html) |
| CART Decision Tree / `decision_tree` | 递归以 Gini 不纯度分裂，生成可解释规则 | `DecisionTreeClassifier(max_depth=8, min_samples_leaf=5)` | [官方 CART 说明](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart) |
| Random Forest / `random_forest` | Bootstrap 树和随机特征降低单棵树方差 | 100 树，`min_samples_leaf=2`，其他采用固定版本默认值 | [Breiman (2001), Random Forests](https://www.stat.berkeley.edu/users/breiman/randomforest2001.pdf) |
| AdaBoost / `adaboost` | 逐轮提高错误样本权重，组合决策树桩 | 100 个深度 1 树桩，`learning_rate=0.5`；sklearn 1.7 的离散 SAMME，在二分类下对应离散 AdaBoost | [Freund & Schapire (1997)](https://www.sciencedirect.com/science/article/pii/S002200009791504X)；[官方说明](https://scikit-learn.org/stable/modules/ensemble.html#adaboost) |
| Gradient Boosting / `gradient_boosting` | 对二项对数损失的负梯度逐轮拟合回归树 | 100 树，深度 3，学习率 0.05，`min_samples_leaf=5` | [Friedman (2001), Greedy Function Approximation](https://doi.org/10.1214/aos/1013203451)；[官方说明](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) |
| XGBoost / `xgboost` | 正则化二阶梯度提升树；高效表格数据基线 | 100 树，深度 4，学习率 0.05，行/列采样 0.8，L2=1，`tree_method='hist'`，二分类 logistic 损失 | [Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754)；[官方 API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) |
| Multilayer Perceptron / `mlp` | 反向传播训练非线性神经网络交易分类器 | 隐层 64/32，ReLU，Adam，L2=1e-4，batch=256，学习率 1e-3，最多 300 epoch，无 early stopping | [Rumelhart, Hinton & Williams (1986)](https://doi.org/10.1038/323533a0)；[官方 MLP 说明](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) |

MLP 使用现代 Adam/ReLU 配置，而非 1986 年实验网络；XGBoost 使用现代 CPU histogram 求解。完整参数（包括参考库默认值）随每个方法的 JSON 输出保存。SVM 输出未校准 margin，其余输出正类概率；它们都可用于排序和各自的验证集阈值选择，不能把 SVM 分数解释为概率。

## 统一实验协议

1. 检查二分类标签及冲突标签，去除整行完全重复观测。不会按不含时间的所有特征对不同时间的交易合并。
2. 默认真实 CSV 按时间排序作约 60%/20%/20% 训练/验证/测试；边界相同时间戳全部进入后一段，避免同时间跨段。每段必须含两个类别，否则显式报错。若需和静态文献基线比较，可手动选择 `--split stratified`，这不代表未来交易泛化。
3. 时间列仅用于切分，标签不进入特征。其他需要排除的 ID 或事后字段由 `--drop-columns` 指定。数值中位数填补、标准化、类别众数填补和 one-hot 全部在训练集拟合，测试新类别忽略。高基数类别会形成密集矩阵，需先过滤 ID，并控制特征规模。
4. 默认 `--balance none`，不重采样、不改自然患病率。可选 `--balance balanced` 根据训练标签计算 n/(2*n_class) 权重，传入支持的估计器（包括 sklearn 1.7 MLP）；kNN 不支持 sample_weight，保持不变并在结果中注明。两种配置不可混称为完全相同的训练目标。
5. 每个方法仅用验证标签选择最大 F1 的阈值；并列时优先较高阈值。可选 `--threshold fpr --max-fpr 0.01`：在验证误报率不超限的候选中最大化召回，并包含“不报警”候选。该约束不是测试集或未来数据的保证。
6. 冻结模型和阈值后测试。保存 Average Precision（AP）、梯形 PR-AUC、ROC-AUC、Precision、Recall、F1、MCC、FPR、报警率及 TN/FP/FN/TP。AP 与梯形 PR-AUC分别命名，避免把两种定义混用。阈值不确定性在正类很少的验证集尤其明显。
7. 保存切分索引、逐笔验证/测试分数、源数据校验、依赖版本、seed、完整参数、训练与评分时间、收敛警告。非空输出目录拒绝覆写，避免覆盖既有实验。

## 局限

真实样本实验只检验了一个数据集、一个 seed、一个时间划分；没有置信区间或统计显著性结论。50k 均匀抽样保留了交易欺诈的稀少性，但不能替代 284,807 条全量基准。ULB 数据只有两天且已经匿名 PCA，无法做账户分组隔离或重建发布方 PCA，也不能代表所有金融交易、反洗钱或订单簿操纵场景。没有调参搜索、标签延迟、概念漂移、在线反馈或生产决策成本建模。

正式部署前应按账户/商户等实体隔离并考虑标签成熟期；不能因为时间留出就认为所有形式的数据泄漏都已消除。
