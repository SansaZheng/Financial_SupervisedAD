# 数据

不提交交易原始数据。`download_creditcard.py` 从 TensorFlow 官方教程使用的公开镜像读取 ULB/Worldline 数据（284,807 笔交易，492 笔欺诈），默认只保存 50,000 条均匀随机样本，非标签平衡抽样。

```bash
python scripts/download_creditcard.py --output data/creditcard_50k.csv.gz --sample-size 50000 --seed 42
```

脚本读取完整远端流并记录 SHA-256、HTTP ETag、源记录索引和样本校验和。默认不会将 151 MB 完整 CSV 落盘。样本仍有严重类别不平衡，具体正类数量以运行的 manifest 为准。原数据中的 V1–V28 是发布方预先计算的 PCA 特征；本仓库无法恢复 PCA 的拟合范围。

完整数据模式：`--sample-size 0 --output /data/financial-supervised-ad/creditcard.csv.gz`。路径应使用你有权限的大容量工作区。完整数据会先载入内存，建议至少 4 GB 内存。该命令不自动提交集群任务或产生云租赁费用。

数据说明：[原始发布页](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)、[TensorFlow 官方教程及镜像出处](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)。数据使用、许可和再分发条件以原发布方为准；仓库代码许可证不覆盖外部数据。

自有数据支持数值、类别特征和缺失值。目标必须明确编码为 0=正常、1=异常。请通过 `--drop-columns` 去除交易 ID、账户 ID、标签代理、事后调查信息；仅使用决策时已知的特征。通用加载器无法自动识别业务泄漏。时间列支持数值或可解析的日期字符串，默认不作为模型特征。
