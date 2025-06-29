import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 读取 CSV 文件
file_path = 'upper_model/ResultList_QN_5000_eps_inrealmap_629.p.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 提取真实标签和预测标签
true_labels = df['TrueMode']
pred_labels = df['PredictMode']

# 计算混淆矩阵
labels = ['GSD', 'GG', 'TS', 'TG']  # 类别标签
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 生成分类报告
report = classification_report(true_labels, pred_labels, labels=labels, output_dict=True)
# print(report)
report_df = pd.DataFrame(report).transpose()

# 绘制分类报告柱状图
report_df.iloc[:-1, :3].plot(kind='bar', figsize=(10, 6), rot=45)
plt.title('Classification Report Metrics')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()