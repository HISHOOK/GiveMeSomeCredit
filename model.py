import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

#数据提取与数据分割
with open("WOE.pkl", "rb") as f:   # 加载PKL文件，训练数据
    data = pickle.load(f)
col_names=data.columns.values
X = data[col_names[1:]]
Y = data[col_names[0]]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=0)

#构建并训练模型
lr = LogisticRegressionCV(cv=10,penalty="l2")
re = lr.fit(X_train,Y_train)
print("训练完成")
print("参数:",re.coef_)
print("截距:",re.intercept_)
#模型的保存与加载
from sklearn.externals import joblib
joblib.dump(lr,"logistic_lr.model")     # 将训练后的线性模型保存
joblib.load("logistic_lr.model")        # 加载模型

# 测试集数据预测
Y_pred = lr.predict(X_test)                                  # 预测分类
prob_pred=[round(u[1],5) for u in lr.predict_proba(X_test)]  # 阳性预测概率

# 模型评估
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import auc
print("准确率：",accuracy_score(Y_test, Y_pred))
# 样本类别不平衡，用PR不好评价，采用ROC曲线
# print("精确度",precision_score(Y_test, Y_pred, average='binary'))
# print("召回率",recall_score(Y_test, Y_pred, average='binary'))
# print("F1",f1_score(Y_test, Y_pred, average='binary'))

FPR, TPR, thresholds = metrics.roc_curve(Y_test, prob_pred, pos_label=1)
print("AUC:",metrics.auc(FPR, TPR))



#画图对预测值和实际值进行比较
#解决中文显示问题
mpl.rcParams["font.sans-serif"] = [u"SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % metrics.auc(FPR, TPR))#生成ROC曲线
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('真正率')
plt.xlabel('假正率')
plt.show()
