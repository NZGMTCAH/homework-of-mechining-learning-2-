import pandas as pd
from sklearn.metrics import accuracy_score  # 计算正确率
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import MinMaxScaler,StandardScaler  # 数据标准化
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier  # 分类AdaBoost
from sklearn.tree import DecisionTreeClassifier  # 分类决策树
from sklearn import svm  # SVM模型
data = pd.read_csv('iris.csv')
#查看各字段数据类型，缺失值
data.info()
#描述统计
data.describe()
#计算协方差
data.cov()
#计算相关系数
data.corr()
print('相关系数\t\t\t因变量\n',data.corr().iloc[-1])
# 划分数据
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
# 数据划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 数据格式化(归一化)，将数据缩放到[0,1]
ss = MinMaxScaler()
X_train = ss.fit_transform(X_train,Y_train)  # 训练模型及归一化数据
X_test = ss.transform(X_test)  # 训练模型及归一化数据

##############分类AdaBoost+决策树
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), algorithm="SAMME.R", n_estimators=200)
bdt.fit(X_train, Y_train)  # 训练模型
bdt_y_score = bdt.predict_proba(X_test) # 得到预测属于某个类别的概率值
# 模型评估：计算模型的准确率/精度
print (bdt.score(X_train, Y_train))
print ('bdt训练集准确率：', accuracy_score(Y_train, bdt.predict(X_train)))
print ('bdt测试集准确率：', accuracy_score(Y_test, bdt.predict(X_test)))
#分类报告
print(classification_report(Y_test, bdt.predict(X_test)))

#########SVM
clf = svm.SVC(C=1, kernel='rbf', gamma=0.1)
# 5.模型训练
clf.fit(X_train, Y_train)
# 6.模型评估：计算模型的准确率/精度
print ('模型精度：',clf.score(X_train, Y_train))
print ('SVM训练集准确率：', accuracy_score(Y_train, clf.predict(X_train)))
print ('SVM测试集准确率：', accuracy_score(Y_test, clf.predict(X_test)))
#分类报告
print(classification_report(Y_test, clf.predict(X_test)))

########逻辑回归
import numpy as np
from sklearn.linear_model import LogisticRegressionCV  # Logistic回归模型
lr = LogisticRegressionCV(Cs=np.logspace(-4,1,50), cv=3, fit_intercept=True,
							penalty='l2', solver='lbfgs', tol=0.01, multi_class='multinomial')
lr.fit(X_train, Y_train)
# 6.模型评估：计算模型的准确率/精度
print ('模型精度：',lr.score(X_train, Y_train))
print ('lr训练集准确率：', accuracy_score(Y_train, lr.predict(X_train)))
print ('lr测试集准确率：', accuracy_score(Y_test, lr.predict(X_test)))
#分类报告
print(classification_report(Y_test, lr.predict(X_test)))

