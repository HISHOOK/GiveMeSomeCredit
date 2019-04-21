'''
'SeriousDlqin2yrs':'好坏客户'(label),
'RevolvingUtilizationOfUnsecuredLines':'可用额度比值',
'age':'年龄',
'NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天笔数',
'DebtRatio':'负债率',
'MonthlyIncome':'月收入',
'NumberOfOpenCreditLinesAndLoans':'信贷数量',
'NumberOfTimes90DaysLate':'逾期90天笔数',
'NumberRealEstateLoansOrLines':'固定资产贷款量',
'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天笔数',
'NumberOfDependents':'家属数量'}
'''
import pandas as pd
import matplotlib.pyplot as plt #导入图像库
from sklearn.ensemble import RandomForestRegressor
path='cs-training.csv'
#载入数据
data = pd.read_csv('cs-training.csv')
#数据集确实和分布情况
#data.describe().to_csv('DataDescribe.csv')

# 用随机森林对缺失值预测填充函数
def set_missing(df):
    # 把已有的数值型特征取出来
    process_df = df.ix[:,[5,0,1,2,3,4,6,7,8,9]]
    # 分成已知该特征和未知该特征两部分
    # dataframe.values获取的是dataframe中的数据为数组array
    known = process_df[process_df.MonthlyIncome.notnull()].values
    unknown = process_df[process_df.MonthlyIncome.isnull()].values
    # X为已知MonthlyIncome的特征属性值
    X = known[:, 1:]
    # y为结果标签值MonthlyIncome
    y = known[:, 0]
    # X与y用于训练随机森林模型，fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200,max_depth=3,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df
data=set_missing(data)              # 用随机森林填补比较多的缺失值
data=data.dropna()                  # 删除比较少的缺失值
data = data.drop_duplicates()       # 删除重复项
data.to_csv('MissingData.csv',index=False)
'''
fig=plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei'] # 图中显示中文字体
ax1=fig.add_subplot(131) 
ax1.boxplot(data['NumberOfTime30-59DaysPastDueNotWorse'])
ax1.set_title('逾期30-59天笔数')
ax2=fig.add_subplot(132)
ax2.boxplot(data['NumberOfTimes90DaysLate'])
ax2.set_title('逾期90天笔数')
ax3=fig.add_subplot(133)
ax3.boxplot(data['NumberOfTime60-89DaysPastDueNotWorse'])
ax3.set_title('逾期60-89天笔数')
plt.show()
'''

describe=data.describe()
#print(describe['NumberOfTime30-59DaysPastDueNotWorse'].ix['25%'])

# 对数据进行异常值处理（排除月收入下，小于Q1-1.5*IQR或大于Q3+1.5IQR的outlier）
'''
def OutlierProcessing(df,variable,describe):
    Q1=describe[variable].ix['25%']
    Q3=describe[variable].ix['75%']
    IQR=Q3-Q1
    print(Q1,Q3,IQR)
    df = df[ Q1-1.5*IQR<df[variable] <Q3+1.5*IQR ]
    return df
data=OutlierProcessing(data,'MonthlyIncome',describe)
'''
# 对于'可用额度比值','负债率'，'信贷数量'，'月收入'，'逾期30-59天的笔数'，'固定资产贷款量'等去除单侧95%上部分异常值
for variable in ['RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','MonthlyIncome','DebtRatio','NumberOfOpenCreditLinesAndLoans',
                 'NumberRealEstateLoansOrLines']:
    data=data[data[variable]<data[variable].quantile(0.99)]
data.describe().to_csv('DataDescribe1.csv')


import pickle
with open("DataProcessing.pkl", 'wb') as f:
    pickle.dump(data, f)



