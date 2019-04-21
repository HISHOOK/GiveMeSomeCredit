import pickle
with open("DataProcessing.pkl", "rb") as f:   # 加载PKL文件，训练数据
    data = pickle.load(f)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
'''
# 相关矩阵
corr = data.corr()                   # 计算各变量的相关性系数
xticks = list(corr.index)            # x轴标签
yticks = list(corr.index)            # y轴标签
fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap="rainbow",ax=ax1,linewidths=.5, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
ax1.set_xticklabels(xticks, rotation=35, fontsize=7)
ax1.set_yticklabels(yticks, rotation=0, fontsize=7)
plt.show()
'''
# 分箱转化
cut1=pd.qcut(data["RevolvingUtilizationOfUnsecuredLines"],4,labels=False)
cut2=pd.qcut(data["age"],8,labels=False)
bins3=[-1,0,1,3,5,13] # 指定多个区间 左开右闭
cut3=pd.cut(data["NumberOfTime30-59DaysPastDueNotWorse"],bins3,labels=False)
cut4=pd.qcut(data["DebtRatio"],3,labels=False)
cut5=pd.qcut(data["MonthlyIncome"],4,labels=False)
cut6=pd.qcut(data["NumberOfOpenCreditLinesAndLoans"],4,labels=False)
bins7=[-1, 0, 1, 3,5, 20]
cut7=pd.cut(data["NumberOfTimes90DaysLate"],bins7,labels=False)
bins8=[-1, 0,1,2, 3, 33]
cut8=pd.cut(data["NumberRealEstateLoansOrLines"],bins8,labels=False)
bins9=[-1, 0, 1, 3, 12]
cut9=pd.cut(data["NumberOfTime60-89DaysPastDueNotWorse"],bins9,labels=False)
bins10=[-1, 0, 1, 2, 3, 5, 21]
cut10=pd.cut(data["NumberOfDependents"],bins10,labels=False)
# WOE值计算
import numpy as np
rate=data["SeriousDlqin2yrs"].sum()/(data["SeriousDlqin2yrs"].count()-data["SeriousDlqin2yrs"].sum())
def get_woe_data(cut,data):
    grouped=data["SeriousDlqin2yrs"].groupby(cut,as_index = True).value_counts()
    woe=np.log(grouped.unstack().iloc[:,1]/grouped.unstack().iloc[:,0]/rate)
    return woe
cut1_woe=get_woe_data(cut1,data)
cut2_woe=get_woe_data(cut2,data)
cut3_woe=get_woe_data(cut3,data)
cut4_woe=get_woe_data(cut4,data)
cut5_woe=get_woe_data(cut5,data)
cut6_woe=get_woe_data(cut6,data)
cut7_woe=get_woe_data(cut7,data)
cut8_woe=get_woe_data(cut8,data)
cut9_woe=get_woe_data(cut9,data)
cut10_woe=get_woe_data(cut10,data)

# IV值计算
def get_IV_data(cut,cut_woe,data):
    grouped=data["SeriousDlqin2yrs"].groupby(cut,as_index = True).value_counts()
    cut_IV=((grouped.unstack().iloc[:,1]/data["SeriousDlqin2yrs"].sum()-grouped.unstack().iloc[:,0]/(data["SeriousDlqin2yrs"].count()-data["SeriousDlqin2yrs"].sum()))*cut_woe).sum()    
    return cut_IV
#计算各分组的IV值
cut1_IV=get_IV_data(cut1,cut1_woe,data)
cut2_IV=get_IV_data(cut2,cut2_woe,data)
cut3_IV=get_IV_data(cut3,cut3_woe,data)
cut4_IV=get_IV_data(cut4,cut4_woe,data)
cut5_IV=get_IV_data(cut5,cut5_woe,data)
cut6_IV=get_IV_data(cut6,cut6_woe,data)
cut7_IV=get_IV_data(cut7,cut7_woe,data)
cut8_IV=get_IV_data(cut8,cut8_woe,data)
cut9_IV=get_IV_data(cut9,cut9_woe,data)
cut10_IV=get_IV_data(cut10,cut10_woe,data)
'''
IV=pd.DataFrame([cut1_IV,cut2_IV,cut3_IV,cut4_IV,cut5_IV,cut6_IV,cut7_IV,cut8_IV,cut9_IV,cut10_IV],index=['可用额度比值','年龄','逾期30-59天笔数','负债率','月收入','信贷数量','逾期90天笔数','固定资产贷款量','逾期60-89天笔数','家属数量'],columns=['IV'])
plt.rcParams['font.sans-serif'] = ['SimHei'] # 图中显示中文字体
iv=IV.plot.bar(color='b',alpha=0.3,rot=30,figsize=(10,5),fontsize=(10))
iv.set_title('特征变量与IV值分布图',fontsize=(15))
iv.set_xlabel('特征变量',fontsize=(15))
iv.set_ylabel('IV',fontsize=(15))
plt.show()
'''

# 新建df_new存放woe转换后的数据
df_new=pd.DataFrame()   
def replace_data(cut,cut_woe):
    a=[]
    for i in cut.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        cut.replace(a[m],cut_woe.values[m],inplace=True)
    return cut
df_new["好坏客户"]=data["SeriousDlqin2yrs"]
df_new["可用额度比值WOE"]=replace_data(cut1,cut1_woe)
df_new["年龄WOE"]=replace_data(cut2,cut2_woe)
df_new["逾期30-59天笔数WOE"]=replace_data(cut3,cut3_woe)
df_new["负债率WOE"]=replace_data(cut4,cut4_woe)
df_new["月收入WOE"]=replace_data(cut5,cut5_woe)
df_new["信贷数量WOE"]=replace_data(cut6,cut6_woe)
df_new["逾期90天笔数WOE"]=replace_data(cut7,cut7_woe)
df_new["固定资产贷款量WOE"]=replace_data(cut8,cut8_woe)
df_new["逾期60-89天笔数WOE"]=replace_data(cut9,cut9_woe)
df_new["家属数量WOE"]=replace_data(cut10,cut10_woe)
df_new.head()
with open("WOE.pkl", 'wb') as f:
    pickle.dump(df_new, f)


