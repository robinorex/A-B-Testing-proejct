import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
data_csv = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

# print(data_csv)
# # data_excel = pd.read_excel('Excel Project Dataset-Bike sales.xlsx',sheet_name='working sheet')
# # print(data_excel)
# print(data_csv.head())
# print(data_csv.info())
# print(data_csv.dropna())
# print(data_csv.fillna('NULL'))
# print(data_csv)
# print(data_csv.drop_duplicates())
# import sqlite3
# conn = sqlite3.connect('Bachelors-Degrees.db')
# query1= 'select col_1 from table_name'
# query2= 'select * from table_name'
# data_sql= pd.read_sql(query2,conn)
# print(data_csv.iloc[10])
# data = pd.DataFrame({
#     "A":[1,2,3],
#     "B":[4,5,6],
#     "C":[7,8,9]},
#  index=['X','Y','Z'])
# print(data)
# print(data.loc['X'])
# print(data.loc[:,'B'])
# print(data.loc['Y',:])

# print(data.loc['Y','A'])
# data_csv.sort_values(by='Year', ascending=False, inplace=True)
# print(data_csv.head())
# data = pd.DataFrame({"Name":['Anna','Karen','John','Alice','Kevin','Sanna','Bob','Emily'],
#                      "Age":[35,44,22,12,23,44,16,28],
#                      "Salary":[10000,40000,3000,200,256,45454,234234,444],
#                      "Department":['Tech','Tech','Tech','Healthcare','Tech','Operation','Operation','Tech']})
# print(data.sort_values(by='Salary', ascending= False))
# print("\n",data.groupby('Department')['Age'].count())
# print(data.groupby('Department')['Salary'].mean())
# print('\n', data.groupby('Department')['Age'].mean())
# print(data[data["Salary"]>10000])
# print('\n',data[(data["Salary"]>10000) & (data["Salary"]<50000)])
# print("\n",data[data["Age"].isin([44,16])])

# data = [ 100,20,5,20,45,-100,46]
# print(np.mean(data))
# print(np.median(data))
# from scipy import stats
# print(stats.mode(data))
# print(np.var(data))
# print(np.std(data))
# print(data_csv.describe())
# data1 = pd.DataFrame({'key':["a","b","c","d","e","f","g"],
#                       'value':[1,2,3,4,5,6,7]})
# data2 = pd.DataFrame({'key':["c","d","e","f","g","h"],
#                       'value':[8,9,10,11,12,13]})
# # print(pd.merge(data1,data2,on='key',how='left'))
# # print('\n',pd.merge(data1,data2,on='key',how='inner'))
# # print('\n',pd.merge(data1,data2,on='key',how='right'))
# merge_left = pd.merge(data1,data2,on='key',how='left',indicator=True)
# print(merge_left)
# merge_anit_left = merge_left[merge_left['_merge']=='left_only']
# print("\n",merge_anit_left)
# merge_anit_left = merge_anit_left.drop(['_merge'],axis=1)   #axis = 1 drop column
# print(merge_anit_left)
# merge_right = pd.merge(data1,data2,on='key',how='right',indicator=True)
# print(merge_right)
# merge_anti_right = merge_right[merge_right['_merge']=='right_only']
# print('\n',merge_anti_right)
# merge_anti_right = merge_anti_right.drop(['_merge'],axis=1)
# print(merge_anti_right)
# x_values = [1,2,3,4,5,6,7,8]
# y_values = [1,4,5,6,9,12,24,19]
# plt.plot(x_values,y_values,'r--')
# plt.title("Title placeholder")
# plt.xlabel("x-placeholder")
# plt.ylabel("y-placeholder")
# # plt.show()
# plt.scatter(x_values,y_values,color='green')
# plt.title("Title placeholder")
# plt.xlabel("x-placeholder")
# plt.ylabel("y-placeholder")
# # plt.show()
# plt.bar(x_values,y_values,color='forestgreen')
# plt.show()

# x_normal = np.random.normal(0,1,10000)
# plt.hist(x_normal,color='green')
# plt.xlabel("X")
# plt.ylabel("Frequency")
# plt.title("Randomly Sample Data")
# plt.show()
# x_values = np.arange(-4, 4, 0.01)
# y_values = norm.pdf(x_values)   #population distribution
# counts,bins,ignored= plt.hist(x_normal,30,density=True,color='green',label='Sample Distribution')
# plt.plot(x_values,y_values,color='y',linewidth=3,label='Population Distribution')
# plt.title('Randomly Generating 10000 obs from standard normal distribution')
# plt.ylabel('Probability')
# plt.legend()
# plt.show()

#arrival rate
# lambda_ = 8
# N = 1000
# X = np.random.poisson(lambda_,N)
# counts, bins, ignored = plt.hist(X, 40, density = True, color = 'purple')
# plt.title("Randomly generating from Poisson Distribution with lambda = 8")
# plt.xlabel("Number of arrivals")
# plt.ylabel("Probability")
# plt.show()
