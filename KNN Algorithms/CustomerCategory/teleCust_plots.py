import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('E:\Pythoncode\Coursera\Classification_Algorithms\KNN Algorithms\CustomerCategory/teleCust1000t.csv')

# print(df.head())

# value_counts() function is used to count different value separately in column custcat
# eg.
#    3    281
#    1    266
#    4    236
#    2    217

# count() function counts the number of value in column custcat, or sum of all value_counts(), here 1000

print(df['custcat'].value_counts())

# if sample is from 1 is to 100 , then bin size of 50 implies 50 range of histgram, from [0,2) to [98,100], Last bin include 100
# basically bins are number of class size.
df.hist(column = 'income', bins = 50)
plt.show()

print(df.count())

plt.scatter(df.custcat, df.income,  color = 'blue')

plt.xlabel('custcat')
plt.ylabel('income')

plt.show()



