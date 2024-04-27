import pandas as pd
import seaborn as sns
import numpy as np

df=pd.read_csv('static/heart.data_/heart.data.csv')
print(df.head())
print(df.info())
df.drop('Unnamed: 0', axis='columns', inplace=True)
sns.lmplot(x='biking', y='heart.disease',data=df)
sns.lmplot(x='smoking', y='heart.disease',data=df)
x_df=df.drop('heart.disease', axis=1)
y_df=df['heart.disease']

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x_df, y_df, test_size=0.3, random_state=42)

from sklearn import linear_model

model=linear_model.LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)