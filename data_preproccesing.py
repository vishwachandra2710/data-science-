import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\vishw\Downloads\18th, 19th - ML\18th - ML\5. Data preprocessing\Data.csv")
print(df)
y=df.iloc[:,3].values
x=df.iloc[:,:3].values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()

x[:,1:3]=imputer.fit_transform(x[:,1:3])
from sklearn.preprocessing import LabelEncoder
l_x=LabelEncoder()
x[:,0]=l_x.fit_transform(x[:,0])

l_y=LabelEncoder()
y=l_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8,random_state=0)
