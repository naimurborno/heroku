import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


data=pd.read_csv('hiring.csv')
data['experience'].fillna(0,inplace=True)

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

data['experience'] = data['experience'].apply(lambda x : convert_to_int(x))
data['test_score'].fillna(data['test_score'].mean(),inplace=True)
from sklearn.linear_model import LinearRegression
X=data.drop('salary',axis=1)
y=data['salary']
X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.2)
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))
model2=pickle.load(open('model.pkl','rb'))
print(model2.predict([[2,9,6]]))