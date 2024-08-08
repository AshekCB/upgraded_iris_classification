import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv("D:\Flask Demo's\Flask Iris Flower Classification\mlmodels\Pre-Processed-Iris.csv")

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['Species']=label.fit_transform(df['Species'])

knn=KNeighborsClassifier(n_neighbors=5)

x=df.drop(['Species'],axis=1)
y=df['Species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

knn.fit(x_train,y_train)

def predictor(x1,x2,y1,y2):
    d={
    'SepalLengthCm':x1,
    'SepalWidthCm':x2,
    'PetalLengthCm':y1,
    'PetalWidthCm':y2
    }

    testing_df=pd.DataFrame(d,index=[0])
    res=knn.predict(testing_df)
    return res[0]

