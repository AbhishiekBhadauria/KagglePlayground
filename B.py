import pandas as pd
import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential
df=pd.read_csv("train.csv")
df=df.drop(columns="id",axis=1)
print(df.columns)
array=df.values
print(array.shape)
features=array[:,0:75]
codes={'Class_1':1,'Class_2':2,'Class_3':3,'Class_4':4,'Class_5':5,'Class_6':6,'Class_7':7,
'Class_8':8,'Class_9':9}
target=df["target"].map(codes)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=0.30,random_state=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(train_x)
X_test = sc.transform(test_x)
n_features = X_train.shape[1]
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile the model
print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train,train_y, epochs=150, batch_size=32, verbose=2)
# evaluate the model
loss, acc = model.evaluate(X_test,test_y,verbose=0)
print('Test Accuracy: %.3f' % acc)