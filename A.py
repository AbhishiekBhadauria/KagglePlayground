import pandas as pd
import numpy as np
df=pd.read_csv("train.csv")
df=df.drop(columns="id",axis=1)
print(df.columns)
array=df.values
print(array.shape)
features=array[:,0:75]
codes={'Class_1':1,'Class_2':2,'Class_3':3,'Class_4':4,'Class_5':5,'Class_6':6,'Class_7':7,
'Class_8':8,'Class_9':9}
target=df["target"].map(codes)
print(features.shape)
print(features[0])
print(target.shape)
print(target[0])
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=0.30,random_state=1)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(train_x)
X_test = sc.transform (test_x)
import sklearn.linear_model
#clf=sklearn.linear_model.LogisticRegression(solver="sag",max_iter=100,n_jobs=-1,verbose=2).fit(X_train,train_y)
clf=sklearn.linear_model.LogisticRegression().fit(X_train,train_y)
y_pred=clf.predict(X_test)
from sklearn.metrics import mean_squared_log_error,accuracy_score,plot_confusion_matrix
print(np.sqrt(mean_squared_log_error(test_y,y_pred)))
print(accuracy_score(test_y, y_pred))
import matplotlib.pyplot as plt
disp = plot_confusion_matrix(clf, X_test, test_y)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import time
start=time.time
clf = OneVsRestClassifier(SVC(kernel='linear',probability=True,verbose=True))
clf.fit(X_train,train_y,)
end = time.time()
print("Single SVC",end-start,clf.score(X_test,test_y))
#proba = clf.predict_proba(X)

n_estimators = 10
start = time.time()
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True,verbose=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
clf.fit(X_train,train_y)
end = time.time()
print("Bagging SVC ",end - start,clf.score(X_test,test_y))
#proba = clf.predict_proba(X)

start = time.time()
clf = RandomForestClassifier(min_samples_leaf=20,verbose=True)
clf.fit(X_train,train_y)
end = time.time()
print("Random Forest",end - start,clf.score(X_test,test_y))
#proba = clf.predict_proba(X)