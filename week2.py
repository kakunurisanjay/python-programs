import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as lda
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC as svc
from sklearn.naive_bayes import GaussianNB as gnb
data=pd.read_csv("iris_data.csv")
#Analysis of data
print(data.shape)
print(data.head(5))
#array=dataset.values
X=data.drop(['species'],axis=1)
Y=data['species']
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=0.20,random_state=(1))
logReg(solver='liblinear',multi_class='ovr')
lda()
knn()
dtc()
gnb()
model=svc(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)
#Evaluate predictions
print('**********************************************************')
print('Accuracy: ')
print(accuracy_score(Y_validation,predictions))
print('**********************************************************')
print('Confusion Matrix :['+' '.join(set(data.species))+']')
print(confusion_matrix(Y_validation,predictions))
print('**********************************************************')
print('**********************************************************')
print('Classification Report: ')
print(classification_report(Y_validation,predictions))
print('**********************************************************')
model=lda()
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)
#Evaluate predictions
print('**********************************************************')
print('Accuracy: ')
print(accuracy_score(Y_validation,predictions))
print('**********************************************************')
print('Confusion Matrix :['+' '.join(set(data.species))+']')
print(confusion_matrix(Y_validation,predictions))
print('**********************************************************')
print('**********************************************************')
print('Classification Report: ')
print(classification_report(Y_validation,predictions))
print('**********************************************************')
model=knn()
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)
#Evaluate predictions
print('**********************************************************')
print('Accuracy: ')
print(accuracy_score(Y_validation,predictions))
print('**********************************************************')
print('Confusion Matrix :['+' '.join(set(data.species))+']')
print(confusion_matrix(Y_validation,predictions))
print('**********************************************************')
print('**********************************************************')
print('Classification Report: ')
print(classification_report(Y_validation,predictions))
print('**********************************************************')
model=dtc()
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)
#Evaluate predictions
print('**********************************************************')
print('Accuracy: ')
print(accuracy_score(Y_validation,predictions))
print('**********************************************************')
print('Confusion Matrix :['+' '.join(set(data.species))+']')
print(confusion_matrix(Y_validation,predictions))
print('**********************************************************')
print('**********************************************************')
print('Classification Report: ')
print(classification_report(Y_validation,predictions))
print('**********************************************************')
model=gnb()
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)
#Evaluate predictions
print('**********************************************************')
print('Accuracy: ')
print(accuracy_score(Y_validation,predictions))
print('**********************************************************')
print('Confusion Matrix :['+' '.join(set(data.species))+']')
print(confusion_matrix(Y_validation,predictions))
print('**********************************************************')
print('**********************************************************')
print('Classification Report: ')
print(classification_report(Y_validation,predictions))
print('**********************************************************')


