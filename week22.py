import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as lda
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC as svc
from sklearn.naive_bayes import GaussianNB as gnb
data=pd.read_csv("GDP.csv")
print(data.shape)
print(data.head(5))
print(data.s)
