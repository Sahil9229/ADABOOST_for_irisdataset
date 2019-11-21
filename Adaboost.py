import pandas as pd import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix


from sklearn.ensemble import AdaBoostClassifier from sklearn import metrics


from sklearn import datasets


#------------------------------------------------------------------------------


irisdata=datasets.load_iris() print(irisdata)


x=irisdata.data y=irisdata.target


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


abc=AdaBoostClassifier(n_estimators=50,learning_rate=1) #abc=AdaBoostClassifier(n_estimators=50,base_estimator=svc,learning_rate=1) abc.fit(x_train,y_train)


y_pred=abc.predict(x_test)


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


print(confusion_matrix(y_test,y_pred)) print(classification_report(y_test,y_pred))
