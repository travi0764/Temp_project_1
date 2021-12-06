import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from utilis import cleaning,embed

data=pd.read_csv("NEW_DATA_RT.csv")
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.dropna(inplace=True)

data.tweet=data.tweet.apply(cleaning)

y=data['label']
x=data['tweet']    

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)

X_train_tweet = embed(list(X_train))
X_test_tweet = embed(list(X_test))

#X_train=embed(list(x))

labels_=['Depression','Normal','Smoking']

svm = LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-4, C=0.1)
svm.fit(X_train_tweet, y_train)

pickle.dump(svm,open("model.pkl","wb"))




