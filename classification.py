import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')

# Load the data
# 데이터 null값 생성한 것으로 변경
df = pd.read_csv('C:/Users/samsung/Desktop/train_dirty.csv')
print(df)
print(df.isnull().sum())

# dirty data preprocessing
# example) dirty data fill mean
df = df.fillna(method='bfill')
print(df)

# encoding
encoder = LabelEncoder()
for cols in df.columns:
    # LabelEncoder: TypeError, not supported between instances of 'float' and 'str'
    # classification에 이용할 column들을 미리 정의하고 encoding을 하는 것이 효율적일 것으로 판단
    df[cols] = encoder.fit_transform(df[cols].astype(str))
    df[cols] = encoder.fit_transform(df[cols])

# split data
x = df.drop('Churn', axis=1)
y = df['Churn'].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33)

# scaling(아직 안함)
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()
X_scaled = MinMax_scaler.fit_transform(X_train)
X_test_scaled = MinMax_scaler.fit_transform(X_test)

# modeling
# logistic regression (categorical data일때 적합한데 현data가 categorical이라 적합할 것으로 판단)
# knn(괜찮을 듯)
# svm(데이터가 많은 것은 비적합한 모델, 사용하지 않을 예정)
# decision tree
# random forest
# gradientboostingclassifier
# xgbclassifier(학습시간이 오래 걸림, 데이터가 많은 것은 비적합 모델(아마도))
# gaussiannb
# votingclassifier(library 선언 안함)

# library
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# model
# train, validation, test set으로 데이터 나눠야 할 듯
# validation dataset (test set이 아니고)을 사용해서 비교할 경우에는 random forest의 정확도가 decision tree보다 항상 좋다
logistic = LogisticRegression()
kneighbors = KNeighborsClassifier()
decisionTree = DecisionTreeClassifier()
randomForest = RandomForestClassifier() # 너무 오래 걸림 (ensemble - bagging)
gradientBoosting = GradientBoostingClassifier() # 너무 오래 걸림 (ensemble - boosting) # 예측 성능이 높다고 알려진 모델인데 너무 정확도 낮음
gaussianNB = GaussianNB() # 너무 오래 걸림

# model parameter
decisionTree_params = {
    'max_depth': [None, 6, 8, 10, 12, 16, 20, 24],
    'min_samples_split': [2, 20, 50, 100, 200],
    'criterion': ['entropy', 'gini']
}
logistic_params = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs', 'sag'],
    'max_iter': [50, 100, 200]
}
randomForest_params = {
    'n_estimators': [],
    'max_featrues': []
}
KNN_params = {
    'n_neighbors': [3, 5, 11, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
randomForest_params = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

gradient_params = {
    "n_estimators": range(50, 100, 25),
    "max_depth": [1, 2, 4],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "subsample": [0.7, 0.9],
    "max_features": list(range(1, 19, 2))
}

# 0 부터 1 까지 0.01간격으로 parameter range 설정
param_range = []
NBarangeSet = np.arange(0, 1, 0.001)
for i in range(len(NBarangeSet)):
    param_range.append([NBarangeSet[i], 1 - NBarangeSet[i]])

gaussian_params = dict(priors=param_range)

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, cv=cv, scoring='f1', n_jobs=4)
gcv_logistic.fit(X_scaled,Y_train)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_logistic.best_score_) # 최고의 점수
logistic_best = gcv_logistic.best_estimator_

# KNN
gcv_kneighbors = GridSearchCV(kneighbors, param_grid=KNN_params, cv=cv, scoring='f1', n_jobs=4)
gcv_kneighbors.fit(X_scaled,Y_train)
print("---------------------------------------------------------------")
print("KNN")
print('final params', gcv_kneighbors.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_kneighbors.best_score_)      # 최고의 점수
knn_best = gcv_kneighbors.best_estimator_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, cv=cv, scoring='f1', n_jobs=4)
gcv_decisionTree.fit(X_scaled,Y_train)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_decisionTree.best_score_)      # 최고의 점수
decisionTree_best = gcv_decisionTree.best_estimator_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, cv=cv, scoring='f1', n_jobs=4)
gcv_randomForest.fit(X_scaled,Y_train)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_randomForest.best_score_)      # 최고의 점수
randomForest_best = gcv_randomForest.best_estimator_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, cv=cv, scoring='f1', n_jobs=4)
gcv_gradientBoosting.fit(X_scaled,Y_train)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gradientBoosting.best_score_)      # 최고의 점수
gradientBoosting_best = gcv_gradientBoosting.best_estimator_

# GaussianNB
gcv_gaussianNB = GridSearchCV(gaussianNB, param_grid=gaussian_params, cv=cv, scoring='f1', n_jobs=4)
gcv_gaussianNB.fit(X_scaled,Y_train)
print("---------------------------------------------------------------")
print("GaussianNB")
print('final params', gcv_gaussianNB.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gaussianNB.best_score_)      # 최고의 점수
gaussianNB_best = gcv_gaussianNB.best_estimator_

# VotingClassifier
eclf2 = VotingClassifier(estimators=[('lr', logistic_best),('knn', knn_best),('dt', decisionTree_best),
                                      ('rf', randomForest_best),('gb', gradientBoosting_best),('gnb', gaussianNB_best)],voting='soft')
eclf2 = eclf2.fit(X_scaled,Y_train)
print('voting score', eclf2.score(X_test_scaled,Y_test))

# 성능 테스트(confusion matrix, ROC curve)
from sklearn.metrics import confusion_matrix
