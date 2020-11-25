import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')

# classification library
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/samsung/Desktop/train_dirty.csv')
df = df.drop('customerID', axis=1) # 단순 customer ID
df = df.drop('SeniorCitizen', axis=1) # 고객이 고령자인지
df = df.drop('Dependents', axis=1) # 부양가족
df = df.drop('TotalCharges', axis=1) # 통신비
#df = df.drop('tenure', axis=1) # 고객에 회사에 머문 달

print(df)
print(df.isnull().sum())
print("---------------------------------------------------------------")

# dirty data preprocessing
# example) dirty data fill mean
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')
print(df)
print(df.isnull().sum())
print("---------------------------------------------------------------")

# encoding
encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])
df['PhoneService'] = encoder.fit_transform(df['PhoneService'])
df['MultipleLines'] = encoder.fit_transform(df['MultipleLines'])
df['InternetService'] = encoder.fit_transform(df['InternetService'])
df['OnlineSecurity'] = encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = encoder.fit_transform(df['DeviceProtection'])
df['TechSupport'] = encoder.fit_transform(df['TechSupport'])
df['StreamingTV'] = encoder.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = encoder.fit_transform(df['StreamingMovies'])
df['Contract'] = encoder.fit_transform(df['Contract'])
df['PaperlessBilling'] = encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = encoder.fit_transform(df['PaymentMethod'])
df['Churn'] = encoder.fit_transform(df['Churn'])

# split data
x = df.drop('Churn', axis=1)
print(x)
print("---------------------------------------------------------------")
y = df['Churn'].values
print(y)
print("---------------------------------------------------------------")
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33)

# scaling(아직 안함)
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()
X_train_scaled = Standard_scaler.fit_transform(X_train)
X_test_scaled = Standard_scaler.fit_transform(X_test)

# model
# modle별 파라미터 정의 필요
logistic = LogisticRegression().fit(X_train_scaled,Y_train)  # 완료
kneighbors = KNeighborsClassifier().fit(X_train_scaled,Y_train)  # 완료
decisionTree = DecisionTreeClassifier().fit(X_train_scaled,Y_train)  # 완료
randomForest = RandomForestClassifier().fit(X_train_scaled,Y_train)  # 완료
gradientBoosting = GradientBoostingClassifier().fit(X_train_scaled,Y_train)  # 완료
gaussianNB = GaussianNB().fit(X_train_scaled,Y_train)  # 완료

# ----------------------------------------------------------------------------
# parameters for GridSearch
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
    "max_features": list(range(1, len(X_train.columns), 2))
}

# 0 부터 1 까지 0.01간격으로 parameter range 설정
param_range = []
NBarangeSet = np.arange(0, 1, 0.001)
for i in range(len(NBarangeSet)):
    param_range.append([NBarangeSet[i], 1 - NBarangeSet[i]])
gaussian_params = dict(priors=param_range)

# ----------------------------------------------------------------------------

# model = [logistic, kneighbors, decisionTree, randomForest, gradientBoosting, gaussianNB]
# clf = GridSearchCV(svc, parameters)
# best parameter 나오면 그걸로 test

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_logistic.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_logistic.best_score_) # 최고의 점수
logistic_best = gcv_logistic.best_estimator_

# KNN
gcv_kneighbors = GridSearchCV(kneighbors, param_grid=KNN_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_kneighbors.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("KNN")
print('final params', gcv_kneighbors.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_kneighbors.best_score_)      # 최고의 점수
knn_best = gcv_kneighbors.best_estimator_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_decisionTree.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_decisionTree.best_score_)      # 최고의 점수
decisionTree_best = gcv_decisionTree.best_estimator_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_randomForest.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_randomForest.best_score_)      # 최고의 점수
randomForest_best = gcv_randomForest.best_estimator_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_gradientBoosting.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gradientBoosting.best_score_)      # 최고의 점수
gradientBoosting_best = gcv_gradientBoosting.best_estimator_

# GaussianNB
gcv_gaussianNB = GridSearchCV(gaussianNB, param_grid=gaussian_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_gaussianNB.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("GaussianNB")
print('final params', gcv_gaussianNB.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gaussianNB.best_score_)      # 최고의 점수
gaussianNB_best = gcv_gaussianNB.best_estimator_

# VotingClassifier
eclf2 = VotingClassifier(estimators=[('lr', logistic_best),('knn', knn_best),('dt', decisionTree_best),
                                      ('rf', randomForest_best),('gb', gradientBoosting_best),('gnb', gaussianNB_best)],voting='soft')
eclf2 = eclf2.fit(X_train_scaled,Y_train)
print('voting score', eclf2.score(X_test_scaled,Y_test))
# ----------------------------------------------------------------------------

# evaluation(confusion matrix, ROC curve)
predict_lr = logistic_best.predict(X_test_scaled)
y_score_lr = logistic_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Logistic Regression")
print(confusion_matrix(Y_test, predict_lr))

predict_knn = knn_best.predict(X_test_scaled)
# y_score_knn = knn_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix KNeighborsClassifier")
print(confusion_matrix(Y_test, predict_knn))

predict_dt = decisionTree_best.predict(X_test_scaled)
# y_score_dt = decisionTree_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Decision Tree")
print(confusion_matrix(Y_test, predict_dt))

predict_rf = randomForest_best.predict(X_test_scaled)
# y_score_rf = randomForest_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Random Forest")
print(confusion_matrix(Y_test, predict_rf))

predict_gb = gradientBoosting_best.predict(X_test_scaled)
# y_score_gb = gradientBoosting_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Gradient Bossting")
print(confusion_matrix(Y_test, predict_gb))

predict_gnb = gaussianNB_best.predict(X_test_scaled)
# y_score_gnb = gaussianNB_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix GaussianNB")
print(confusion_matrix(Y_test, predict_gnb))

predict_voting = eclf2.predict(X_test_scaled)
# y_score_voting = eclf2.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix VotingClassifier")
print(confusion_matrix(Y_test, predict_voting))

# ----------------------------------------------------------------------------
# Roc curve
def rocvis(true , prob , label ) :
    from sklearn.metrics import roc_curve
    if type(true[0]) == str :
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true = le.fit_transform(true)
    else :
        pass
    fpr, tpr, thresholds = roc_curve(true, prob)
    plt.plot(fpr, tpr, marker='.', label = label  )

fig , ax = plt.subplots(figsize= (10,10))
plt.plot([0, 1], [0, 1], linestyle='--')
rocvis(Y_test , y_score_lr , "Logistic Regression")
# rocvis(Y_test , y_score_knn , "KNN")
# rocvis(Y_test , y_score_dt , "Decision Tree")
# rocvis(Y_test , y_score_rf , "Random Forest")
# ocvis(Y_test , y_score_gb , "Gradient Bossting")
# rocvis(Y_test , y_score_gnb , "GaussianNB")
# rocvis(Y_test , y_score_voting , "VotingClassifier")
plt.legend(fontsize = 18)
plt.title("Models Roc Curve" , fontsize= 25)
plt.show()
