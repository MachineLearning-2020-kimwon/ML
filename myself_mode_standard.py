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

# evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

# Read file, Feature select
df = pd.read_csv('C:/Users/samsung/Desktop/train_dirty.csv')
print("*",df.dtypes)
df['TotalCharges']=df['TotalCharges'].apply(pd.to_numeric, errors='coerce')
df = df.drop('customerID', axis=1)
df = df.drop('SeniorCitizen', axis=1)
df = df.drop('Dependents', axis=1)
# df = df.drop('TotalCharges', axis=1)
df = df.drop('gender', axis=1)


print(df)
print(df.isnull().sum()) # check null value
print("---------------------------------------------------------------")

# dirty data preprocessing
# dirty data fill mode
df = df.fillna(method='ffill')
df = df.fillna(df.mode())
print(df)
print(df.isnull().sum()) # check, fill nan
print("---------------------------------------------------------------")

# encoding
encoder = LabelEncoder()
# df['gender'] = encoder.fit_transform(df['gender'])
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

# split data(train, test)
x = df.drop('Churn', axis=1)
print(x)
print("---------------------------------------------------------------")
y = df['Churn'].values
print(y)
print("---------------------------------------------------------------")
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33)

# scaling(MinMax)
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()
X_train_scaled = Standard_scaler.fit_transform(X_train)
X_test_scaled = Standard_scaler.fit_transform(X_test)

# ----------------------------------------------------------------------------
# model
logistic = LogisticRegression().fit(X_train_scaled,Y_train)
kneighbors = KNeighborsClassifier().fit(X_train_scaled,Y_train)
decisionTree = DecisionTreeClassifier().fit(X_train_scaled,Y_train)
randomForest = RandomForestClassifier().fit(X_train_scaled,Y_train)
gradientBoosting = GradientBoostingClassifier().fit(X_train_scaled,Y_train)
gaussianNB = GaussianNB().fit(X_train_scaled,Y_train)

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

param_range = []
NBarangeSet = np.arange(0, 1, 0.001)
for i in range(len(NBarangeSet)):
    param_range.append([NBarangeSet[i], 1 - NBarangeSet[i]])
gaussian_params = dict(priors=param_range)

# ----------------------------------------------------------------------------
# model = [logistic, kneighbors, decisionTree, randomForest, gradientBoosting, gaussianNB]

# GridSearch
cv = KFold(n_splits=10, random_state=1)

# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_logistic.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)   # print best parameter
print('best score', gcv_logistic.best_score_) # best score
logistic_best = gcv_logistic.best_estimator_

# KNN
gcv_kneighbors = GridSearchCV(kneighbors, param_grid=KNN_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_kneighbors.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("KNN")
print('final params', gcv_kneighbors.best_params_)   # print best parameter
print('best score', gcv_kneighbors.best_score_)      # best score
knn_best = gcv_kneighbors.best_estimator_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_decisionTree.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # print best parameter
print('best score', gcv_decisionTree.best_score_)      # best score
decisionTree_best = gcv_decisionTree.best_estimator_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_randomForest.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # print best parameter
print('best score', gcv_randomForest.best_score_)      # best score
randomForest_best = gcv_randomForest.best_estimator_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_gradientBoosting.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)   # print best parameter
print('best score', gcv_gradientBoosting.best_score_)      # best score
gradientBoosting_best = gcv_gradientBoosting.best_estimator_

# GaussianNB
gcv_gaussianNB = GridSearchCV(gaussianNB, param_grid=gaussian_params, scoring='accuracy', cv=cv, verbose=1,n_jobs=-1,)
gcv_gaussianNB.fit(X_train_scaled,Y_train)
print("---------------------------------------------------------------")
print("GaussianNB")
print('final params', gcv_gaussianNB.best_params_)   # print best parameter
print('best score', gcv_gaussianNB.best_score_)      # best score
gaussianNB_best = gcv_gaussianNB.best_estimator_

# VotingClassifier (decision tree, randomForest, gradientBoosting, gaussianNB)
eclf2 = VotingClassifier(estimators=[('dt', decisionTree_best),('rf', randomForest_best),('gb', gradientBoosting_best),('gnb', gaussianNB_best)],voting='soft')
eclf2 = eclf2.fit(X_train_scaled,Y_train)
print('voting score', eclf2.score(X_test_scaled,Y_test))

# ----------------------------------------------------------------------------
# evaluation(confusion matrix, ROC curve)
# Confusion Matrix
predict_lr = logistic_best.predict(X_test_scaled)
print("Confusion matrix Logistic Regression")
lr_p = precision_score(Y_test, predict_lr)
print(lr_p)
lr_r = recall_score(Y_test, predict_lr)
print(lr_r)
lr_f1 = f1_score(Y_test, predict_lr)
print(lr_f1)
print(confusion_matrix(Y_test, predict_lr))

predict_knn = knn_best.predict(X_test_scaled)
print("Confusion matrix KNeighborsClassifier")
knn_p = precision_score(Y_test, predict_knn)
print(knn_p)
knn_r = recall_score(Y_test, predict_knn)
print(knn_r)
knn_f1 = f1_score(Y_test, predict_knn)
print(knn_f1)
print(confusion_matrix(Y_test, predict_knn))

predict_dt = decisionTree_best.predict(X_test_scaled)
# y_score_dt = decisionTree_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Decision Tree")
dt_p = precision_score(Y_test, predict_dt)
print(dt_p)
dt_r = recall_score(Y_test, predict_dt)
print(dt_r)
dt_f1 = f1_score(Y_test, predict_dt)
print(dt_f1)
print(confusion_matrix(Y_test, predict_dt))

predict_rf = randomForest_best.predict(X_test_scaled)
# y_score_rf = randomForest_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Random Forest")
rf_p = precision_score(Y_test, predict_rf)
print(rf_p)
rf_r = recall_score(Y_test, predict_rf)
print(rf_r)
rf_f1 = f1_score(Y_test, predict_rf)
print(rf_f1)
print(confusion_matrix(Y_test, predict_rf))

predict_gb = gradientBoosting_best.predict(X_test_scaled)
# y_score_gb = gradientBoosting_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix Gradient Bossting")
gb_p = precision_score(Y_test, predict_gb)
print(gb_p)
gb_r = recall_score(Y_test, predict_gb)
print(gb_r)
gb_f1 = f1_score(Y_test, predict_gb)
print(gb_f1)
print(confusion_matrix(Y_test, predict_gb))

predict_gnb = gaussianNB_best.predict(X_test_scaled)
# y_score_gnb = gaussianNB_best.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix GaussianNB")
gnb_p = precision_score(Y_test, predict_gnb)
print(gnb_p)
gnb_r = recall_score(Y_test, predict_gnb)
print(gnb_r)
gnb_f1 = f1_score(Y_test, predict_gnb)
print(gnb_f1)
print(confusion_matrix(Y_test, predict_gnb))

predict_voting = eclf2.predict(X_test_scaled)
# y_score_voting = eclf2.fit(X_test_scaled,Y_test).decision_function(X_test_scaled)
print("Confusion matrix VotingClassifier")
voting_p = precision_score(Y_test, predict_voting)
print(voting_p)
voting_r = recall_score(Y_test, predict_voting)
print(voting_r)
voting_f1 = f1_score(Y_test, predict_voting)
print(voting_f1)
print(confusion_matrix(Y_test, predict_voting))

# ----------------------------------------------------------------------------
# Roc curve
fig , ax = plt.subplots(figsize= (10,10))
plt.plot([0, 1], [0, 1], linestyle='--')
lg_disp = plot_roc_curve(logistic_best,X_test_scaled,Y_test,ax=ax,alpha=0.8)
knn_disp = plot_roc_curve(knn_best,X_test_scaled,Y_test,ax=ax,alpha=0.8)
df_disp = plot_roc_curve(decisionTree_best,X_test_scaled,Y_test,ax=ax,alpha=0.8)
rfc_disp = plot_roc_curve(randomForest_best,X_test_scaled,Y_test,ax=ax,alpha=0.8)
gb_disp = plot_roc_curve(gradientBoosting_best,X_test_scaled,Y_test,ax=ax,alpha=0.8)
gnb_disp = plot_roc_curve(gaussianNB_best,X_test_scaled,Y_test,ax=ax,alpha=0.8)
vt_disp = plot_roc_curve(eclf2,X_test_scaled,Y_test,ax=ax,alpha=0.8)
plt.legend(fontsize = 18)
plt.title("Models Roc Curve" , fontsize= 25)
plt.show()
