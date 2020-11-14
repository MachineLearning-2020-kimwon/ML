import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')

# Load the data
# 데이터 null값 생성한 것으로 변경
df = pd.read_csv('C:/Users/samsung/Desktop/hotel_bookings.csv')
df = df.drop('agent', axis=1)
df = df.drop('company', axis=1)
print(df)

# dirty data preprocessing
# example) dirty data fill mean
df = df.fillna(df.mean())

# encoding
encoder = LabelEncoder()
for cols in df.columns:
    # LabelEncoder: TypeError, not supported between instances of 'float' and 'str'
    # 다양한 값들이 나올 수 있는 reservation_status_date을 encoding 하는게 의미가 있을까
    # classification에 이용할 column들을 미리 정의하고 encoding을 하는 것이 효율적일 것으로 판단
    df[cols] = encoder.fit_transform(df[cols].astype(str))
    df[cols] = encoder.fit_transform(df[cols])
print(df.head())

# split data
x = df.drop('is_canceled', axis=1)
y = df['is_canceled'].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33)

# scaling
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()

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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
