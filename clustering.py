import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('C:/Users/samsung/Desktop/hotel_bookings.csv') #데이터변경
df = df.drop('agent', axis=1)
df = df.drop('company', axis=1)

# dirty data preprocessing


# encoding
encoder = LabelEncoder()
for cols in df.columns:
    df[cols] = encoder.fit_transform(df[cols])

#scaling
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()