from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
microsoft = pd.read_csv(r'MicrosoftStock.csv')
print(microsoft.head())
microsoft.shape
microsoft.info()
microsoft.describe()
plt.plot(microsoft['date'], microsoft['open'], color="blue", label="open")
plt.plot(microsoft['date'], microsoft['close'], color="green", label="close")
plt.title("Microsoft Open-Close Stock")
plt.legend()
plt.plot(microsoft['date'], microsoft['volume'])
plt.show()

correlation_data = microsoft.select_dtypes(include=[np.number])
sns.heatmap(correlation_data.corr(), annot=True,cbar=False)
plt.show()
microsoft['date'] = pd.to_datetime(microsoft['date'])
prediction = microsoft.loc[(microsoft['date']
> datetime(2013, 1, 1))
& (microsoft['date']
< datetime(2018, 1, 1))]
plt.figure(figsize=(10, 10))
plt.plot(microsoft['date'], microsoft['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Microsoft Stock Prices")
msft_close = microsoft.filter(['close'])
dataset = msft_close.values
training = int(np.ceil(len(dataset) *0.95))
ss = StandardScaler()
ss = ss.fit_transform(dataset)
train_data = ss[0:int(training), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
 x_train.append(train_data[i-60:i, 0])
 y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train),\
 np.array(y_train)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape =(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
print(model.summary())
plt.show()
