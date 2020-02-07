#https://datascienceplus.com/keras-regression-based-neural-networks/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('IGBfinaldata.csv')

data['Est Time'][data['Est Time']=='-']=0
data['Est Time']=pd.to_numeric(data['Est Time'])

X = data.drop(['By','Est Time'], axis=1).values
Y = data.iloc[:,9].values.reshape(-1,1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
li = [1,2,3,7,8,9,10,12,13]
for i in li:
    X[:,i] = le.fit_transform(X[:,i])

scaler = StandardScaler().fit(X)
xtrain=scaler.transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(X , Y , test_size=0.20 , random_state=5)

model = Sequential()
model.add(Dense(32, input_dim=14, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(xtrain, ytrain, epochs=150, batch_size=20,  verbose=1, validation_split=0.2)

results = model.evaluate(xtest, ytest)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#predicted value
Xnew = np.array([[4.0, 166, 51, 461, 0.0, 68, 18.13, 7, 16, 26, 245, 325, 1, 11]])# actual value:18.42
Xnew= scaler.transform(Xnew)
ynew= model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
