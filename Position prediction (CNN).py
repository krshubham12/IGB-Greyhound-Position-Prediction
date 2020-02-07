from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import pandas as pd
import numpy as np

data = pd.read_csv('IGBfinaldata.csv')

data.drop(["Greyhound"],axis=1,inplace=True)
colNames=list(data.columns)
for col in colNames:
    if( data[col].dtype == np.dtype('object')):
        dummies = pd.get_dummies(data[col],prefix=col)
        data = pd.concat([data,dummies],axis=1)
        #drop the encoded column
        data.drop([col],axis = 1 , inplace=True)

data['Pos.'][data['Pos.']>3.]=4
data.drop(['Venue_YGL','Venue_WFD','Venue_TRL','Venue_THR','Grade_H3'],axis = 1 , inplace=True)

xdata = data.drop(['Pos.'], axis=1).values
ydata = data.iloc[:,0].values.reshape(-1,1)
encoder = OneHotEncoder(sparse=False)
ydata = encoder.fit_transform(ydata)

xdata = xdata.reshape((106696,6,6,1))

x_train,x_test,y_train,y_test = train_test_split(xdata,ydata,test_size = 0.25)

mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)
x_train = (x_train - mean_px)/(std_px)

x_train = np.pad(x_train, ((0,0),(1,1),(1,1),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(1,1),(1,1),(0,0)), 'constant')

from keras.layers import Conv2D

model = Sequential()

model.add(Conv2D(filters = 3,
                 kernel_size = 2,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (8,8,1)))

model.add(MaxPooling2D(pool_size = 2, strides = 1))

model.add(Conv2D(filters = 16,
                 kernel_size = 2,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (6,6,3)))
model.add(MaxPooling2D(pool_size=2, strides=1))

model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 4, activation = 'softmax'))
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history=model.fit(x_train ,y_train, steps_per_epoch = 16, epochs = 20)
results = model.evaluate(x_test, y_test)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
y_pred = model.predict(x_test)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.ylabel('Value')
plt.xlabel('epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper left')
plt.show()
#Converting one hot vectors to labels
labels = np.argmax(y_pred, axis = 1)
index = np.arange(1, 26675)

labels = labels.reshape([len(labels),1])
index = index.reshape([len(index), 1])

final = np.concatenate([index, labels], axis = 1)
