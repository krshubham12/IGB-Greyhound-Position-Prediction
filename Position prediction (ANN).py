import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = pd.read_csv('IGBfinaldata.csv')

data['Pos.'][data['Pos.']>3.]=4

og = OneHotEncoder(sparse=False)
d = data['Grade'].values.reshape(-1,1)
grade = og.fit_transform(d)
gr=pd.DataFrame(grade)
data = data.drop(['Grade'],axis=1)

ov = OneHotEncoder(sparse=False)
d = data['Venue'].values.reshape(-1,1)
venue = ov.fit_transform(d)
ve =pd.DataFrame(venue)
data = data.drop(['Venue'],axis=1)
data = pd.concat([data,gr,ve],axis=1)


X = data.drop(['Pos.'], axis=1).values
Y = data.iloc[:,0].values.reshape(-1,1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(Y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])

xtrain, xtest, ytrain, ytest = train_test_split(X , y , test_size=0.30 , random_state=5)

scaler = StandardScaler().fit(xtrain)
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)

model = Sequential()
model.add(Dense(64, input_shape=(42,), activation='relu', name='fc1'))
model.add(Dense(128, activation='relu', name='fc2'))
model.add(Dense(32, activation='sigmoid', name='fc3'))
model.add(Dense(4, activation='softmax', name = 'output'))
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print('Neural Network Model Summary: ')
print(model.summary())

model.fit(xtrain, ytrain, verbose=2, batch_size=100, epochs=100)

results = model.evaluate(xtest, ytest)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

# PREDICTION

input_data = ['CHAOTIC OAK', 40, 74, 30.17, 29.81, 82, 'A6', 525, 'CML', -0.36]

Greyhound = input_data[0]
Greyhound = le.transform([Greyhound])
#Grade
Grade = np.array([input_data[6]]).reshape(-1,1)
Grade = og.transform(Grade)
#Venue
Venue = np.array([input_data[8]]).reshape(-1,1)
Venue = ov.transform(Venue)

#'Greyhound','Prize','Wt.','Win Time','Est Time',SP.
Xnew = [Greyhound[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5]]
Xnew.extend(*Grade.tolist())
#Distance
Xnew.extend([input_data[7]])
Xnew.extend(*Venue.tolist())
#est-win
Xnew.extend([input_data[9]])
Xnew = np.array([Xnew])

Xnew= scaler.transform(Xnew)
ynew= model.predict(Xnew)
labels = np.argmax(ynew, axis = 1)
print("Predicted Position= %s" % (labels[0]))
