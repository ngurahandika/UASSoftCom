#Nama : Gusti Ngurah Andika Martha Prabonia
#NIM : 1601020017
#Prodi : IF
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

dtset=pd.read_csv('Churn_Modelling.csv')

X=dtset.iloc[:, 3:-1].values
Y=dtset.iloc[: 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
lblencode_negara_x = LabelEncoder()
lblencode_gender_x = LabelEncoder()

X[:, 1] = lblencode_negara_x.fit_transform(X[:, 1])
X[:, 2] = lblencode_gender_x.fit_transform(X[:, 2])

ohenco=OneHotEncoder(categorical_features=[1])
X=ohenco.fit_transform(X).toarray()

X=X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sscl=StandardScaler()
X_train=sscl.fit_transform(X_train)
X_test=sscl.transform(X_test)

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
