#IMPORTS FOR PROJECT

# Import pandas
import pandas as pd

# Import matplotlib
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

from keras.callbacks import ModelCheckpoint


########################################################################################################

# Read in white wine data
#white = pd.read_csv("C:\\Users\\Daniel\\Documents\\Unal\\EjemploKeras\\winequality-white.csv", sep=';')
white = pd.read_csv("C:\\Users\\Daniel\\Documents\\Unal\\EjemploKeras\\positivo.csv", sep=';')


# Read in red wine data
#red = pd.read_csv("C:\\Users\\Daniel\\Documents\\Unal\\EjemploKeras\\winequality-red.csv", sep=';')
red = pd.read_csv("C:\\Users\\Daniel\\Documents\\Unal\\EjemploKeras\\negativo.csv", sep=';')


# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

print("Wines: ")
print( wines )

# Specify the data
X=wines.ix[:,0:11]

print("X: ")
print( X )

# Specify the target labels and flatten the array
y= np.ravel(wines.type)
print("y: ")
print(y )

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train: ")
print( X_train )

print("X_test: ")
print( X_test )

print("y_train: ")
print( y_train )

print("y_test: ")
print( y_test )


# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))
#model.add(Dense(12, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))

#Compilar y Ajustar el modelo a los datos.
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])


#save the model weights when the model improve the validation accuracy
checkpoint = ModelCheckpoint('weights_mnist.h5',
                             monitor = 'val_acc',
                             save_best_only = True,
                             mode = 'max',
                             save_weights_only = True,
                             verbose = 0)

model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, validation_data=(X_test, y_test),
          callbacks = [checkpoint])


#Cargar el modelo ya entrenado
#model.load_weights('weights_mnist.h5')

#Hacer predicciones para el conjunto de pruebas
y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    y_pred[i][0]=round(y_pred[i][0])

y_pred = np.array(y_pred).astype(np.int32)

print("prediccion: ", y_pred[:5])

print("test :", y_test[:5])


#EVALUATE MODEL
score = model.evaluate(X_test, y_test,verbose=1)
print("score: ",score)

# Confusion matrix
print("confusion_matrix: ",confusion_matrix(y_test, y_pred))


# Precision
print("precision_score: ",precision_score(y_test, y_pred))


# Recall
print("recall_score): ",recall_score(y_test, y_pred))


# F1 score
print("f1_score(): ",f1_score(y_test,y_pred))


# Cohen's kappa
print("cohen_kappa_score: ",cohen_kappa_score(y_test, y_pred))