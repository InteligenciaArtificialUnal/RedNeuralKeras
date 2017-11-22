from sknn.mlp import Regressor, Layer
import numpy as np


input_entrenamiento = np.array([[1, 1],[0,0],[1.1,1.15],[1.6,1.8]])

output_entrenamiento = np.array([1,0,2,4])

weights = [0.8,0,0.5,0,1]

nn = Regressor(
    layers=[
        Layer("Rectifier", units=50),
        Layer("Linear")],
    learning_rate=0.002,
    n_iter=5000)


nn.fit(input_entrenamiento, output_entrenamiento)

param = nn.get_parameters()

input_test = np.array([[1.1,1.15]])
output_test = nn.predict(input_test)
output_test



print "input entrenamiento"
print input_entrenamiento

print "output entrenamiento"
print output_entrenamiento


print "nn"
print nn

print "param"
print param


print "input test"
print input_test

print "output test"
print output_test