from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
import math
import matplotlib.pylab as plt
import numpy as np

length = 100

seqX = np.zeros(length)
seqy = np.zeros(length)

for a in range(length):
    seqX[a] = (math.sin(a/6))
    seqy[a] = (math.sin((a+9)/6))

X = seqX.reshape(length, 1, 1)
y = seqy.reshape(length, 1)

n_neurons = length
n_batch = 5
n_epoch = 200

model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)

result = model.predict(X, batch_size=n_batch, verbose=0)
newGraph = list()
for value in result:
    newGraph.append(value)

plt.plot(newGraph)
plt.plot(seqX)
plt.plot(seqy)
plt.legend(["Prediction", "Test X", "Test y"])
plt.show()
