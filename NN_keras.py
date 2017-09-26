import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time

from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD
from keras import regularizers

np.random.seed(1)

# load the data
X_train = np.load('input/train_set.npy')
Y_train = np.load('input/encoded_y.npy')
X_test = np.load('input/test_set.npy')
test_index = np.load('input/test_index.npy')
timestr = time.strftime("%Y%m%d-%H%M%S")
dirname = 'output/'  # output directory
filename = ''

print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)


input_dim = 350
n_1 = input_dim
n_2 = 350
n_3 = 300
n_4 = 30
n_L = 9


def baseline_model():
    model = Sequential()
    model.add(Dense(n_1, input_dim=input_dim, init='normal', activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.3))
    model.add(Dense(n_2, init='normal', activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(n_3, init='normal', activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(n_4, init='normal', activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(n_L, init='normal', activation="softmax", kernel_regularizer=regularizers.l2(0.1)))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


model = baseline_model()
model.summary()

estimator=model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2)

print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))

filename = timestr + '_cost_{:.2f}-{:.2f}_acc_{:.2f}-{:.2f}'.format(
    estimator.history['loss'][-1], estimator.history['val_loss'][-1],
    estimator.history['acc'][-1], estimator.history['val_acc'][-1])

# summarize history for accuracy
plt.plot(estimator.history['acc'])
plt.plot(estimator.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(dirname + filename + '_ACC.png')
plt.show()

# summarize history for loss
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(dirname + filename + '_COST.png')
plt.show()


prediction = model.predict_proba(X_test)
submission = pd.DataFrame(prediction)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv(dirname + filename + '.csv', index=False)
