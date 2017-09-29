import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time

from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, log_loss

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD, Adam
from keras.initializers import Zeros, glorot_normal
from keras import regularizers

np.random.seed(1)

# load the data
X_train = np.load('input/stage2_train_set.npy')
Y_train = np.load('input/stage2_encoded_y.npy')
X_test = np.load('input/stage2_test_set.npy')
test_index = np.load('input/stage2_test_index.npy')
timestr = time.strftime("%Y%m%d-%H%M%S")
dirname = 'output/'  # output directory
filename = ''

print('X_train: ', X_train.shape)
print('Y_train: ', Y_train.shape)


# bagging functions
def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :]
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


input_dim = 350
n_1 = input_dim
n_2 = 350
n_3 = 100
n_4 = 100
n_5 = 100
n_L = 9
beta = 0.05

def baseline_model():
    model = Sequential()
    model.add(Dense(n_1, input_dim=input_dim,
                    activation='relu',
                    kernel_initializer=glorot_normal(seed=1),
                    bias_initializer=Zeros(),
                    kernel_regularizer=regularizers.l2(beta)
                    ))
    model.add(Dropout(0.3))
    model.add(Dense(n_2,
                    activation='relu',
                    kernel_initializer=glorot_normal(seed=1),
                    bias_initializer=Zeros(),
                    kernel_regularizer=regularizers.l2(beta)
                    ))
    model.add(Dropout(0.5))
    model.add(Dense(n_3,
                    activation='relu',
                    kernel_initializer=glorot_normal(seed=1),
                    bias_initializer=Zeros(),
                    kernel_regularizer=regularizers.l2(beta)
                    ))
    # model.add(Dense(n_4,
    #                 activation='relu',
    #                 kernel_initializer=glorot_normal(seed=1),
    #                 bias_initializer=Zeros(),
    #                 kernel_regularizer=regularizers.l2(beta)
    #                 ))
    # model.add(Dense(n_5,
    #                 activation='relu',
    #                 kernel_initializer=glorot_normal(seed=1),
    #                 bias_initializer=Zeros(),
    #                 kernel_regularizer=regularizers.l2(beta)
    #                 ))
    model.add(Dense(n_L,
                    activation='softmax',
                    kernel_initializer=glorot_normal(seed=1),
                    bias_initializer=Zeros(),
                    kernel_regularizer=regularizers.l2(beta)
                    ))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model



## cv-folds
nfolds = 5
folds = KFold(n_splits = nfolds, shuffle = True, random_state = 111)
batch_size=128
batch_sizep=800

## train models
i = 0
nbags = 5
nepochs = 55
pred_oob = np.zeros((X_train.shape[0],Y_train.shape[1]))
pred_test = np.zeros((X_test.shape[0], Y_train.shape[1]))

for (inTr, inTe) in folds.split(X_train, Y_train):
    xtr = X_train[inTr]
    ytr = Y_train[inTr]
    xte = X_train[inTe]
    yte = Y_train[inTe]
    pred = np.zeros((yte.shape[0], yte.shape[1]))
    for j in range(nbags):
        model = baseline_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, batch_size, True),
                                  steps_per_epoch= xtr.shape[0]/batch_size,
                                  epochs = nepochs,
                                  verbose = 0)
        pred += model.predict_generator(generator = batch_generatorp(xte, batch_sizep, False), steps = xte.shape[0]/batch_sizep)[:,:]
        pred_test += model.predict_generator(generator = batch_generatorp(X_test, batch_sizep, False), steps = X_test.shape[0]/batch_sizep)[:,:]
    pred /= nbags
    pred_oob[inTe] = pred
    score = log_loss(yte, pred)
    i += 1
    print('Fold ', i, '- log_loss:', score)

print('Total - log_loss:', log_loss(Y_train, pred_oob))

## test predictions
pred_test /= (nfolds*nbags)
submission = pd.DataFrame(pred_test)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv(dirname + timestr + filename + 'kerasbagging.csv', index=False)

#estimator=model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2)

#print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))

'''
filename = timestr + '_NNkeras2stagebag_cost_{:.2f}-{:.2f}_acc_{:.2f}-{:.2f}'.format(
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
plt.close()

# summarize history for loss
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig(dirname + filename + '_COST.png')
plt.close()

prediction = model.predict_proba(X_test)
submission = pd.DataFrame(prediction)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv(dirname + filename + '.csv', index=False)
'''
