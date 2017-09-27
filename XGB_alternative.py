import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import xgboost as xgb

np.random.seed(1)

# load the data
X_train_orig = np.load('input/stage2_train_set.npy')
Y_train_orig = np.load('input/stage2_encoded_y.npy')
Y_train_orig = np.argmax(Y_train_orig, axis=1) # xgboost needs the classes labeled as int from 0 to num_classes
X_test = np.load('input/stage2_test_set.npy')
test_index = np.load('input/stage2_test_index.npy')
timestr = time.strftime("%Y%m%d-%H%M%S")
dirname = 'output/'  # output directory
filename = ''

# Use custom train-test split
# X_train, X_val, Y_train, Y_val = train_test_split(X_train_orig, Y_train_orig, test_size=0.20, random_state=42)
# X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T
# X_test = X_test.T

X_train = X_train_orig
Y_train = Y_train_orig

print('X_train: ', X_train.shape)
# print('X_val: ', X_val.shape)
print('Y_train: ', Y_train.shape)
# print('Y_val: ', Y_val.shape)

params = {
    'eta': 0.2,
    'max_depth': 4,
    'min_child_weight': 1,
    #'gamma': 0.1,
    'subsample': 1,
    'objective': 'multi:softprob',
    'num_class' : 9,
    'eval_metric': ['mlogloss', 'merror'],
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test)


def run_xgb(dtrain, dtest=None, params=params, cv=False):

    num_boost_rounds = 50
    t0 = time.time()

    if not cv:
        # train model
        model = xgb.train(params, dtrain, num_boost_round=num_boost_rounds)
        print('Finished training model in {:.1f}'.format(time.time()-t0))

        # make predictions
        prediction = model.predict(dtest)

        # name file
        filename = timestr + \
                   '_XGB2ndstage_eta_{}_depth_{}_childw_{}_sample_{}_rounds_{}'.format(
                       params['eta'], params['max_depth'], params['min_child_weight'], params['subsample'],
                       num_boost_rounds)

        # create submission file
        submission = pd.DataFrame(prediction)
        submission['id'] = test_index
        submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9',
                              'id']
        submission.to_csv(dirname + filename + '.csv', index=False)
        print('Success')

    else:
        print('Cross validating')
        # watchlist  = [(dtest,'test'), (dtrain,'train')]
        num_folds = 5
        modelcv = xgb.cv(params, dtrain, num_boost_rounds, num_folds)#, early_stopping_rounds=5)
        print("Finished cross validation in {:.1f} s".format(time.time() - t0))

        train_cost = modelcv['train-mlogloss-mean'][num_boost_rounds-1]
        test_cost = modelcv['test-mlogloss-mean'][num_boost_rounds-1]
        train_accuracy = 1 - modelcv['train-merror-mean'][num_boost_rounds-1]
        test_accuracy = 1 - modelcv['test-merror-mean'][num_boost_rounds-1]

        print("Train Cost:", train_cost)
        print("Test Cost:", test_cost)
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        # name file
        filename = timestr + \
                   '_XGB2ndstage_eta_{}_depth_{}_childw_{}_sample_{}_rounds_{}_cost_{:.2f}-{:.2f}_acc_{:.2f}-{' \
                   ':.2f}'.format(
                       params['eta'], params['max_depth'], params['min_child_weight'], params['subsample'],
                       num_boost_rounds, train_cost, test_cost, train_accuracy, test_accuracy)

        # plot costs
        modelcv[['test-mlogloss-mean', 'train-mlogloss-mean']].plot()
        plt.ylabel('cost')
        plt.xlabel('num_boost_rounds')
        plt.title(timestr + '_XGB_eta_{}_depth_{}_child_{}'.format(params['eta'], params['max_depth'],
                                                                params['min_child_weight']))
        plt.savefig(dirname + filename + '.png')

        print('Success')


run_xgb(dtrain, dtest)
#run_xgb(dtrain, cv=True)
