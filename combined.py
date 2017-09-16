import numpy as np
import pandas as pd
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
test_index = np.load('input/test_index.npy')
dirname = 'output/'

pred1 = pd.read_csv('output/20170914-191758_NN4L_lr_0.0001_beta_0.05_cost_1.59-1.75_acc_0.61-0.54.csv')
pred2 = pd.read_csv('output/20170916-210638_XGB_eta_0.2_depth_4_childw_1_sample_1_rounds_50.csv')

# average predictions from stacked model and xgb
prediction = pred1*0.25 + pred2*0.75

# name file
filename = timestr + '_NN4L_XGB'

# create submission file
submission = pd.DataFrame(prediction)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9',
                      'id']
submission.to_csv(dirname + filename + '.csv', index=False)
print('Success')
