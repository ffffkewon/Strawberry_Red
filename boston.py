# import tensorflow as tf

# gpu = tf.config.experimental.list_physical_devices('GPU') # 내 컴에 장착된 GPU를 list로 반환
# try:
#     tf.config.experimental.set_memory_growth(gpu[0], True) # GPU Memory Growth를 Enable
# except RuntimeError as e:
#     print(e) # Error 발생하면 Error 내용 출력


#%% Loading Data
from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()

#%% Data processing
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
#%% Modeling
from keras import models
from keras import layers

def build_model():
    review = models.Sequential()
    review.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    review.add(layers.Dense(64,activation='relu'))
    review.add(layers.Dense(1))
    review.compile(optimizer='rmsprop',
                   loss='mse',
                   metrics=['mae'])
    return review

#%% K겹 검증
import numpy as np

k=4

num_val_samples = len(train_data) // k
# num_epochs = 100
# all_scores = []

# for i in range(k):
#     print('처리중인 폴드 #',i)
#     val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
#     val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    
#     partial_train_data = np.concatenate(
#         [train_data[:i*num_val_samples],
#          train_data[(i+1)*num_val_samples:]],
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[:i*num_val_samples],
#          train_targets[(i+1)*num_val_samples:]],
#         axis=0)
    
#     review = build_model()
#     review.fit(partial_train_data,partial_train_targets,epochs=num_epochs, batch_size = 1, verbose = 0)
#     val_mse, val_mae = review.evaluate(val_data,val_targets, verbose=0)
#     all_scores.append(val_mae)

#%%
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('처리중인 폴드 #',i)
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)
    
    review = build_model()
    history = review.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=num_epochs, batch_size = 1, verbose = 0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
#%%
average_mae_history = [ np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt

# plt.plot(range(1,len(average_mae_history)+1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

#%%
def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point * (1-factor))
        else:
                smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
