from keras import layers
from keras import models

model_no_max_pool = models.Sequential()
model_no_max_pool.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model_no_max_pool.add(layers.Conv2D(64,(3,3),activation='relu'))
model_no_max_pool.add(layers.Conv2D(64,(3,3),activation='relu'))


model_no_max_pool.summary()

#Pooling 사용하는 이유 : Down scaling을 통한 연산량 감