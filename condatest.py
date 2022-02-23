import time
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print("Train Image Data의 크기는: ",train_images.shape)
print("Train Image Data의 양식은: ",train_images.dtype)
#%% Check dataset
import random
import matplotlib.pyplot as plt
print("학습할 Dataset의 형태를 표시합니다 in 3 Seconds")
time.sleep(1)
for i in range(0,10):
    plt.imshow(train_images[random.randint(0,60000)],cmap=plt.cm.binary)
    plt.show()
    time.sleep(0.1)
#%%network
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

#%%compiler
from keras import optimizers
network.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])
#%%Data processing
train_images = train_images.reshape((60000,28*28)) 
train_images = train_images.astype('float32') /255 #unit to float + normalize

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') /255
#%%One hot encoding
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#%% Learning
network.fit(train_images,train_labels,epochs=5,batch_size=128)

#%% Validate
print("학습된 신경망을 검증합니다...")
time.sleep(5)
test_loss,test_acc= network.evaluate(test_images,test_labels)
print(test_loss, test_acc)


#%% 