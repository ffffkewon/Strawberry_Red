from keras import layers
from keras import models
#%% model generating
cnnex = models.Sequential()
cnnex.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnnex.add(layers.MaxPooling2D((2,2)))
cnnex.add(layers.Conv2D(64,(3,3),activation='relu'))
cnnex.add(layers.MaxPooling2D((2,2)))
cnnex.add(layers.Conv2D(64,(3,3),activation='relu'))
cnnex.add(layers.Flatten())
cnnex.add(layers.Dense(64,activation='relu'))
cnnex.add(layers.Dense(10,activation='softmax'))

cnnex.summary()
#%% loading train data
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical# for one_hot encoding

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images = train_images.reshape((60000,28,28,1)) 
train_images = train_images.astype('float32') /255 #unit to float + normalize

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') /255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from tensorflow.keras import optimizers
cnnex.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

cnnex.fit(train_images,train_labels,epochs=5,batch_size=64)

#%% validation
test_loss, test_acc = cnnex.evaluate(test_images, test_labels)
print(test_acc)
