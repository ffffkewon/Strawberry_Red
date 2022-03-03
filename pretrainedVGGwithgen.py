from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))

conv_base.summary()
conv_base.trainable = False
#%% conv trainable
# conv_base.trainable = True

# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False


#%%
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as IDG

base_dir = 'D:/OneDrive - pusan.ac.kr/2.coding/datasets/dogs vs cats/dogs vs cats_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = IDG(rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
test_datagen = IDG(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')
batch_size = 20

# def extract_features(directory,sample_count):
#     features = np.zeros(shape=(sample_count,4,4,512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory,
#         target_size=(150,150),
#         batch_size=batch_size,
#         class_mode='binary')
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i*batch_size : (i+1)*batch_size] = features_batch
#         labels[i*batch_size : (i+1) * batch_size] = labels_batch
#         i += 1
#         if i*batch_size >= sample_count:
#             break
#     return features, labels

# train_features, train_labels = extract_features(train_dir,2000)
# validation_features, validation_labels = extract_features(validation_dir,1000)
# test_features, test_labels = extract_features(test_dir,1000)

# train_features = np.reshape(train_features,(2000,4*4*512))
# validation_features = np.reshape(validation_features,(1000,4*4*512))
# test_features = np.reshape(test_features,(1000,4*4*512))

#%%
from keras import layers
from keras import models
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.00002),
                loss = 'binary_crossentropy',
                metrics=['acc'])

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50,verbose=2)


#%%
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']

epochs = range(1,len(acc)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and VValidation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training Acc')
plt.plot(epochs,val_acc,'b',label='Validation Acc')
plt.title('Training and VValidation Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()