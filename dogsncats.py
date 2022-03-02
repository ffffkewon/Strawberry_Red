import os, shutil
#%% 경로 지정-최초 한번만 실행할 것
original_dataset_dir = 'D:/OneDrive - pusan.ac.kr/2.coding/datasets/dogs vs cats/train/train'

base_dir = 'D:/OneDrive - pusan.ac.kr/2.coding/datasets/dogs vs cats/dogs vs cats_small'
# os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir,'cats')
# os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
# os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats')
# os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'dogs')
# os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
# os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs')
# os.mkdir(test_dogs_dir)

# #%% load data
# #500개의 고양이 이미지를 train, validation, test용으로 나누기 (500개씩)
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(train_cats_dir,fname)
#     shutil.copyfile(src,dst)
    
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(validation_cats_dir,fname)
#     shutil.copyfile(src,dst)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(test_cats_dir,fname)
#     shutil.copyfile(src,dst)
    
# #500개의 개 이미지를 train, validation, test용으로 나누기 (500개씩)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(train_dogs_dir,fname)
#     shutil.copyfile(src,dst)
    
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(validation_dogs_dir,fname)
#     shutil.copyfile(src,dst)

# fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(test_dogs_dir,fname)
#     shutil.copyfile(src,dst)
    
# #%% load check
# print('훈련용 고양이 이미지 전체 개수: ', len(os.listdir(train_cats_dir)))
# print('훈련용 강아지 이미지 전체 개수: ', len(os.listdir(train_dogs_dir)))

# print('검증용 고양이 이미지 전체 개수: ', len(os.listdir(validation_cats_dir)))
# print('검증용 강아지 이미지 전체 개수: ', len(os.listdir(validation_dogs_dir)))

# print('테스트용 고양이 이미지 전체 개수: ', len(os.listdir(test_cats_dir)))
# print('테스트용 강아지 이미지 전체 개수: ', len(os.listdir(test_dogs_dir)))
#%% model gen
from keras import layers
from keras import models

catdog = models.Sequential()
catdog.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
catdog.add(layers.MaxPooling2D((2,2)))
catdog.add(layers.Conv2D(64,(3,3),activation='relu'))
catdog.add(layers.MaxPooling2D((2,2)))
catdog.add(layers.Conv2D(128,(3,3),activation='relu'))
catdog.add(layers.MaxPooling2D((2,2)))
catdog.add(layers.Conv2D(128,(3,3),activation='relu'))
catdog.add(layers.MaxPooling2D((2,2)))
catdog.add(layers.Flatten())
catdog.add(layers.Dropout(0.5))
catdog.add(layers.Dense(512,activation='relu'))
catdog.add(layers.Dense(1,activation='sigmoid'))

catdog.summary()

#%% optimizer
from tensorflow.keras import optimizers
catdog.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                loss = 'binary_crossentropy',
                metrics=['acc'])
#%% data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

train_datagen = IDG(rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,)
test_datagen = IDG(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')

#%% Train

history = catdog.fit_generator(train_generator,steps_per_epoch=100,epochs=100,validation_data=validation_generator,validation_steps=50)

catdog.save('cats_and_dogs_small_1.h5')

#%%
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss)+1)
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











