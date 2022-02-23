#%% Loading Data
from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)#훈련데이터에서 가장 자주 나타나는 단어 1만개만 사용하겠다
#%% +@ Index to Word
# word_index = imdb.get_word_index()#word_index는 단어와 정수 인덱스를 매핑한 dictionary
# reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) # 정수 인덱스랑 단어 뒤집기
# decoded_reviews = ''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])#3, ?는 padding, 사전에 없음, 문서 시작 제외
# print(decoded_reviews)
#%% Data processing
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#%% Modeling
from keras import models
from keras import layers

review = models.Sequential()
review.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
review.add(layers.Dense(16,activation='relu'))
review.add(layers.Dense(16,activation='relu'))
review.add(layers.Dense(1,activation='sigmoid'))

#%% Optimizer_Basic
review.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['accuracy'])
#optimizer : 최적화 방식 (SGD, RMSprop(Momentum을 사용하는 SGD의 일종),...)
#loss : Loss Fn
#metrics : 측정 지표. 여기서 Accuracy는 label과 신경망의 answer간의 정확도

#%% Optimizer_Custum
# from keras import optimizers
# from keras import losses
# from keras import metrics

# review.compile(optimizer=optimizers.RMSprop(lr=0.001),
#                loss=losses.binary_crossentropy,
#                metrics=[metrics.binary_accuracy])

#%% Validate

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = review.fit(partial_x_train,partial_y_train,
                    epochs=20,
                    batch_size = 512,
                    validation_data=(x_val,y_val))

#%% Visualize
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
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training Acc')
plt.plot(epochs,val_acc,'b',label='Validation Acc')
plt.title('Training and VValidation Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()

#%% Re-validate
review.fit(x_train,y_train,
                    epochs=4,
                    batch_size = 512,
                    )
results = review.evaluate(x_test,y_test)
print(results)
