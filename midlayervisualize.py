from keras.models import load_model
model = load_model('cats_and_dogs_small_1.h5')
model.summary()


img_path = 'D:/OneDrive - pusan.ac.kr/2.coding/datasets/dogs vs cats/dogs vs cats_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]


plt.matshow(first_layer_activation[0,:,:,15],cmap='viridis')