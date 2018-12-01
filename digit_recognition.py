
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers


# In[28]:


#load MINST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(test_images.shape)


# In[35]:


#build model
model = Sequential()
model.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[37]:


#process data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[38]:


#training
model.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[39]:


#test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('test_accuracy: ', test_accuracy)

