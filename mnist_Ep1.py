#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf


# In[6]:


# import tensorflow as tf

mnist = tf.keras.datasets.mnist 

#28x28 images of handwritten digits unque image from 0 to 9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalise the data from 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#building the model
#building as a Sequential model. 2 types. Most common type. FeedForword.
model = tf.keras.models.Sequential()

#add input layer
# flattened takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Flatten())

#add a hidden layer
#simple fully-connected layer.
#128 units/neurones to the layer
#relu go to activation function.
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#add a hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#add an output layer
# for the number of classifications or numbers 0 to 9.
#softmax is a probablity distrubion.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#loss is like the degree of error.A NN doesn't try to maximize accuracy.
#The NN tries to rather minimize loss
#the method of calculating loss can have huge impact.
#It all depends on what is loss relationship to our accuracy.

#the optimizer is the most complicated part. adam is a bit of a default
#optimizer, but there are others.

#there are many ways to calculate loss. Such as binary for 'cats' and 'dogs'
#for example.
#categorical_crossentropy or some version of that,
# is a very popular way of calculating loss.

#metrics are the what metrics we want to track.
model.compile(optimizer='adam',
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

#with this we are now ready to train the model.


#To train we give what we want to get fit.
# epochs are the number of times the NN saw each unique sample.
model.fit(x_train, y_train, epochs=3)
#You don't want to over fit.
# Because the model can start memorising all samples.
#the hope is that the model actually generalised.
#You want it to be learning patterns and attributes
# of what is making a '4' for e.g


# In[7]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print("loss:", val_loss, "\naccuracy:", val_acc)
#to test test the model. It is okay if it doesn't have exactly the same values as long are they are in a reasonable delta


# In[8]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
print(x_train[0])


# In[9]:


#To save the model
# model.save("my_first_model")


# In[10]:


# To open model.
new_model = tf.keras.models.load_model('my_first_model')


# In[11]:


#To make a prediction on our saved model
predictions = new_model.predict([x_test])


# In[12]:


plt.imshow(x_test[0])
print(len(x_test))


# In[13]:


#to view the first prediction
import numpy as np

print(np.argmax(predictions[0]))


# In[26]:


import time
i = 34
print("prediction", str(i) + ":\n")
plt.imshow(x_test[i])
print(np.argmax(predictions[i]))


# In[ ]:




