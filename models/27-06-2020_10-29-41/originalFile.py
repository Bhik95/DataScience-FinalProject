#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


train_path = 'fruits-360/Training'
test_path = 'fruits-360/Test'
classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato not Ripened', 'Tomato Yellow', 'Walnut', 'Watermelon']


# In[3]:


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(100,100), classes=classes, batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(100,100), classes=classes, batch_size=10)


# In[4]:


images, labels = next(train_batches)


# # Build and train CNN

# In[5]:


model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
    Conv2D(16, (3,3), activation='relu', input_shape=(50,50,)),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
    Conv2D(32, (3,3), activation='relu', input_shape=(25,25,)),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
    Conv2D(64, (3,3), activation='relu', input_shape=(9,9,)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(131, activation='softmax'),
])


# In[6]:


model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[7]:


# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
iter_count=0


# In[28]:


if not os.path.exists("models/"+dt_string):
    os.mkdir("models/"+dt_string)
for i in range(1,3+1):
    iter_count+=1
    print("Iteration "+str(iter_count))
    model.fit(x=train_batches, validation_data=test_batches, epochs=5, verbose=2)
    filename_output = 'fruit_'+dt_string+'('+str(iter_count)+').h5'
    model.save("models/"+dt_string+"/"+filename_output)
    print("Saved model as "+filename_output)


# # Prediction

# In[22]:


test_imgs, test_labels = next(test_batches)


# In[23]:


test_labels = np.argmax(test_labels, axis=-1)
test_labels


# In[24]:


predictions = np.argmax(model.predict(x=test_imgs, steps=1, verbose=0), axis=-1)


# In[25]:


predictions


# In[26]:


cm = confusion_matrix(test_labels, predictions)


# In[27]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion Matrix, without normalization")
        
    print(cm)
    
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')


# In[15]:


plot_confusion_matrix(cm, classes)


# In[ ]:





# In[ ]:





# In[ ]:




