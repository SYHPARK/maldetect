#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import ember
import csv


# import tensorflow as tf
# from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical


datadir = './data/ember2017_1/'


# In[ ]:


# create vectorized features
X_train_path = os.path.join(datadir, "X_train.dat")
y_train_path = os.path.join(datadir, "y_train.dat")
if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
    print("[*] Creating vectorized features")
    ember.create_vectorized_features(datadir, 1)


# In[ ]:


print("[*] training: read vectorized features")
x_train, y_train = ember.read_vectorized_features(datadir, "train", 1)


# In[ ]:


print("[*] testing: read vectorized features")
x_test, y_test = ember.read_vectorized_features(datadir, "test", 1)


# In[ ]:


train_rows = y_train != -1
print(train_rows.size)
print(train_rows[500000])

#filter training data by calling x_train[train_rows] and y_train[train_rows]


# In[ ]:


# train the model

n_inputs = 2351

print("[*] Building neural network")
model = Sequential()
model.add(Dense(70, input_shape=(n_inputs,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

print("[*] Compiling neural network")
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpath = os.path.join(os.path.dirname(os.path.abspath('')), "checkpoints")
os.makedirs(checkpath, exist_ok=True)
checkpath = os.path.join(checkpath, 'model-epoch{epoch:03d}-acc{val_acc:03f}.h5')

stopper = EarlyStopping(monitor = 'val_acc', min_delta=0.0001, patience = 5, mode = 'auto')

saver = ModelCheckpoint(checkpath, save_best_only=True, verbose=1, monitor='val_loss', mode='min')

print("[*] Training neural network...")
# train the model
#! error with validation_data shape..
fitted_model = model.fit(x_train[train_rows], y_train[train_rows],
          epochs=3,
          verbose=2, 
          validation_data=(x_test, y_test)
         )


# In[ ]:


y_binary = to_categorical(y_test)
print(y_binary.shape)


# In[ ]:


# EMBER model

params = {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "max_depth": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5
}

print("training lightGBM model")
lgbm_model = ember.train_model(datadir, params, 2)
lgbm_model.save_model(os.path.join(datadir, "model.txt"))


# In[ ]:




