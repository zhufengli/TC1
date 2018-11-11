# coding: utf-8

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
#from keras.utils import plot_model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#load dataset
dataset = pd.read_csv("./train.csv")
features = dataset.loc[:,'feat_1':'feat_93'].values
labels = dataset['target'].values

categories=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']

#transformer la label on one hot label
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)




train_data, valid_data, train_label, valid_label = train_test_split(features, integer_encoded, test_size=0.05, random_state=65)

print (np.shape(train_data))
print (np.shape(train_label))

#build model #372s-186s-93-46-9 10 epo
model = keras.Sequential([
    keras.layers.Dense(372, activation=tf.nn.sigmoid,input_dim=93,use_bias=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(186, activation=tf.nn.sigmoid,use_bias=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(93, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(46, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(9, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Draw model
#keras.utils.plot_modelplot_model(model, to_file='./model.png', show_shapes=True)

#train model

#model.fit(features, integer_encoded, epochs=60, batch_size=256)
model.fit(train_data, train_label, epochs=200, batch_size=256, validation_data=(valid_data, valid_label))




#prediction with this NN
test_set = pd.read_csv("./test.csv",sep=',')
test_features = test_set.loc[:,'feat_1':'feat_93'].as_matrix()
predictions = model.predict(test_features)

#print (predictions[0:3])

#print (predictions[0:3])
#predictions = predictions.tolist()
#onehot_predictions = np.zeros((144368,9))
#for i in range(len(predictions)):
#    onehot_predictions[i,predictions[i].index(np.max(predictions[i]))]=1





# In[96]:


class_list = ['id','Class_1','Class_2','Class_3','Class_4','Class_5',
             'Class_6','Class_7','Class_8','Class_9']
class_list2 = ['Class_1','Class_2','Class_3','Class_4','Class_5',
             'Class_6','Class_7','Class_8','Class_9']
d = pd.DataFrame(0, index=np.arange(144368), columns=class_list)
d['id'] = test_set['id']
d[class_list2] = d[class_list2].astype('float')
d.loc[:,1:] = predictions
d.head()
d.to_csv("./submission.csv",index=False)

