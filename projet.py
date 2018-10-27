import pandas as pd 
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

#load dataset
dataset = pd.read_csv("./train.csv")
features = dataset['feat_1':'feat_93']
labels = dataset['target']

train_data, valid_data, train_label, valid_label = train_test_split(features, labels, test_size=0.2, random_state=65)


#build model
model = keras.Sequential([
    keras.layers.Dense(93, activation=tf.nn.relu),
    keras.layers.Dense(93, activation=tf.nn.relu),
    keras.layers.Dense(18, activation=tf.nn.relu),
    keras.layers.Dense(9, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model
model.fit(train_data, valid_label, epochs=5)

#test model
test_loss, test_acc = model.evaluate(valid_data, valid_label)

print (test_acc)

