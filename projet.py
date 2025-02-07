import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras


#load dataset
dataset = pd.read_csv("./train.csv")
features = dataset.loc[:,'feat_1':'feat_93'].as_matrix()
labels = dataset['target'].as_matrix()

categories=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']

#transformer la label on one hot label
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
#print(integer_encoded)

#onehot_encoder = OneHotEncoder(categories='auto',sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print (onehot_encoded)



train_data, valid_data, train_label, valid_label = train_test_split(features, integer_encoded, test_size=0.2, random_state=65)

print (np.shape(train_data))
print (np.shape(train_label))

#build model
model = keras.Sequential([
    keras.layers.Dense(93, activation=tf.nn.relu),
    keras.layers.Dense(93, activation=tf.nn.relu),
    keras.layers.Dense(27, activation=tf.nn.relu),
    keras.layers.Dense(9, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model


model.fit(train_data, train_label, epochs=12)

#test model
test_loss, test_acc = model.evaluate(valid_data, valid_label)

print (test_acc)

#prediction with this NN
test_set = pd.read_csv("./test.csv")
test_features = test_set.loc[:,'feat_1':'feat_93'].as_matrix()
predictions = model.predict(test_features)

pred_with_id = np.concatenate((test_set.id.values.reshape([144368,1]),predictions),axis=1)
#Cast float to int to meet submission requirements
pred_with_id = pred_with_id.astype(int)

#Generate submission data frame from numpy array
submission = pd.DataFrame(data=pred_with_id,    # values
              columns=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
#Save submission as csv, ready to submit
submission.to_csv("./submission.csv",index=False)

