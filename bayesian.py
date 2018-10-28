import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

def error_rate(predictions, labels):
    count=0
    for i in range(len(predictions)):
        if predictions[i]==labels[i]:
            count+=1
    print (count/len(predictions))


#load dataset
dataset = pd.read_csv("./train.csv")
features = dataset.loc[:,'feat_1':'feat_93'].as_matrix()
labels = dataset['target'].as_matrix()

categories=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']

#transformer la label on one hot label
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
#print(integer_encoded)

train_data, valid_data, train_label, valid_label = train_test_split(features, integer_encoded, test_size=0.2, random_state=65)

print (np.shape(train_data))
print (np.shape(train_label))

clf = MultinomialNB()
clf.fit(train_data, train_label)

error_rate(clf.predict(valid_data), valid_label)