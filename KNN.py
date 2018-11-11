import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss

#load dataset
dataset = pd.read_csv("./train.csv")
features = dataset.loc[:,'feat_1':'feat_93'].values
labels = dataset['target'].values
test_set = pd.read_csv("./test.csv")
test_features = test_set.loc[:,'feat_1':'feat_93'].values

categories=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']

#transformer la label on one hot label
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

#parameter n_neighbors to be changed 10， 100， 300
neigh = KNeighborsClassifier(n_neighbors=1000)
neigh.fit(features, labels) 
print ("Making prediction....................")
preds=neigh.predict_proba(test_features)

sample = pd.read_csv('./sampleSubmission.csv')
# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('KNN_submission.csv', index_label='id')
