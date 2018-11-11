import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import log_loss


#load dataset
# import data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
sample = pd.read_csv('./sampleSubmission.csv')

features = train.loc[:,'feat_1':'feat_93'].values
labels = train['target'].values

categories=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']

#transformer la label on one hot label
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)



# Train SVM by using different kernel 'rbf','poly','linear'
clf = SVC(C=1.0, kernel='sigmoid',probability=True)
clf.fit(features, integer_encoded)



#prediction 
test = pd.read_csv('./test.csv')
test_features = test.loc[:,'feat_1':'feat_93'].values
predictions = clf.predict_proba(test_features)

#print (predictions)
#enc = OneHotEncoder()
#onehot=enc.fit_transform(predictions.reshape(-1, 1)).toarray()

# create submission file
preds = pd.DataFrame(predictions, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('SVM_submission.csv', index_label='id')
