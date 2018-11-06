import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import log_loss


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

#
clf = MultinomialNB()
clf.fit(train_data, train_label)

print (log_loss(valid_label,clf.predict_proba(valid_data)))

#prediction with this NN
test_set = pd.read_csv("./test.csv")
test_features = test_set.loc[:,'feat_1':'feat_93'].as_matrix()
predictions = clf.predict_proba(test_features)

df = pd.DataFrame(predictions)
df.index = np.arange(1, len(df) + 1)
df = df.reset_index()
df.columns = ['id']+categories
df.to_csv("NBsubmission.csv",index=False)