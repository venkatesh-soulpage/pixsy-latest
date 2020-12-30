import dask.dataframe as dd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
import pickle

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data = dd.read_csv(r'pixsy_data_for_classification/*.part')
print('dataset loaded into dask')
data = data.drop(['Unnamed: 0'],axis='columns').compute()

### adding sim score as a feature ###
# sim_score = metrics.pairwise.cosine_similarity(
#     query_array.filter(like='photo', axis=1),
#     query_array.filter(like='match', axis=1))

photos = list(data['photo_id'].unique())
random.seed(2020)
sampled_photos = random.sample(photos,int(len(photos)*0.2))
test = data[data.photo_id.isin(sampled_photos)]
train = data[~data.photo_id.isin(sampled_photos)]

print('train test split done')
X_train1,X_test1,y_train,y_test = train.drop(['FP'],axis='columns'),test.drop(['FP'],axis='columns'),train.FP,test.FP

X_train = X_train1.drop(['photo_id','match_id'],axis='columns')
X_test = X_test1.drop(['photo_id','match_id'],axis='columns')
del X_train1,test,train,sampled_photos
X_test1 = X_test1[['photo_id','match_id']]

#### MLP ####
print('scaling features with RobustScaler')
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
mlp = MLPClassifier(alpha=0.0001, learning_rate_init=0.01,random_state=2020,verbose=True)

print('fitting multi layer perceptron...')
mlp.fit(X_train_scaled,y_train)

# Pickling to files
print('saving mlp model..')
model_filename = "pickled_mlp_model_new.pkl"
scaler_filename = 'pickled_scaler_new.pkl'
with open(model_filename, 'wb') as model_file, open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(mlp, model_file)
    pickle.dump(scaler,scaler_file)

# Load from files
with open(model_filename, 'rb') as model_file, open(scaler_filename, 'rb') as scaler_file:
    mlp = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

print('mlp fit done. Predicting...')
train_pred = mlp.predict(X_train_scaled)
print('scaling test features using RobustScaler')
X_test_scaled = scaler.fit_transform(X_test)
test_pred = mlp.predict(X_test_scaled)
probablities = mlp.predict_proba(X_test_scaled)

print('**train set**')
print("Accuracy:",metrics.accuracy_score(y_train, train_pred))
print("Precision:",metrics.precision_score(y_train, train_pred))
print("Recall:",metrics.recall_score(y_train, train_pred))

tn, fp, fn, tp = metrics.confusion_matrix(y_train, train_pred).ravel()
print('True Negatives:',tn)
print('False Positives:',fp)
print('False Negatives:',fn)
print('True Positives:',tp)

print('**test set**')
print("Accuracy:",metrics.accuracy_score(y_test, test_pred))
print("Precision:",metrics.precision_score(y_test, test_pred))
print("Recall:",metrics.recall_score(y_test, test_pred))

tn, fp, fn, tp = metrics.confusion_matrix(y_test, test_pred).ravel()
print('True Negatives:',tn)
print('False Positives:',fp)
print('False Negatives:',fn)
print('True Positives:',tp)


print('saving predictions to csv')
predictions = X_test1.copy()
predictions['FP_actual'] = y_test
predictions['FP_predicted'] = test_pred
predictions['probability'] = probablities[:,1]
print(predictions.head())

predictions.to_csv('mlp_classifier_predictions.csv',index=False)