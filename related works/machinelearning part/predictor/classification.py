from sklearn import neural_network
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

trainset = pd.read_csv('crime_all_2012_2015.txt', names = ["latitude", "longitude", "x_coordinate", "y_coordinate"
                                                  , "crime_day", "crime_month", "crime_day_of_week"
                                                  , "crime_season", "crime_type_group_id", "crime_type_group_1"
                                                  , "crime_type_group_2", "crime_type_group_3", "crime_type_group_4"
                                                  , "crime_type_group_5", "crime_type_group_6"], sep = '\t')
X_train = trainset[['latitude','longitude',"crime_day", "crime_month", "crime_day_of_week", "crime_season"]]
y_train= trainset[['crime_type_group_1','crime_type_group_2','crime_type_group_3','crime_type_group_4'
          ,'crime_type_group_5','crime_type_group_6']]

testset = pd.read_csv('crime_all_2016.csv', names = ["latitude", "longitude", "x_coordinate", "y_coordinate"
                                                  , "crime_day", "crime_month", "crime_day_of_week"
                                                  , "crime_season", "crime_type_group_id", "crime_type_group_1"
                                                  , "crime_type_group_2", "crime_type_group_3", "crime_type_group_4"
                                                  , "crime_type_group_5", "crime_type_group_6"])
X_test = testset[['latitude','longitude',"crime_day", "crime_month", "crime_day_of_week", "crime_season"]]
y_test = testset[['crime_type_group_1','crime_type_group_2','crime_type_group_3','crime_type_group_4'
          ,'crime_type_group_5','crime_type_group_6']]

#X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
#save scaler as pkl file
joblib.dump(scaler, 'scaler.pkl')

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(20,20), alpha=0.01)
mlp.fit(X_train,y_train)
#save mlp as pkl file
joblib.dump(mlp, 'mlp.pkl')

predictions = mlp.predict(X_test)
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))

#confusion & report work only 1 target, need to specify
#print(confusion_matrix(y_test['crime_type_group_1'],predictions[:,0]))
#access test and predicted result
accuracy_score(y_test['crime_type_group_1'],predictions[:,0])
accuracy_score(y_test['crime_type_group_2'],predictions[:,1])
accuracy_score(y_test['crime_type_group_3'],predictions[:,2])
accuracy_score(y_test['crime_type_group_4'],predictions[:,3])
accuracy_score(y_test['crime_type_group_5'],predictions[:,4])
accuracy_score(y_test['crime_type_group_6'],predictions[:,5])



#set pipeline process, normalised then estimate
#pipe = make_pipeline(StandardScaler(), mlp)
#scores = cross_val_score(pipe, X, y, cv = 3)

## Model Selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
X_train.shape
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train['crime_type_group_1'])
model = SelectFromModel(lsvc , prefit=True)
X_new = model.transform(X_train)
X_new.shape

print(confusion_matrix(y_test['crime_type_group_6'],predictions[:,5]))
print(confusion_matrix(y_test['crime_type_group_5'],predictions[:,4]))
print(confusion_matrix(y_test['crime_type_group_4'],predictions[:,3]))
print(confusion_matrix(y_test['crime_type_group_3'],predictions[:,2]))
print(confusion_matrix(y_test['crime_type_group_2'],predictions[:,1]))
print(confusion_matrix(y_test['crime_type_group_1'],predictions[:,0]))

print(classification_report(y_test['crime_type_group_6'],predictions[:,5]))
print(classification_report(y_test['crime_type_group_5'],predictions[:,4]))
print(classification_report(y_test['crime_type_group_4'],predictions[:,3]))
print(classification_report(y_test['crime_type_group_3'],predictions[:,2]))
print(classification_report(y_test['crime_type_group_2'],predictions[:,1]))
print(classification_report(y_test['crime_type_group_1'],predictions[:,0]))

