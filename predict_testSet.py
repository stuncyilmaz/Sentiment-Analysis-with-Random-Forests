import numpy as np
import pandas as pd
import os
import cPickle
from sklearn.ensemble import RandomForestClassifier
from function_tree_forest import *


current_path=os.path.dirname(os.path.abspath("__file__"))
train_path=current_path+"/myData/train/"
test_path=current_path+"/myData/test/"

my_path_keys=['train']
my_paths=[train_path]

# make addWords_bool=True if you want the appearance of 
#individual words as features 
addWords_bool=False
all_results=create_Train_Test_data(my_paths,my_path_keys,addWords_bool=addWords_bool)
results=all_results[0]

df_train=results['train']
(X_train, y_train)=transform_pd(df_train, target_col='rating')
#### train random forest classifier
clf_forest = RandomForestClassifier(n_estimators=500, oob_score=False)
clf_forest.set_params(max_features=5)
clf_forest= clf_forest.fit(X_train, y_train)
print("in sample",clf_forest.score(X_train,y_train))

##### save the classifier
with open('forest_kaggle.pkl', 'wb') as fid:
    cPickle.dump(clf_forest, fid)  
####### save parameters to create the test features
names=["results","ratings_words","ratings_freq","ratings_tfidf"]
if addWords_bool:names.append("AllWords")
inJSON={}
for i in range( np.where(addWords_bool,5,4) ):
    inJSON[names[i]]=all_results[i]
   
with open('parameters_for_forest.pkl', 'wb') as fid:
    cPickle.dump(inJSON, fid)  
print("parameters done")
all_results=None


######### create test data
# make addWords_bool=True if you want the appearance of 
#individual words as features 

fileName="kaggle.json"
df_test=create_Test_data(test_path,inJSON,fileName,\
test_rating=False,addWords_bool=addWords_bool)
X_test=transform_pd(df_test)
inJSON=None
df_test=None
my_result=clf_forest.predict(X_test)

inputFName=test_path+'kaggle.json'
inJSON = json.load(open(inputFName, "r"))
y_all=json.load(open(inputFName, "r"))
y_all=pd.DataFrame(y_all)
submission=pd.DataFrame()
submission['PhraseId']=y_all['PhraseId']
submission['Sentiment']=my_result
submission.to_csv('my_kaggle_submission.csv',index=False)

