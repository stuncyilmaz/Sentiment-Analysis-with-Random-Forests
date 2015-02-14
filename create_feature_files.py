import os
from function_tree_forest import *

    

current_path=os.path.dirname(os.path.abspath("__file__"))
my_path=current_path+"/myData/"

# create json files for test and train sets
if not os.path.exists(my_path+"test/"):
    os.makedirs(my_path+"/test")
if not os.path.exists(my_path+"train/"):
    os.makedirs(my_path+"/train")

cols=["PhraseId", "SentenceId","review"]
tsvToJSON(my_path+'test.tsv',my_path+'test/kaggle.json',cols)
cols=["PhraseId", "SentenceId","review","rating"]
tsvToJSON(my_path+'train.tsv',my_path+'train/kaggle.json',cols)

    
#have kaggle.json in folder "myData"
# create "test" and "train" folders
my_path_keys=['train','test']
my_paths=[my_path+"train/",my_path+"test/"]

# creating files to calculate features
print("creating features")
create_Train_Test_files(my_paths,my_path_keys)
print("feature files created,")