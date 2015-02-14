import os
from function_tree_forest import *


def AllFilesExists(outputFilenames,directory):
    for myfile in outputFilenames:
        if myfile not in os.listdir(directory):
            return False
    return True
    
current_path=os.path.dirname(os.path.abspath("__file__"))
train_path=current_path+"/myData/train/"
test_path=current_path+"/myData/test/"

############# check if feature files exist ####################
############ create feature files if necessary ###############
fileName=current_path+"/create_feature_files.py"
outputFilenames=["splitWords.json","puncCount.json","remove_punctuation.json","avgWordLength.json",\
                     "isFirstPerson.json","posTag.json","negation.json","propNounConcat.json","lower_words.json",\
                     "removeStopWords.json","stemming.json"]


if (not AllFilesExists(outputFilenames,test_path)) or \
(not AllFilesExists(outputFilenames,train_path)):
    execfile(fileName)
    

############ predict test data ###############
fileName=current_path+"/predict_testSet.py"
execfile(fileName)
