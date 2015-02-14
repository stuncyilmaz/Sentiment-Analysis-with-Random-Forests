import operator
import json
import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
import random
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# convert tsv file to json
def tsvToJSON(inputFName,outputFName,cols):
    """
    :param fName: name of tsv file
    :return: JSONified string of tsv file
    """

    tsvData = pd.read_table(inputFName, header=0)
    tsvData.columns = cols
    outJSON = tsvData.to_json(orient='records')
    outJSON=json.loads(outJSON)
    jsonOut = open(outputFName, "w")
    json.dump(outJSON, jsonOut)

############################################################################################################
############################################## Functions: Data Transform ###################################
############################################################################################################
def transform_pd(data, target_col=None, ignore_cols=None):

    if ignore_cols==None: ignore_cols=[]

    df_feature_names=[i for i in data.columns.values if not i in ([target_col]+list(ignore_cols))]
    df_data=data[df_feature_names]
     
    X=np.asarray(df_data)
    if not target_col==None:
        y=np.asarray(data[target_col])
        return X, y
    return X


############################################################################################################
############################################## Functions: Prepare Data Files ###############################
############################################################################################################


def split_train_test(inputFName,outputFName,train_path,test_path,test_percentage=0.15,shuffle=False):
    inJSON = json.load(open(inputFName, "r"))
    if shuffle:
        random.shuffle(inJSON)
        
        
    test_len=int(len(inJSON)*test_percentage)
    train_len=int(len(inJSON)-test_len)

    
    jsonOut = open(test_path+outputFName, "w")
    json.dump(inJSON[:test_len], jsonOut)
    jsonOut.close()
    jsonOut = open(train_path+outputFName, "w")
    json.dump(inJSON[test_len:], jsonOut)
    jsonOut.close()
    

    
def lower_words(inputFName,outputFName):
    '''
    removes punctuation from at the end and beginning of words
    split_clean_words
    '''
    inJSON = json.load(open(inputFName, "r"))
    my_lower_words=[]
    for entry in inJSON:
        my_lower_words.append({'words':[elt.lower() for elt in entry['words']]})
        
    with open(outputFName, 'wb') as outfile:
        json.dump(my_lower_words, outfile)

    
def remove_punctuation(inputFName,outputFName):
    '''
    removes punctuation from at the end and beginning of words
    split_clean_words
    '''
    inJSON = json.load(open(inputFName, "r"))
    my_remove_punctuation=[]
    for entry in inJSON:
        dummy={}
        exclude = "?:!.,;.-_`~+=#()/\|][*' "
        dummy['words']=entry['words']
        dummy['words']=[i.strip(exclude) for i in entry['words'] if i.strip(exclude) !=[] ]
        dummy2={}
        dummy2['words']=[]
        # remove only-exclude or entries
        for s in dummy['words']:
            s = ''.join(ch for ch in s if ch not in exclude)
            if not (s=='' or s=='s'):
                dummy2['words'].append(s)
        my_remove_punctuation.append(dummy2)
        
    with open(outputFName, 'wb') as outfile:
        json.dump(my_remove_punctuation, outfile)
    
def set_of_AllWords(inputFName,outputFName):
    inJSON = json.load(open(inputFName, "r"))
    s=set([])
    for i in range(len(inJSON)):
        t=set(inJSON[i]['words'])
        s=s.union(t)
        
    with open(outputFName, 'wb') as outfile:
        json.dump(list(s), outfile)


def splitWords(inputFName, outputFName):
    """
    :param fName: name of raw JSON file
    :return: JSON string w/ "words": [word, word, word]
    """
    inJSON = json.load(open(inputFName, "r"))
    split_Words=[]
    for entry in inJSON:
        entry["words"] = entry["review"].split(" ")
        split_Words.append( {'words':entry["words"]})
    jsonOut = open(outputFName, "w")
    json.dump(split_Words, jsonOut)
    jsonOut.close()  

def avgWordLength(inputFName, outputFName):
    """
    :param fName: name of JSON file w/ word splits
    :return: JSON string w/ "avgWordLength" and "numWords"
    """
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    avgWord=[]
    for entry in inJSON:
        dummy={}
        totalLength = 0
        numWords = 0
        for word in entry["words"]:
            if len(word) > 1:
                totalLength += len(word)
                numWords += 1
        dummy["numWords"] = numWords
        if numWords != 0:
            dummy["avgWordLength"] = round(totalLength/float(numWords), 3)
        else:
            dummy["avgWordLength"] = 0
        avgWord.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(avgWord, outJSON)
    outJSON.close()
def puncCount(inputFName, outputFName):
    """
    :param fName: name of JSON file w/ word splits
    :return: JSON string w/ "innerPunctuation", "exclamationPoints", "numQuestMarks"
    """
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    punc_Count=[]
    for entry in inJSON:
        dummy={}
        numInnerPunc = 0
        numExclamation = 0
        numQuestion = 0
        for word in entry["words"]:
            if re.match(r'[",", ";", ".", ":"]+', word):
                numInnerPunc += 1
            if re.match(r'["?"]+', word):
                numExclamation += 1
            if re.match(r'["!"]+', word):
                numQuestion += 1
        dummy["innerPunctuation"] = numInnerPunc
        dummy["exclamationPoints"] = numExclamation
        dummy["numQuestMarks"] = numQuestion
        punc_Count.append(dummy)

    outJSON = open(outputFName, "w")
    json.dump(punc_Count, outJSON)
    outJSON.close()

def isFirstPerson(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    num_FirstPerson=[]
    for entry in inJSON:
        dummy={}
        numFirstPerson = 0
        for word in entry["words"]:
            if word == "I":
                numFirstPerson = 1
        dummy["isFirstPerson"] = numFirstPerson
        num_FirstPerson.append(dummy)

    outJSON = open(outputFName, "w")
    json.dump(num_FirstPerson, outJSON)
    outJSON.close()
def posTag(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    i = 0
    pos_Tag=[]
    for entry in inJSON:
        dummy={}
        numPropNoun = 0
        numOtherNoun = 0
        numPronouns = 0
        numConj = 0
        numPresVerb = 0
        numPastVerb = 0
        numParticiple = 0
        numAdj = 0
        numDet = 0
        for word in entry["words"]:
            pos = nltk.pos_tag([word])
            pos = pos[0][1]
            if re.match('NNP', pos):
                numPropNoun += 1
            elif re.match('NN.*', pos):
                numOtherNoun += 1
            elif re.match('VBD', pos):
                numPastVerb += 1
            elif re.match('VBG', pos):
                numParticiple += 1
            elif re.match('VB[Z,P]', pos):
                numPresVerb += 1
            elif re.match('[W,PR]P', pos):
                numPronouns += 1
            elif re.match('CC', pos):
                numConj += 1
            elif re.match('JJ', pos):
                numAdj += 1
            elif re.match('DT', pos):
                numDet += 1
        dummy["numPropNoun"] = numPropNoun
        dummy["numOtherNoun"] = numOtherNoun
        dummy["numPronouns"] = numPronouns
        dummy["numConj"] = numConj
        dummy["numPresVerb"] = numPresVerb
        dummy["numPastVerb"] = numPastVerb
        dummy["numParticiple"] = numParticiple
        dummy["numAdj"] = numAdj
        dummy["numDet"] = numDet
        pos_Tag.append(dummy)
        i += 1


    outJSON = open(outputFName, "w")
    json.dump(pos_Tag, outJSON)
    outJSON.close()

def negation(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    numProcessed = 0
    stop = stopwords.words('english')
    mynegation=[]
    for entry in inJSON:
        modifier = None
        negativeTerritory = 0

        for j in range(len(entry["words"])):
            word = entry["words"][j]
            if word in ["not", "n't","hardly"]:
                modifier = "vrbAdj"
                negativeTerritory = 4
            elif word in ["no", "none"]:
                modifier = "nouns"
                negativeTerritory = 4
            else:
                if negativeTerritory > 0:
                    pos = nltk.pos_tag([word])
                    pos = pos[0][1]
                    if ((re.match('VB[G,P,D]*', pos) or re.match(('JJ|RB'), pos)) and modifier == "vrbAdj"):
                        if word not in stop: entry["words"][j] = "not_" + word
                    elif (re.match('NN.*', pos) and modifier == "nouns"):
                        if word not in stop: entry["words"][j] = "not_" + word
                    negativeTerritory -= 1
        mynegation.append({'words':entry["words"]})
        numProcessed += 1


    outJSON = open(outputFName, "w")
    json.dump(mynegation, outJSON)
    outJSON.close()

from nltk.stem import WordNetLemmatizer
def stemming(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    #porter = nltk.PorterStemmer()
    my_stemming=[]
    for entry in inJSON:
        dummy={}
        for j in range(len(entry["words"])):
            word = entry["words"][j]

            if re.match('not_.*', word):
                word = word[4:]
                entry["words"][j] = "not_" + WordNetLemmatizer().lemmatize(word)
            else:
                entry["words"][j] = WordNetLemmatizer().lemmatize(word)
        dummy['words']=entry['words']
        my_stemming.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(my_stemming, outJSON)
    outJSON.close()

def propNounConcat(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    my_propNounConcat=[]
    for entry in inJSON:
        dummy={}
        numWords = len(entry["words"])
        j = 0
        while j < numWords:
            word = entry["words"][j]
            if word[0].isupper() and j+1 < numWords:
                word2 = entry["words"][j+1]
                if word2[0].isupper() and j>0:
                    joinedWord = word + "_" + word2
                    entry["words"][j] = joinedWord
                    del entry["words"][j+1]
                    numWords -= 1
            j += 1
        dummy['words']=entry['words']
        my_propNounConcat.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(my_propNounConcat, outJSON)
    outJSON.close()




def removeStopWords(inputFName, outputFName):
    '''
    :param inputFName: name of JSON file with word splits
    :param outputFName: name of output JSON file
    :return: JSON file with stop words removed
    '''
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    stop = stopwords.words('english')
    stop.append('nt')
    my_removeStopWords=[]
    for entry in inJSON:
        dummy={}
        entry["words_nostopwords"] = []
        for word in entry["words"]:
            if word not in stop:
                entry["words_nostopwords"].append(word)
        dummy['words']=entry["words_nostopwords"]
        my_removeStopWords.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(my_removeStopWords, outJSON)
    outJSON.close()

############################################################################################################
########################################### Functions: Prepare Data Frames I ###############################
############################################################################################################

def get_ratings_words(inputFName1,inputFName2,myindex=None):
    
    
    inJSON1 = json.load(open(inputFName1, "r"))
    inJSON2 = json.load(open(inputFName2, "r"))
    
    word_category=pd.DataFrame(inJSON1)
    word_category.columns=['words']
    word_category['rating']=pd.DataFrame(inJSON2)['rating']
    if not myindex==None:
        word_category=word_category.ix[myindex,:]

    is_train=True
    ratings_words={}
    for i in range(0,5):
        ratings_words[i]=word_category[word_category['rating']==i][['words']]

    return (ratings_words)
def get_frequencies(ratings_words):
    
    ratings_freq={}
    for i in range(0,5):
        a=list(ratings_words[i].ix[:,0])
        aa=np.concatenate(a)
        ratings_freq[i]=nltk.FreqDist(aa)
        min_len=len(ratings_freq[0])
        if len(ratings_freq[i])<min_len:
            min_len=len(ratings_freq[i])

    return (ratings_freq)
def get_frequency_columns(ratings_freq,inputFName1,myindex=None):
    # cerates the data frame for the frequencies
    inJSON1 = json.load(open(inputFName1, "r"))
    word_category=pd.DataFrame(inJSON1)
    word_category.columns=['words']
    if not myindex==None:
        word_category=word_category.ix[myindex,:]
    def fregFunc(elt,k):
        return sum([ratings_freq[k][i] if  ratings_freq[k].has_key(i) else 0 for i in elt])   
    for i in range(0,5):
        dummy=word_category['words'].map(lambda x: fregFunc(x,i))
        word_category['freq'+str(i)]=dummy
        
    freq_cols=[elt for elt  in word_category.columns if not elt =='words']
    
    #df=word_category[freq_cols]
    #return(df.div(np.where(df.sum(axis=1)>0,df.sum(axis=1),1), axis=0))

    return (word_category[freq_cols])


def get_ratings_tfidf(ratings_words):

    ratings_text={}
    #create text files for each category
    for i in range(5):
        ratings_text[i]=list(ratings_words[i].ix[:,0])
        ratings_text[i]=np.concatenate(ratings_text[i])
        ratings_text[i]=' '.join(ratings_text[i])
    ratings_text=list(ratings_text.values())
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(ratings_text)
    ratings_tfidf={}
    for i in range(5):  
        response = tfidf.transform([ratings_text[i]])
        feature_names = tfidf.get_feature_names()
        ratings_tfidf[i]={}
        for col in response.nonzero()[1]:
            ratings_tfidf[i][feature_names[col]]=response[0, col]
    return(ratings_tfidf)

def get_TFIDF_columns(ratings_tfidf,inputFName1,myindex=None):
    # cerates the data frame for the TFIDF
    inJSON1 = json.load(open(inputFName1, "r"))
    word_category=pd.DataFrame(inJSON1)
    word_category.columns=['words']
    if not myindex==None:
        word_category=word_category.ix[myindex,:]

    def tfidfFunc(elt,k):
        return sum([ratings_tfidf[k][i] if  ratings_tfidf[k].has_key(i) else 0 for i in elt])

    for i in range(0,5):
        dummy=word_category['words'].map(lambda x: tfidfFunc(x,i))
        word_category['tfidf'+str(i)]=dummy
        
    tfidf_cols=[elt for elt in word_category.columns if not elt =='words']
    #df=word_category[tfidf_cols]
    #return(df.div(np.where(df.sum(axis=1)>0,df.sum(axis=1),1), axis=0))
    return(word_category[tfidf_cols])

def get_Features_columns(inputFiles,current_path,myindex):
    df=pd.DataFrame()
    for myfile in inputFiles:
        inputFName=current_path+myfile
        inJSON = json.load(open(inputFName, "r"))
        dummy=pd.DataFrame(inJSON)
        cols=[i for i in dummy.columns if not i=='words']
        if myindex==None:
            df[cols]=dummy
        else:
            df[cols]=dummy.ix[myindex,:]
    return(df)

        
        
def get_AllWords(ratings_tfidf,numWords):
    my_min_words=np.min([len(ratings_tfidf[i]) for i in range(5)])
    my_min_words=np.min([my_min_words,numWords])
    AllWords_dict={}
    for i in range(5):
        my_zip=zip(*list(ratings_tfidf[i].iteritems()))
        my_index=np.argsort(my_zip[1])[::-1]
        my_zipp=np.array(zip(*my_zip))
        # tuple of(word, tfidf), ordered with tfidf
        #i is the rating
        AllWords_dict[i]=my_zipp[my_index][:my_min_words]
        
    AllWords=[0]*5
    for i in range(5):
        AllWords[i]=set(zip(*AllWords_dict[i])[0])
        
    AllWords_dict=None
        
    my_union=set()
    my_intersection=set()
    for i in range(5):
        my_union=my_union.union(AllWords[i])
        my_intersection=my_intersection.intersection(AllWords[i])

    AllWords=list(my_union.difference(my_intersection))
        
    return(AllWords)


def get_Words_columns(AllWords,inputFName1,myindex=None):
    
    def is_words_in(my_word,elt):
        return np.where((my_word in elt),1,0)
    words_df=pd.DataFrame()

    inJSON1 = json.load(open(inputFName1, "r"))
    word_category=pd.DataFrame(inJSON1)
    word_category.columns=['words']
    if not myindex==None:
        word_category=word_category.ix[myindex,:]

    for my_word in AllWords:
        words_df[my_word]=word_category['words'].map(lambda elt: is_words_in(my_word,elt) )
    
    words_cols=[elt for elt in words_df.columns if not elt =='words']
    return(words_df[words_cols])
        
        

def get_ratings(inputFName2,myindex=None):
    
    inJSON2 = json.load(open(inputFName2, "r"))
    
    df_ratings=pd.DataFrame()
    df_ratings['rating']=pd.DataFrame(inJSON2)['rating']
    if not myindex==None:
        df_ratings=df_ratings.ix[myindex,:]

    return (df_ratings)

############################################################################################################
########################################### Functions: Prepare Data Frames II ###############################
############################################################################################################
def prepare_data(current_path,inputFilenames,outputFilenames,myFunctions):
    for i in range(len(myFunctions)):
        
        myFunction=myFunctions[i]      
        inputFName=current_path+inputFilenames[i]
        outputFName=current_path+outputFilenames[i]
        myFunction(inputFName, outputFName)


def create_Train_Test_files(my_paths,my_path_keys,fileName="kaggle.json"):


    inputFilenames=[fileName,"splitWords.json","splitWords.json","remove_punctuation.json",\
                    "remove_punctuation.json","remove_punctuation.json","remove_punctuation.json",\
                    "negation.json","propNounConcat.json","lower_words.json","removeStopWords.json"]
    outputFilenames=["splitWords.json","puncCount.json","remove_punctuation.json","avgWordLength.json",\
                     "isFirstPerson.json","posTag.json","negation.json","propNounConcat.json","lower_words.json",\
                     "removeStopWords.json","stemming.json"]
    myFunctions=[splitWords,puncCount,remove_punctuation,avgWordLength,isFirstPerson,posTag,negation,propNounConcat,\
                 lower_words,removeStopWords,stemming]

    

    for i,current_path_key in enumerate(my_path_keys):
        
        current_path=my_paths[i]
        prepare_data(current_path,inputFilenames,outputFilenames,myFunctions)
        

def create_Train_Test_data(my_paths,my_path_keys,myindex=(None,None),addWords_bool=True,test_rating=True):
    
    inputFiles=["puncCount.json","avgWordLength.json","isFirstPerson.json","posTag.json"]
    results={}
    for i,current_path_key in enumerate(my_path_keys):
        df_current=pd.DataFrame()
        current_path=my_paths[i]

        inputFName1=current_path+"stemming.json"
        inputFName2=current_path+"kaggle.json" 
        
        if (current_path_key=='train'):
            ratings_words=get_ratings_words(inputFName1,inputFName2,myindex=myindex[i])
            ratings_freq=get_frequencies(ratings_words)
            ratings_tfidf=get_ratings_tfidf(ratings_words)
            
            ratings_df_current=get_ratings(inputFName2,myindex=myindex[i])
            df_current[ratings_df_current.columns]=ratings_df_current
            
            if (addWords_bool):
                AllWords=get_AllWords(ratings_tfidf,250)

        
        freq_df_current=get_frequency_columns(ratings_freq,inputFName1,myindex=myindex[i])
        tfidf_df_current=get_TFIDF_columns(ratings_tfidf,inputFName1,myindex=myindex[i])
        features_df_current=get_Features_columns(inputFiles,current_path,myindex[i])

        
        df_current[tfidf_df_current.columns]=tfidf_df_current
        df_current[features_df_current.columns]=features_df_current
        df_current[freq_df_current.columns]=freq_df_current
        
        if ((current_path_key=='test') & test_rating):
            ratings_df_current=get_ratings(inputFName2,myindex=myindex[i])
            df_current[ratings_df_current.columns]=ratings_df_current
     
        if (addWords_bool):
            AllWords_df_current=get_Words_columns(AllWords,inputFName1,myindex=myindex[i])
            df_current[AllWords_df_current.columns]=AllWords_df_current
        
        results[current_path_key]=df_current
        
    if (addWords_bool):       
        return(results,ratings_words,ratings_tfidf,ratings_freq,AllWords)
    return(results,ratings_words,ratings_freq,ratings_tfidf)





def create_Test_data(test_path,all_results,fileName,test_rating=False,addWords_bool=True):
    
    ratings_tfidf=all_results["ratings_tfidf"]
    ratings_words=all_results["ratings_words"]
    ratings_freq=all_results["ratings_freq"]
    if addWords_bool:
        AllWords=all_results['AllWords']

    inputFiles=["puncCount.json","avgWordLength.json","isFirstPerson.json","posTag.json"]
    results={}

    df_current=pd.DataFrame()
    current_path=test_path

    inputFName2=fileName
    inputFName1=current_path+"stemming.json"
    inputFName2=current_path+"kaggle.json" 


    
    freq_df_current=get_frequency_columns(ratings_freq,inputFName1,myindex=None)
    tfidf_df_current=get_TFIDF_columns(ratings_tfidf,inputFName1,myindex=None)
    features_df_current=get_Features_columns(inputFiles,current_path,myindex=None)


    df_current[tfidf_df_current.columns]=tfidf_df_current
    df_current[features_df_current.columns]=features_df_current
    df_current[freq_df_current.columns]=freq_df_current

    if (test_rating):
        ratings_df_current=get_ratings(inputFName2,myindex=None)
        df_current[ratings_df_current.columns]=ratings_df_current

    if (addWords_bool):
        AllWords_df_current=get_Words_columns(AllWords,inputFName1,myindex=None)
        df_current[AllWords_df_current.columns]=AllWords_df_current

    return(df_current)

    

    


