import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from numba import jit
from tqdm import tqdm
import time

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)

#@jit(nopython=True)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

dfLeft = pd.read_csv('C:/Users/rayri/Desktop/Georgia Tech/Spring 2021/CS 4400/Final/data/data/ltable.csv')
dfRight = pd.read_csv('C:/Users/rayri/Desktop/Georgia Tech/Spring 2021/CS 4400/Final/data/data/rtable.csv')
#print(df.to_string())
sentencesLeft = list(dfLeft['title'])
sentencesRight = list(dfRight['title'])

#print(sentencesLeft)
#print(sentencesRight)

listFin = []
for left in tqdm(sentencesLeft):
    for right in sentencesRight:
        if dfLeft.loc[dfLeft["title"] == left, "brand"].tolist() == dfRight.loc[dfRight["title"] == right, "brand"].tolist():
            simil = cosine(model([left])[0], model([right])[0])
            lmodel = dfLeft.loc[dfLeft["title"] == left, "modelno"].tolist()
            rmodel = dfRight.loc[dfRight["title"] == right, "modelno"].tolist()
            if simil >= 0.75 or (lmodel == rmodel and pd.notnull(lmodel) and pd.notnull(rmodel)):
                tempList = []
                tempList.append(dfLeft.loc[dfLeft["title"] == left, "id"].item())
                tempList.append(dfRight.loc[dfRight["title"] == right, "id"].item())
                tempList.append(simil)
                listFin.append(tempList)
                #print(tempList)
                #simil = 1
                #print("Sentence = ", left, "AND ", right, "; similarity = ", simil)
                #print(dfRight.loc[dfRight["title"] == right, "modelno"].tolist())
            #else:
                #simil = 0
        
#print(listFin)
dfFin = pd.DataFrame.from_records(listFin, columns = ['ltable_id', 'rtable_id', 'simil'], index = None)

dfFin.to_csv(r'C:/Users/rayri/Desktop/Georgia Tech/Spring 2021/CS 4400/Final/data/data/Final.csv', index = False)

#Deleting training data
df = pd.read_csv('C:/Users/rayri/Desktop/Georgia Tech/Spring 2021/CS 4400/Final/data/data/Final.csv')
dfTrain = pd.read_csv('C:/Users/rayri/Desktop/Georgia Tech/Spring 2021/CS 4400/Final/data/data/train.csv')
del dfTrain['label']
del df['simil']
dfSubmit = pd.merge(df,dfTrain, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
dfSubmit.to_csv(r'C:/Users/rayri/Desktop/Georgia Tech/Spring 2021/CS 4400/Final/data/data/dfFinal.csv', index = False)