import json
import urllib
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.io.json import json_normalize
import csv
import pymongo
from pymongo import MongoClient

# connect to mongo
connection = pymongo.MongoClient("mongodb://localhost")

# Connect to twitter database
db = connection.sugarkube
twit = db.twitter

#Initiate table
mongo = []

# Put the tweets into our blank list
for tw in twit.find({}, {'_id':0, 'text':1, 'created_at':1}):
    mongo.append(tw)

# From json format to DataFrame
tweets = json_normalize(mongo)

# Get rid of non unicode character
tx = tweets['text'].str.encode('utf-8')
tweets['text'] = tx

tweetsfinal = tweets[['text']]
tweetsfinal['Date'] = (pd.to_datetime(tweets['created_at'])).dt.date

# save cleaned tweets into csv
# /!\ the first time you run it, change path, header = True, mode = w
# Then the second time, header = False, mode = a (to append the existing file)

tweetsfinal.to_csv(path_or_buf="C:\\Data\\Twitter\\Tweets\\Clean_Tweets.csv", sep=',', na_rep='',
               float_format=None, columns=['text', 'Date'], header=False, index=False, index_label=None,
                mode='a', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n',
                chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')