import json
import urllib
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.io.json import json_normalize
import datetime as dt
import pymongo
from pymongo import MongoClient

# Connect to Mongo
connection = pymongo.MongoClient("mongodb://localhost")

# Create twitter database
db = connection.sugarkube
twittercol = db.twitter

# Drop existing collection - wipe and load method
#twittercol.drop()

from twython import TwythonStreamer

tweets = []

CONSUMER_KEY = 'iIGro5OoOYFmmJBbweRApLYEL'
CONSUMER_SECRET = 'tIEz0ak8XJmgprbWiR9TEPGT687zvcRS0jOsIVvsNJqFXZXjEe'
ACCESS_TOKEN = '2296555754-6BWSExNbF8QfcC2BuDzA8ZkPiKRuenJUzUzhTjq'
ACCESS_TOKEN_SECRET = 'eTkSwRw0pFKtSTHaAnlziiPa2UxQST3d7sHq7xdvjJwyw'


class MyStreamer(TwythonStreamer):
    def on_success(self, data):
        if data['lang'] == 'en':
            tweets.append(data)
            print 'received tweet no ', len(tweets)

        if len(tweets) >= 10:
            self.disconnect()

    def on_error(self, status_code, data):
        self.disconnect()


stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
stream.statuses.filter(track= ['breez london', 'cloudy london','cold london', 'ice london','icy london',
                      'icey london', 'drizzle london', 'frost london','wind london', 'mild london',
                      'dew london', 'freez london', 'downpour london', 'shower london', 'rain london',
                      'frost london', 'nippy london', 'hail london', 'temp london', 'gail london',
                      'gust london', 'sleet london', 'heat london', 'storm london', 'slush london',                               'fog london', 'flood london', 'weather london', \
                       'visibility london', 'warm london', 'mist london','chill london',
                       'thunder london', 'lightning london', 'snow london', 'hot london',
                       'sun london', 'boiling london', 'baltic london', 'burn london'])

#  Put the data in a json file then import them into sugarkube.twitter

with open("C:\\Data\\Twitter\\TwitterDaily_Pull.json", "w") as tw:
     json.dump(tweets, tw )

with open("C:\\Data\\Twitter\\TwitterDaily_Pull.json", "r") as j:
    parsed = json.load(j)

# Iterate through every news item on the page
for item in parsed:
    twittercol.insert_one(item)

