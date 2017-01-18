
# coding: utf-8

# In[1]:

import scipy
import numpy as np
import pandas as pd
import plotly.plotly as py
# import visplots
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats.distributions import randint
import csv
import numpy as np
from pandas import Series, DataFrame
import plotly.graph_objs as go
from plotly import tools

from nltk import word_tokenize, wordpunct_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs
from nltk.classify.api import ClassifierI

# getting around the ascii characters
from django.utils.encoding import smart_str, smart_unicode


# In[2]:

# Import the clean tweets (text and date)
twitter_raw = pd.read_csv("C:\\Data\\Twitter\\Tweets\\Clean_Tweets1.csv", sep=',', delimiter=None)
twitter_raw['text'].count()


# # Delete tweets containing key words

# In[3]:

twitter_cleaned=twitter_raw[twitter_raw['text'].str.lower().str.contains                    ("great smog of london|spanish|sex|porn|anal|pov|bbw|milf|sexy|shemale|sexyfishrestaurant|                   nude|sluts|super hot blonde|adult video|erotic|18+|dirty fun|killer fog|rihanna")==False]

twitter_cleaned = twitter_cleaned.reset_index(drop=True)

twitter_cleaned['text'].count()


# #  Machine Learning to clean tweets

# # 1. Create the Maching learning

# In[4]:

twlist = []

with open(r"C:\\Git\\Weather\\trained_weather_NoRT.csv", "r") as t:
    tweets_raw = pd.read_csv(t)

tweets = tweets_raw[['text', 'weather']].values.tolist()

twlist = [tuple(l) for l in tweets] # turn nested list of lists into list of tuples
twtokens = []


# In[5]:

for (words, weather) in twlist:
#    words_filtered = [e.lower().decode('utf8') for e in words.split() if len(e) >= 3 and len(e) <= 10] # and <= 10
    words_filtered = [unicode(e.lower(), errors = 'replace') for e in words.split() if len(e) >= 3 and len(e) <= 10] # and <= 10
    twtokens.append((words_filtered, weather))


# In[6]:

import nltk
def get_words_in_tweets(tweets):
    all_words = []
    for (words, weather) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


# In[7]:

word_features = get_word_features(get_words_in_tweets(twtokens))


# In[8]:

# The Classifier
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[10]:

training_set = nltk.classify.apply_features(extract_features, tweets)

# train the classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(32)


# # 2. Classify London Tweets

# In[13]:

df = pd.DataFrame(twitter_cleaned)

tx = df['text']
df['text'] = tx


# In[14]:

dffinal = df[['text']]
dffinal['date'] = df['Date']


# In[15]:

for i in range(len(df.index)):
    if classifier.classify(extract_features((smart_str(df['text'][i])).split())) == 'yes':
        dffinal['text'][i] = smart_str(df['text'][i])
    else:
        dffinal['text'][i] = None

count = 0
for i in range(len(df.index)):
    if dffinal['text'][i] != None:
        count += 1

count2 = 0
dffinaltrained = pd.DataFrame({'date' : pd.Series(range(count), index=range(count)), 'text' : pd.Series(range(count), index=range(count))})
for i in range(len(df.index)):
    if dffinal['text'][i] != None:
        dffinaltrained['text'][count2] = dffinal['text'][i]
        dffinaltrained['date'][count2] = dffinal['date'][i]
        count2 += 1


# In[16]:

dffinaltrained['text'].count()


# # Define function which will count the weather words

# In[17]:

def wordCount(tweet):
    # List of words we are looking at
    weather_words = ['breeze', 'breezy', 'cloudy', 'cold', 'ice', 'icy', 'icey', 'drizzle', 'frost', 'wind', 'mild', 'dew', 
                     'freezing', 'downpour', 'shower', 'rain', 'frost', 'nippy', 'hail', 'temperature', 'gail', 'gust',
                     'sleet', 'heat', 'storm', 'slush', 'fog', 'foggy', 'flood', 'visibility', 'warm', 'warmer',
                     'mist', 'frosty', 'misty', 'chilly', 'thunder', 'lightning', 'snow', 'snowing', 'hot', 'sun', 'sunny',
                     'boiling', 'baltic', 'burn']
    # Create a new dictionnary
    counts = dict()
    for t in tweet:
        if t in weather_words:
            counts[t] = counts.get(t,0) + 1

    return counts


# # Get the number of final clean tweets per day

# In[18]:

# Check the days and number of tweets per day and create dataframe for nb of tweets per day
df_count = dffinaltrained.groupby(['date']).count()

# Resets index and rename column "text" as total_count
df_count.reset_index(level = 0, inplace = True)
df_count.rename(columns = {'text':'total_count'}, inplace = True)

df_count


# In[19]:

# Create a new dataframe with the date
tw_date = dffinaltrained.groupby(['date']).count()
tw_date.index


# # Going through all tweets then count words and put them in final DataFrame (per day)

# In[20]:

# Initialise the final dataframe
tw_final = DataFrame()
tw_final['date'] = tw_date.index
tw_final['Keywords'] = None

# list of tuples (keyword, count) - its length represents the number of keywords encountered
keyw_tup=[]

# list with the number of keywords per day
keyw_count_day=[]


# In[21]:

for i in range(len(tw_date.index)):
    df = dffinaltrained[(dffinaltrained['date'] == tw_final['date'][i])]
    twitterlist = []
    for t in range(len(df.index)):
        twitterlist.append(dffinaltrained['text'][t].lower().split())

    # Flatten the list so all values in the same list
    tweets_flatten = [j for sublist in twitterlist for j in sublist]

    # Counting the number of key words in the tweets
    d = wordCount(tweets_flatten)
    
    # give a list of the sorted tweets, with most popular coming first
    d2 = sorted(d.items(), key = lambda x: x[1], reverse = True)
    keyw_tup.append(d2)
    tw_final['Keywords'][i] = keyw_tup[i]
    keyw_count_day.append(len(d2)) 


# In[22]:

#if no count keyword for that day, inserts (0,0)
for i in range(len(tw_date.index)):
    if len(tw_final['Keywords'][i])<max(keyw_count_day):
        for j in range (max(keyw_count_day)-len(tw_final['Keywords'][i])):
            tw_final['Keywords'][i].append((0,0))


# In[23]:

#Spilits colom 'Keywords' into many coloums
for z in range(max(keyw_count_day)):
    b=[]
    b.extend(a[z] for a in tw_final['Keywords'])
    o=z+1
    keyword=[]
    keyword.append('Keyword_'+str(o))
    keyword_str = ''.join(keyword)
    tw_final[keyword_str] = b

# Drop column with list of tuples
tw_final = tw_final.drop(['Keywords'], axis=1)
tw_final.head()


# # Reformat Final Dataframe : unpivot and split by keyword and count

# In[24]:

# Unpivots tw_final_all and rename column 'variable' for 'rank'
tw_final_upv = pd.melt(tw_final, id_vars = ['date'])
tw_final_upv.rename(columns = {'variable':'rank'}, inplace = True)

# Split 'value' in two columns: keyword and count
b = []
c = []

b.extend(a[1] for a in tw_final_upv['value'])
tw_final_upv['count'] = b

c.extend(a[0] for a in tw_final_upv['value'])
tw_final_upv['keyword'] = c

# Delete useless column
tw_final_upv = tw_final_upv.drop(['value'], axis=1)


# In[25]:

# Twist to get all keywords per day (ordered)
tw_final_upv = pd.merge(tw_final_upv, df_count, how='inner', on='date')
tw_final_upv = tw_final_upv.drop(['total_count'], axis=1)

tw_final_upv.head()


# In[26]:

# Replace 0 by NaN in count and keyword
tw_final_upv = tw_final_upv.replace(0, np.nan)


# In[27]:

# Create CSV with the results
tw_final_upv.to_csv(path_or_buf="C:\\Data\\Twitter\\Tweets\\Daily_Keyword_Occurrence.csv",                     sep=',', na_rep='', float_format=None,                     columns=['date', 'rank', 'keyword', 'count'],                     header=True, index=False, index_label=None, mode='w',quotechar='"', line_terminator='\n', decimal='.')


# # Initialisation Plots

# In[28]:

# Plot options
# init_notebook: On jupyter
# plotly : on plotly

#init_notebook_mode()
import plotly
plotly.tools.set_credentials_file(username='Laura_Foulquier', api_key='zehu3fat3Mfs6v3pTNMY')


# # Building Cold - Hot Plots

# In[29]:

# Define list of keywords which describe if the weather is cold or hot 

weather_words_hot=['heat', 'warm', 'warmer','hot', 'sun', 'sunny','boiling',  'burn']

weather_words_cold=['breeze', 'breezy', 'cloudy', 'cold', 'ice', 'icy', 'icey', 'drizzle', 'frost', 'wind', 'mild', 'dew', 
                     'freezing', 'downpour', 'shower', 'rain', 'frost', 'nippy', 'hail', 'temperature', 'gail', 'gust',
                     'sleet',  'storm', 'slush', 'fog', 'foggy', 'flood', 'visibility', 
                     'mist', 'frosty', 'misty', 'chilly', 'thunder', 'lightning', 'snow', 'snowing', 'baltic']


# In[30]:

# Count the total of keywords for hot or cold
count_cold=0
count_hot=0

for i in range (len(tw_final_upv)):
    if tw_final_upv['keyword'][i] in weather_words_cold:
        count_cold = count_cold + tw_final_upv['count'][i]
    elif tw_final_upv['keyword'][i] in weather_words_hot:
        count_hot = count_hot + tw_final_upv['count'][i]
count_cold, count_hot


# In[31]:

# Plot the number of cold and hot keywords over the entire project time (sum)
trace = Bar(
    x=['Cold','Hot'],
    y=[count_cold,count_hot] ,
    marker = dict(
        color=['blue','red'])
    )

data = Data([trace])

layout = go.Layout(
    title='Total count of hot keywords vs cold keywords',
    yaxis = dict(title = 'Keywords Count')
) 

fig = go.Figure(data=data, layout=layout)
#py.plot(fig, filename = 'Hot_vs_cold_Keywords_Totals') 
#iplot(fig)
#py.plot(data, filename = 'Summer_vs_winter_Totals')


# # PLOTTING WITH METOFFICE DATA

# # 1. Import MetOffice Data and Join final tables

# In[32]:

# Import the MetOffice Data
final_weather = pd.read_csv("C:\\Data\\Twitter\\Final_weather_for_plotting.csv", sep=',', delimiter=None)

#Rename Date column for futur join
final_weather.rename(columns = {'Date':'date'}, inplace = True)


# In[33]:

# Join on dates with tweeter results
final_results = pd.merge(tw_final_upv, final_weather, how='inner', on='date')

# Insert column for Weather depending on Bucket_Weather  number
final_results['Weather'] = None

for i in range(len(final_results.index)):
    if (final_results['Bucket_Weather'][i] == 1) == True:
        final_results['Weather'][i] = 'Sunny'

    elif (final_results['Bucket_Weather'][i] == 2) == True:
        final_results['Weather'][i] = 'Cloudy'
    
    elif (final_results['Bucket_Weather'][i] == 3) == True:
        final_results['Weather'][i] = 'Light Rain'
        
    elif (final_results['Bucket_Weather'][i] == 4) == True:
        final_results['Weather'][i] = 'Heavy Rain'

    elif (final_results['Bucket_Weather'][i] == 5) == True:
        final_results['Weather'][i] = 'Sleet / Hail'
        
    elif (final_results['Bucket_Weather'][i] == 6) == True:
        final_results['Weather'][i] = 'Snow'

    elif (final_results['Bucket_Weather'][i] == 7) == True:
        final_results['Weather'][i] = 'Thunder'

    elif (final_results['Bucket_Weather'][i] == 8) == True:
        final_results['Weather'][i] = 'Fog / Mist'

# Column with constant for plotting
final_results['plot'] = 16


# In[34]:

final_results.head()


# # 2. Create Array for final results tables

# In[35]:

npfinal = np.array(final_results)

# Put the date in X, the rest of the parameter in Y
X = npfinal[:,0]
Y = npfinal[:,1:]


# # 3. Build DF for Occurence of Cold and Hot keywords per day

# In[36]:

#To get a count of cold and hot keywords for each day

count_cold_2_list=[]
count_hot_2_list=[]
i_list=[]
dump_list=[]

for j in range(len(df_count)):
    count_cold_2=0
    count_hot_2=0
    dump=0
    for i in range(max(keyw_count_day)):     # takes the number of keywords per day
        i_list.append(i)
        x=len(i_list)-1                      # number of how many for loops have run
        if tw_final_upv['keyword'][x] in weather_words_cold:
                count_cold_2 = count_cold_2 + tw_final_upv['count'][x]
        elif tw_final_upv['keyword'][x] in weather_words_hot:
                count_hot_2 = count_hot_2 + tw_final_upv['count'][x]
        else:
            dump=dump + tw_final_upv['count'][x]
    
    count_cold_2_list.append(count_cold_2)
    count_hot_2_list.append(count_hot_2)
    dump_list.append(dump)


# In[37]:

# Create new dataframe with new information

tw_hot_cold = DataFrame()
tw_hot_cold['date'] = df_count['date']
tw_hot_cold['cold'] = count_cold_2_list
tw_hot_cold['hot'] = count_hot_2_list
tw_hot_cold['total count'] = tw_hot_cold['cold'] + tw_hot_cold['hot']
tw_hot_cold['% hot'] =(tw_hot_cold['hot'] / tw_hot_cold['total count'])*100
tw_hot_cold['% cold'] = 100 - tw_hot_cold['% hot']

tw_hot_cold = tw_hot_cold.round(2)

tw_hot_cold.head()


# In[38]:

# Convert DF to array for plotting purposes
npArray_hot_cold = np.array(tw_hot_cold)

# Put the date in X, the rest of the parameter in Y
X1 = npArray_hot_cold[:,0]
Y1 = npArray_hot_cold[:,1:]


# # 4. Plotting temperature and occurence (count) of cold and hot keywords per day

# In[39]:

# Plotting the temperature
trace0 = go.Scatter(
    x = X,
    y = Y[:,3],
    name = 'Temperature at noon',
    yaxis='y2',
    line = dict(
        color = 'orange',
        width = 3
    )
)

# Plot count of cold keywords
trace1 = go.Bar(
    x = X1,
    y = Y1[:,0],
    name = 'Cold',
    marker = dict(
        color = 'rgb(100,250,250)',
        line = dict(
            color = 'rgb(100,200,250)',
            width = 1.5)
    )
)


# Plotting count of hot keywords
trace2 = go.Bar(
    x = X1,
    y = Y1[:,1],
    name = 'Hot',
    marker = dict(
        color = 'rgb(255,100,100)',
        line = dict(
            color = 'rgb(255,0,0)',
            width = 1.5)
    )
)



layout = Layout(
    title = 'Keywords (Hot or Cold) Occurence from tweets and temperature',
    xaxis = dict(
        title = 'Date',
        tickangle=320,
        autotick = 'False',
        ticks = 'outside'
    ),
    yaxis =dict(
        title='Count of word occurence',
        autotick = 'False',
        ticks = 'outside',
        tickfont=dict(
            color='rgb(148, 103, 189)',
        ) 
    ),
    yaxis2 = dict(
        title='Temperature (degrees C)',
        range = [0,20],
        autotick = 'False',
        ticks = 'outside',
        overlaying='y',
        side='right'
    ),
    showlegend = True,
    legend = dict(
         x = 0,
         y = -0.75
    ),
    font=dict(family='Old Standard TT, serif', size=14, color='purple')
)


data = [trace0, trace1, trace2]

fig = dict(data=data, layout = layout)
#iplot(fig)

# Put the plot in the plotly account
py.plot(fig, filename = 'Keywords Occurence (Hot or cold) from tweets')


# # 4.2 Plotting temperature and occurence (%) of cold and hot keywords per day

# In[40]:

# Plotting the temperature
trace0 = go.Scatter(
    x = X,
    y = Y[:,3],
    name = 'Temperature at noon',
    yaxis='y2',
    line = dict(
        color = 'orange',
        width = 3
    )
)

# Plot count of cold keywords
trace1 = go.Scatter(
    x = X1,
    y = Y1[:,4],
    name = 'Cold',
    marker = dict(
        color = 'rgb(100,250,250)',
        line = dict(
            color = 'rgb(100,200,250)',
            width = 1.5)
    )
)


# Plotting count of hot keywords
trace2 = go.Scatter(
    x = X1,
    y = Y1[:,3],
    name = 'Hot',
    marker = dict(
        color = 'rgb(255,100,100)',
        line = dict(
            color = 'rgb(255,0,0)',
            width = 1.5)
    )
)



layout = Layout(
    title = 'Keywords (Hot or Cold) % Occurence from tweets and temperature',
    xaxis = dict(
        title = 'Date',
        tickangle=320,
        autotick = 'False',
        ticks = 'outside'
    ),
    yaxis =dict(
        title='% of word occurence',
        range = [20,70],
        autotick = 'False',
        ticks = 'outside',
        tickfont=dict(
            color='rgb(148, 103, 189)',
        ) 
    ),
    yaxis2 = dict(
        title='Temperature (degrees C)',
        range = [0,20],
        autotick = 'False',
        ticks = 'outside',
        overlaying='y',
        side='right'
    ),
    showlegend = True,
    legend = dict(
         x = 0,
         y = -0.75
    ),
    font=dict(family='Old Standard TT, serif', size=14, color='purple')
)


data = [trace0, trace1, trace2]

fig = dict(data=data, layout = layout)
#iplot(fig)

# Put the plot in the plotly account
py.plot(fig, filename = 'Keywords Occurence (Hot or cold) % from tweets')


# # 5. Temperature vs top 3 keywords per day

# In[41]:

# join tables to get total count

final_results_plot = pd.merge(final_results, tw_hot_cold, how='inner', on='date')
final_results_plot = final_results_plot.drop(['Bucket_Weather', 'cold', 'hot', 'plot', 'Weather', 'Temperature',                                               '% hot', '% cold'], axis=1)

#Limited to 2 decimals for plotting
final_results_plot['%daily occurrence'] = ((final_results_plot['count'] / final_results_plot['total count'])*100).round(2)


# In[42]:

final_results_plot.head()


# In[43]:

# Build new dataframes for plotting purposes

# DataFrame for rank = Keyword_1
rank1 = (final_results_plot[(final_results_plot['rank'] == 'Keyword_1')])
rank1 = rank1.reset_index(drop=True)

npArray = np.array(rank1)
X1 = npArray[:,0]
Y1 = npArray[:,1:]

# DataFrame for rank = Keyword_2
rank2 = (final_results_plot[(final_results_plot['rank'] == 'Keyword_2')])
rank2 = rank2.reset_index(drop=True)

npArray = np.array(rank2)
X2 = npArray[:,0]
Y2 = npArray[:,1:]

# DataFrame for rank = Keyword_3
rank3 = (final_results_plot[(final_results_plot['rank'] == 'Keyword_3')])
rank3 = rank3.reset_index(drop=True)

npArray = np.array(rank3)
X3 = npArray[:,0]
Y3 = npArray[:,1:]


# In[44]:

rank3.head()


# In[45]:

# Plotting the temperature
trace0 = go.Scatter(
    x = X,
    y = Y[:,3],
    name = 'Temperature at noon',
    yaxis='y2',
    line = dict(
        color = 'orange',
        width = 3
    )
)

# Plotting KeyWord1 occurence and name
trace1 = go.Bar(
    x = X1,
    y = Y1[:,4],
    text = Y1[:, 2],
    name = 'Occurence Keyword_1',
    marker = dict(
        color = 'rgb(255,100,100)',
        line = dict(
            color = 'rgb(255,0,0)',
            width = 1.5)
    )
)

# Plotting KeyWord2 occurence and name
trace2 = go.Bar(
    x = X2,
    y = Y2[:,4],
    text = Y2[:, 2],
    name = 'Occurence Keyword_2',
    marker = dict(
        color = 'rgb(158,202,225)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5)
    )
)

# Plotting KeyWord3 occurence and name
trace3 = go.Bar(
    x = X3,
    y = Y3[:,4],
    text = Y3[:, 2],
    name = 'Occurence Keyword_3',
    marker = dict(
        color = 'rgb(100,250,250)',
        line = dict(
            color = 'rgb(100,200,250)',
            width = 1.5)
    )
)

layout = Layout(
    title = 'Keywords Occurence from tweets and temperature',
    xaxis = dict(
        title = 'Date',
        tickangle=320,
        autotick = 'False',
        ticks = 'outside'
    ),
    yaxis =dict(
        title='% of word occurence',
       # range = [4,16],
        autotick = 'False',
        ticks = 'outside',
        tickfont=dict(
            color='rgb(148, 103, 189)',
        ) 
    ),
    yaxis2 = dict(
        title='Temperature (degrees C)',
        range = [0,20],
        autotick = 'False',
        ticks = 'outside',
        overlaying='y',
        side='right'
    ),
    showlegend = True,
    legend = dict(
         x = 0,
         y = -0.75
    ),
    font=dict(family='Old Standard TT, serif', size=14, color='purple')
)

data = [trace0, trace1, trace2, trace3]

fig = dict(data=data, layout = layout)
#iplot(fig)

# Put the plot in the plotly account
py.plot(fig, filename = 'Keywords Occurence from tweets and temperature from MetOffice')


# # 6. Plot Actual Weather vs KeyWords

# In[46]:

# Display the actual weather at noon from MetOffice
trace0 = go.Scatter(
    x = X,
    y = Y[:, -1],
    text = Y[:, -2],
    name = 'MetOffice Weather',
    mode = 'markers',
    marker = dict(
        size = 1,
        color = 'blue',
        opacity = 0.8
    )
)

# Plotting KeyWord1 occurence and name
trace1 = go.Bar(
    x = X1,
    y = Y1[:,4],
    text = Y1[:, 2],
    name = 'Occurence Keyword_1',
    marker = dict(
        color = 'rgb(255,100,100)',
        line = dict(
            color = 'rgb(255,0,0)',
            width = 1.5)
    )
)

# Plotting KeyWord2 occurence and name
trace2 = go.Bar(
    x = X2,
    y = Y2[:,4],
    text = Y2[:, 2],
    name = 'Occurence Keyword_2',
    marker = dict(
        color = 'rgb(158,202,225)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5)
    )
)

# Plotting KeyWord3 occurence and name
trace3 = go.Bar(
    x = X3,
    y = Y3[:,4],
    text = Y3[:, 2],
    name = 'Occurence Keyword_3',
    marker = dict(
        color = 'rgb(100,250,250)',
        line = dict(
            color = 'rgb(100,200,250)',
            width = 0.5)
    )
)

layout = Layout(
    title = 'Keywords Occurence from tweets and actual weather',
    xaxis = dict(
        title = 'Date',
        tickangle=320,
        autotick = 'False',
        ticks = 'outside'
    ),
    yaxis =dict(
        title='% of word occurence',
        range = [0, 18],
        autotick = 'False',
        ticks = 'outside',
        tickfont=dict(
            color='rgb(148, 103, 189)',
        ) 
    ),
    showlegend = True,
    legend = dict(
         x = 0,
         y = -0.75
    ),
    font=dict(family='Old Standard TT, serif', size=14, color='purple')
)

data = [trace0, trace1, trace2, trace3]

fig = dict(data=data, layout = layout)
#iplot(fig)

# Put the plot in the plotly account
py.plot(fig, filename = 'Keywords Occurence from tweets and actual weather from MetOffice')


# # 7 . Plotting all other keywords but cold vs temperature

# In[47]:

Word_count = tw_final_upv.merge(tw_hot_cold)
Word_count = Word_count.drop(['cold', 'hot', '% hot', '% cold'], axis =1)
Word_count['% daily occurence'] = ((Word_count['count'] / Word_count['total count'] )*100).round(2)

Word_count.head()


# In[48]:

# Plot 1- % cold keyword against temperature at noon

trace0 = Scatter(
    x = Word_count[(Word_count['keyword'] == 'cold')]['date'],
    y = (100 - Word_count[(Word_count['keyword'] == 'cold')]['% daily occurence']), # % Not talking about cold
    name = 'Twitter not "cold" % Occurrence',
    line = dict(
        color = 'navy',
        width = 3
    ),
    yaxis='y2'
)

trace1 = go.Scatter(
    x = X,
    y = Y[:,3],
    name = 'Met Office Temperature at Noon',
    line = dict(
        color = 'orange',
        width = 2,
    )
)

data = [trace0, trace1]
layout = go.Layout(
    title='% of keywords not being cold and the actual temperature',
    yaxis=dict(
        title='Temperature (degrees C)',
        range = [0,20]
    ),
    yaxis2=dict(
        title='% Different from Cold',
        overlaying='y',
        side='right',
        range = [80, 90]
    ),
    legend = dict(
        x = 0,
        y = -0.75
    ),
    font=dict(family='Old Standard TT, serif', size=14, color='purple')
)
fig = go.Figure(data=data, layout=layout)
#iplot(fig)
plot_url = py.plot(fig, filename = 'Twitter %cold and Met Office')


# In[ ]:



