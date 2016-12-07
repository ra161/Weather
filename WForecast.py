import json
import urllib
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.io.json import json_normalize
import csv

# URL to extract the weather information for London

url = 'http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/json/352409?res=daily&key=cf64269e-4a94-489f-bad3-7dbffe86e969'

response = urllib.urlopen(url)

data = json.loads(response.read())

# Clean raw data so we keep only the weather parameters

data = json_normalize(data['SiteRep']['DV']['Location'])
data = DataFrame(data['Period'][0])
data = data[['Rep', 'value']]

# Create the dataframe with the forecast, with temperature, data, days from pull and weather

forecast = pd.DataFrame()
forecast ['Date'] = data['value'].map(lambda x: str(x)[:-1])
forecast['Weather'] = None
forecast['Temperature'] = None
forecast['Days From Pull'] = None

for i in range(5):
    forecast['Weather'][i] = data['Rep'][i][0]['W']
    forecast['Temperature'][i] = data['Rep'][i][0]['FDm']
    forecast['Days From Pull'][i] = i

# Write csv file with the weather results. Note the 'a' in mode used to append the existing table

forecast.to_csv(path_or_buf='C:\\Data\\Twitter\\Weather-forecast.csv', sep=',', na_rep='NA',
                columns=['Date', 'Weather', 'Temperature','Days From Pull'], header=False, index=False,
                 index_label=None, mode='a', encoding=None, compression=None, quoting=None, quotechar='"',
                 line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True,
                 escapechar=None, decimal='.')