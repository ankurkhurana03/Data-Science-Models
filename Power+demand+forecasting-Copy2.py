
# coding: utf-8

# In[45]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[46]:

data=pd.read_csv('interval_lp_sample_3.txt')


# In[47]:

data.replace('\\N',np.nan,inplace=True)


# In[48]:

data['timestamp']=pd.to_datetime(data['timestamp'])


# In[49]:

data.index=data['timestamp']
del[data.index.name]


# In[50]:

t=data[data.index.minute!=30].ix[:,~data.columns.isin(['meter_id','timestamp'])].values[0].tolist()
del[t[-1]]
t=[np.nan]+t

#data[data.index.minute!=30].shift(1,axis=1)
data[data.index.minute!=30].ix[:,~data.columns.isin(['meter_id','timestamp'])]=np.array(t)


# In[51]:

data[data.index.minute!=30].ix[:,~data.columns.isin(['meter_id','timestamp'])]


# In[52]:

date_range_orig=pd.date_range(data['timestamp'].values[0], periods=(pd.Timestamp(data['timestamp'].values[-1])-pd.Timestamp(data['timestamp'].values[0])).components.days, freq='D')


# In[53]:

date_range_orig


# In[54]:

data = data.reindex(date_range_orig, fill_value=np.nan)

data.index=data.timestamp
del(data.index.name)
# In[55]:

data.drop(['meter_id','timestamp'],inplace=True,axis=1)


# In[56]:

plt.figure(figsize=(20,10))
plt.plot(data.interpolate(method='time')['C002'],color='y')
plt.plot(data['C002'],color='k')
plt.show()


# In[57]:

data.interpolate(method='time',inplace=True)


# In[58]:

dindex = pd.date_range('2015-03-08 12:30:00', periods=1, freq='15min')
for index, row in data.iterrows():
    dindex=dindex.append(pd.date_range(index, periods=96, freq='15min'))
dindex=dindex.delete(0)


# In[59]:

data1=data.stack(dropna=False)


# In[60]:

data1=data1.astype(np.float)


# In[61]:

data1=data1.reset_index().reset_index(drop=True)


# In[62]:

data1.drop(['level_0','level_1'],inplace = True, axis=1)


# In[63]:

data1.index=dindex


# In[64]:

print(dindex[-1])
import datetime


# In[70]:

split_number=15860
df_train, df_test = data1.ix[:split_number, :].copy(), data1.ix[split_number:, :].copy()
def split_data(split_number):
    df_train, df_test = data1.ix[:split_number, :].copy(), data1.ix[split_number:, :].copy()
split_data(split_number)


# In[71]:

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def feature_generation(data):
    data2=data.copy()
    data2.loc[data2.index[-1]+datetime.timedelta(minutes=15)]=0#np.nan
    df_train.loc[data2.index[-1]+datetime.timedelta(minutes=15)]=0#np.nan
    data2['dayofweek']=data2.index.dayofweek
    data2['hour']=data2.index.hour
    data2['businesshours'] = (data2['hour'] >= 9) & (data2['hour'] < 17)
    data2['AvgPastHour']=data2[0].shift(1).rolling(window=4,center=False).mean()
    cal = calendar()
    holidays = cal.holidays(start=data2.index.min(), end=data2.index.max())
    data2['Date'] = data2.index
    data2['Holiday'] = data2['Date'].isin(holidays)
    data2.drop(['Date'],inplace=True,axis=1)
    d=pd.DataFrame(data2[0].shift(96))
    for i in range(2,9):
        d[i]=data2[0].shift(96*i)
    data2['AvgPastWeekSameTime']=d.sum(axis=1)
    data2['lag1']=data2[0].shift(1)
    data2=data2.bfill()
    return data2


# In[72]:

from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()


from sklearn.metrics import *
def train(data2):
    y_train, X_train=data2.ix[:,data2.columns.isin([0])], data2.ix[:,~data2.columns.isin([0])]
    gbr.fit(X_train,y_train)
    


# In[75]:

#y_pred=[]

def predict():
    pred_list=[]
    split_data(split_number)
    data=feature_generation(df_train)
    train_data=data.iloc[:-1].copy(deep=True)
    test_data=data.iloc[-1,1:].copy(deep=True)
    train(train_data)
    pred=gbr.predict(test_data.values.reshape(1,-1))
    pred_list.append(pred.tolist()[0])
    df_train.loc[df_train.index[-1]+datetime.timedelta(minutes=15)]=pred
    return pred_list

def update_record(user_value):
    data1.loc[df_train.index[-1]+datetime.timedelta(minutes=15)]=user_value
    split_data(split_number+1)


# y_test=df_test
# #def evaluate():
#     #print(explained_variance_score(y_test, np.array(y_pred).astype('float64')))
#     #print(mean_absolute_error(y_test, y_pred))
#     #print(mean_squared_error(y_test, y_pred))
#     #print(median_absolute_error(y_test, y_pred))
#     #print(r2_score(y_test, y_pred))
#     #print(list(zip(gbr.feature_importances_,data1.columns[1:])))

# plt.plot(y_test[0].values[0:100])
# plt.plot(y_pred[0:100])
# plt.show()

# In[ ]:

from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
app = Flask(__name__)
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    user = request.args.get('id')
    if user:
        update_record(user)
    prediction = predict()
        
    return jsonify({'prediction': prediction})
if __name__ == '__main__':
    #clf = joblib.load('model.pkl')
    app.run(port=8080)


# In[ ]:




# In[ ]:




# In[ ]:



