#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Author: Ivan Sheng
Assignment: Alternate Programming Assignment 2
Date Submitted: 5/12/2020
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix,f1_score,precision_score
from sklearn.model_selection import train_test_split,cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


user = r'C:\Users\is104\Documents'
path = user + '\PA2\coronavirusdataset'

f= open(user+'\PA2\PA2_ISHENG_OUT.txt','w+')

search_data = pd.read_csv(path +'\\SearchTrend.csv')
case = pd.read_csv(path +'\\case.csv')
patientroute = pd.read_csv(path +'\\PatientRoute.csv')
patientinfo = pd.read_csv(path +'\\PatientInfo.csv')
region = pd.read_csv(path +'\\Region.csv')
time = pd.read_csv(path +'\\Time.csv')
timeage = pd.read_csv(path +'\\TimeAge.csv')
timegender = pd.read_csv(path +'\\TimeGender.csv')
timeprovince = pd.read_csv(path +'\\TimeProvince.csv')
weather = pd.read_csv(path +'\\Weather.csv')
pop_num = pd.read_csv(path+ '\\SeoulFloating.csv')


# In[3]:


f.write('Files from coronavirusdataset have been read \n');


# In[4]:


case_sum = case[['province','confirmed']].groupby(['province']).sum()
provinceCenter = region.drop(['code'],axis=1).groupby(['province']).mean()[['latitude','longitude']]
provinceCenter.index = provinceCenter.index.set_names(['province'])
case_byProvince = case_sum.join(provinceCenter).reset_index()

#append region data to province data
region = region[np.mod(region['code'],1000)==0]
case_byProvince = case_byProvince.join(region.set_index('province').drop(['latitude','longitude'],axis=1), on='province')


# # Problem 1-1

# In[5]:


fig = px.scatter_mapbox(case_byProvince,
                        lat='latitude', lon='longitude', 
                        size='confirmed',
                        color='confirmed',
                        color_continuous_scale=px.colors.sequential.Sunsetdark,
                        size_max=50,
                        opacity=0.7,
                        hover_name='province')

fig.update_layout(mapbox_style= 'carto-positron', 
                  mapbox_zoom=6,
                  title = 'Confirmed Cases in South Korea',
                  margin={"r":0,"t":25,"l":0,"b":0})

fig.show()


# In[6]:


f.write('Problem 1-1 Visuals Generated \n');


# # Problem 1-2

# In[7]:


daegu = patientroute[patientroute['province']=='Daegu']
route_fig = px.line_mapbox(daegu, 
                     lat='latitude', lon='longitude', 
                     color='patient_id')

route_fig.update_layout(mapbox_style= 'carto-positron', 
                        mapbox_zoom=11,
                        title = 'Daegu Case Routes',
                        margin={"r":0,"t":25,"l":0,"b":0},
                        showlegend=False)
route_fig.show()


# For performance reasons, the number of routes displayed were limited. The Daegu region was specifically selected because of the Shincheonji Church of Jesus incident.

# In[8]:


timeprovince = timeprovince.set_index('province').join(provinceCenter).reset_index()


# In[9]:


fig = px.scatter_mapbox(timeprovince,
                        lat='latitude', lon='longitude', 
                        size='confirmed', size_max=50,
                        animation_frame='date',
                        hover_name='province')

fig.update_layout(mapbox_style= "carto-positron", 
                  mapbox_zoom=5.5,
                  title = 'Confirmed Cases in South Korea by Time',
                  margin={"r":0,"t":25,"l":0,"b":0})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 150
fig.layout.sliders[0].active = 1
fig.show()


# In[10]:


f.write('Problem 1-2 Visuals Generated \n');


# # Problem 1-3

# In[11]:


search_data['date'] = search_data['date'].astype('datetime64[ns]')
search_data.set_index('date', inplace=True)
search_data.describe()


# The first order statistics show that coronavirus searches have a standard deviation that's almost 5 times its mean. The mean is also significantly higher than cold, flu, and pneumonia searches despite its quartile limits being less than the other three. This is probably due to the fact that it maxes out at 100, while the other three don't even come close. After looking at both the first order test statistics along with the line plots, a box and whisker plot that includes the Coronavirus timeframe would include significant outliers, so two box plots will be created: with and without the pandemic timeline.

# In[12]:


np.log(search_data).plot.box()
plt.title('Search Trends Distibution w/ COVID19 Pandemic Timeline')
np.log(search_data.iloc[:1450,:]).plot.box()
plt.title('Search Trends Distibution w/o COVID19 Pandemic Timeline')
plt.show()


# The number of outliers above the max whisker for coronavirus searches is significant, which is expected from looking at the 2020 search data. Because a relationship between pneumonia and coronavirus searches was observed during the 2020 timeframe, it makes sense that almost all of the outliers for pneumonia disappeared when that timeframe was cut off. Unfortunately, there still remains a significant number of cold search outliers.

# In[13]:


plt.figure()
plt.plot(search_data)
plt.title('Virus Search Trends')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Search Volume')
plt.legend(['cold', 'flu', 'pneumonia', 'coronavirus']);
plt.show()


# Looking at the raw search data, it’s difficult to notice any details because Coronavirus searches in South Korea have drastically increased in volume compared to historical search volume of the cold, flu, and pneumonia. However, three things to note are the spikes in flu searches near the end of 2016, cold searches around the first quarter of 2019, and pneumonia searches mimicking Coronavirus searches in 2020.
# 
# * The spike in flu searches can be attributed to the bird flu outbreak, H5N8, that affected South Korea. 
# * The spike in cold searches in 2019 could possibly be attributed to the rising tensions and mentions of a Cold War between the US, North Korea, and South Korea during March. 
# * Because pneumonia is telling symptom of coronavirus, its search trends will mimic coronavirus searches.
# 
# To analyze the data in a view that would allow trends to show, the axes must be set to cut off the Coronavirus pandemic, and the data itself should be smoothed out by applying a rolling n-window median.

# In[14]:


#smooth the data out with a rolling median with a window of size 5
window_size = 5
smooth_search = search_data.rolling(window_size,center=True).median().dropna()


# In[15]:


plt.figure()
plt.plot(smooth_search.iloc[:1450,:])
plt.title('Virus Search Trends - Rolling 5 Window Median')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Search Volume')
plt.legend(['cold', 'flu', 'pneumonia', 'coronavirus'])
plt.show()


# After these techniques were applied, it can be observed that there is an obvious seasonal pattern for flu searches that occurs at the end of the year, with the exception of 2019 where there was an increase in searches within the first half of the year. As for cold and pneumonia searches, they peak more often than flu searches – approximately every half year – but their second peaks line up with flu search peaks in December.

# In[16]:


f.write('Problem 1-3 Visuals Generated \n');


# # Problem 1-4
# 
# Seasonality was identified in the previous analysis, but are there any correlations that would show a relationship between virus search trends and weather patterns? 
# 
# As mentioned previously, a 5-window rolling median had been applied to the search data to smooth it out – the same procedure will be applied to the weather data as well to filter and smooth out the noise seen in sporadic daily data. 
# 
# Additionally, both pre-COVID-19 and post-COVID-19 timeframes will be correlated.

# In[17]:


weather_byDate = weather.drop('code',axis=1).groupby('date').mean()
smooth_temp = weather_byDate.iloc[:,1:].rolling(window_size,center=True).median().dropna()
smooth_combined = smooth_search.join(smooth_temp)

corr = smooth_combined.corr()
corr = corr.iloc[:, 0:4] 
print(corr)


# At first, there isn’t anything too strong between weather data and search trends. Generally, there is a negative relationship between temperature data and flu searches: as temperature decreases, flu searches increase, however, nothing else is particularly notable. It’s worth mentioning that strong correlations are observed between Coronavirus, the common cold, and pneumonia searches because when looking at the search trend graphs, these two viruses were also searched in conjunction with the rise of Coronavirus searches. 

# In[18]:


truncated_combined = smooth_combined.iloc[:1450,:]
corr = truncated_combined.corr()
corr = corr.iloc[:, 0:4]
print(corr)


# When looking at the data before the Coronavirus pandemic began, there are negative correlations similar to the flu: as temperatures decrease, searches for all four viruses will increase. There is also not as strong of a correlation between Coronavirus, the common cold, and pneumonia.

# In[19]:


f.write('Problem 1-4 Correlations generated: \n')
f.write(str(corr))
f.write('\n');


# # Problem 1-5

# In[20]:


patientinfo['sex'] = patientinfo['sex'].map({'male': 1, 'female': -1})
patientinfo['released_date'] = patientinfo['released_date'].fillna(patientinfo['deceased_date'])
patientinfo[['released_date','confirmed_date']] = patientinfo[['released_date','confirmed_date']].astype('datetime64[ns]') 
patientinfo = patientinfo[patientinfo['released_date'].notnull()]
patientinfo['days_since_confirm'] = (patientinfo['released_date']-patientinfo['confirmed_date']).dt.days

conf_vs_dec = patientinfo[patientinfo['state'] != 'isolated']
conf_vs_dec = conf_vs_dec[conf_vs_dec['days_since_confirm']>=0]

region = region[region['province']==region['city']]

conf_vs_dec_region = conf_vs_dec.set_index('province').join(region.set_index('province').drop(['city'],axis=1)).reset_index()
deceased = conf_vs_dec_region[conf_vs_dec_region['state']=='deceased']
deceased_region = deceased.groupby('province')['patient_id'].count()
print(deceased_region)


# When filtering the patientinfo dataset for only those who have passed due to from the, there were only 61 patients who had both confirmation date and date of passing. Of those 63 patients:
# 
# * 39 were from the province of Gyeongsangbuk-do, which has the second highest average elderly population ratio and elderly alone ratio.
# 
# * 20 were from Daegu, which was the first Coronavirus epicenter outside of China due to Shincheonji Church of Jesus followers who purposefully spread the virus.
# 

# In[21]:


features = ['sex','birth_year','days_since_confirm','latitude','longitude','elderly_population_ratio','elderly_alone_ratio']
sns.pairplot(conf_vs_dec_region, diag_kind = 'kde', hue='state', vars = features)
plt.show()


# Looking at the pairplots of the patientinfo dataset filtered to patients who had both a date of confirmation and date of release or passing, it seems that elderly population and alone ratios are actually poor class separators for released and deceased. A significant majority of deceased patients fall within very similar longitudes and latitudes – this was previously referenced as Daegu (35.872, 128.602) and Gyeongsangbuk-do (36.576,128.506). 
# 
# Both birth year and days since confirmation seem to be good separators – it seems that those who have passed weren’t diagnosed for long and were older in age. 
# 
# Diving a bit further into birth year and days since confirmation, the average number of days since confirming a case before passing is 8.88 days with an average birth year of 1944 (76 years old) while a release is 22.42 days with an average birth year of 1976 (54 years old). This raises concerns because bed occupancy would take a bit more than two-thirds of a month, which could overload hospitals if nothing was done to slow down the rate of spread.

# In[22]:


byState = conf_vs_dec.groupby('state').mean()[['birth_year','days_since_confirm']]
print(byState)


# In[23]:


#calculate the number of active confirmed cases
active_confirmed = timeage['confirmed']-timeage['deceased']
timeage.insert(4,'active_confirmed',active_confirmed)

#calculate percent compositions
timeage['percent_active_confirmed'] = timeage['active_confirmed']/timeage['confirmed']*100
timeage['percent_deceased'] = timeage['deceased']/timeage['confirmed']*100

timeage_melt_perc = pd.melt(timeage.drop(['time','confirmed','active_confirmed','deceased'],axis=1),
                            id_vars=['date','age'],
                            var_name='status',
                            value_name='percent')

timeage_melt_num = pd.melt(timeage.drop(['time','confirmed','percent_active_confirmed','percent_deceased'],axis=1),
                           id_vars=['date','age'],
                           var_name='status',
                           value_name='count')

timeage_melt=timeage_melt_num.join(timeage_melt_perc['percent'])


# In[24]:


byAge = px.bar(timeage_melt,
               x='age',
               y='percent', 
               text='count', 
               color='status', 
               animation_frame='date')

byAge.update_layout(
    title="Active Confirmed Cases vs Fatalities in South Korea",
    xaxis_title="Age",
    yaxis_title="Percent",
    uniformtext_mode="hide")

byAge.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
byAge.layout.sliders[0].active = 1
byAge.show()

activeConf_byTime = px.line(timeage_melt[timeage_melt['status']=='active_confirmed'],
                            x='date',y='count', 
                            color = 'age')

activeConf_byTime.update_layout(title="Actived Confirmed Cases in South Korea by Time")
activeConf_byTime.show()

deceased_byTime = px.line(timeage_melt[timeage_melt['status']=='deceased'],
                          x='date',y='count', 
                          color = 'age')

deceased_byTime.update_layout(title="Fatalities in South Korea by Time")
deceased_byTime.show()


# Graphing out the number of confirmed cases and fatalities in aggregate and by time, the data supports the observation that the elderly are more susceptible to succumbing to COVID-19. A couple things pop out:
# * In comparison to global death rates from Statista as of February 11th 2020, South Korea has a significantly higher mortality rate of their 80+ population.
#  * The global rate for 80+ is 14.8% versus South Korea’s 24.3%
# * South Korea has a high number of confirmed cases for the age group of 20-29 years old. 
#  * In comparison, Canada’s highest distribution of cases by age group is between 50-59 years old and 80+ years old.
# 

# In[25]:


#calculate the number of active confirmed cases
active_confirmed = timegender['confirmed']-timegender['deceased']
timegender.insert(4,'active_confirmed',active_confirmed)

gender_melt = pd.melt(timegender.drop(['time','confirmed'],axis=1),
                      id_vars=['date','sex'],
                      var_name='status',
                      value_name='count')

byGender = px.bar(gender_melt,
                  x='sex',y='count', 
                  text='count', 
                  color='status', 
                  animation_frame='date')

byGender['layout']['yaxis1'].update(title='', range=[0, 7500], autorange=False)
byGender.show()


# Looking at the status of COVID-19 patients by gender, while there are over 2,000 more confirmed cases for females, both the volume of fatalities along with the rate are less than that of males. The number of fatalities didn't make up a large enough portion of confirmed cases to warranty viewing the patient status by gender as a percentage bar chart - both were between 2-3%.

# In[26]:


f.write('Problem 1-5 Analysis Completed. \n');


# # Problem 2-1

# In[27]:


week1_path = user+'\PA2\wk1'
week2_path = user+'\PA2\wk2'
week3_path = user+'\PA2\wk3'
week5_path = user+'\PA2\wk5'

week1_data = pd.read_csv(week1_path+'\\train.csv')
#rename the columns for week1_data to match weeks 2 and 3
week1_data.columns = ['Id', 'Province_State', 'Country_Region', 'Lat', 'Long', 'Date', 'ConfirmedCases', 'Fatalities']
week2_data = pd.read_csv(week2_path+'\\train.csv')
week3_data = pd.read_csv(week3_path+'\\train.csv')
week5_data = pd.read_csv(week5_path+'\\train.csv')


# In[28]:


f.write('Files from weekly COVID19 files have been read. \n');


# In[29]:


#create keys to provide latitude and longitude data to other week files
week1_data['concat']=week1_data['Province_State'].fillna('') + ', ' + week1_data['Country_Region']
latlong = week1_data[['concat','Lat','Long']].groupby('concat').mean()
week3_data = week3_data.set_index(week3_data['Province_State'].fillna('') + ', ' + week3_data['Country_Region'])
week3_wLatLong = week3_data.join(latlong)


# In[30]:


fig = px.scatter_mapbox(week3_wLatLong,
                        lat='Lat', lon='Long', 
                        size='ConfirmedCases', size_max=30,
                        animation_frame='Date',
                        zoom=.8,
                        hover_name=week3_wLatLong.index)

fig.update_layout(mapbox_style= "carto-positron", 
                  margin={"r":0,"t":25,"l":0,"b":0},
                  title = 'COVID19 GLobal Spread')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 150
fig.layout.sliders[0].active = 1
fig.show()


# In[31]:


f.write('Problem 2-1 Visuals Generated \n');


# # Problem 2-2

# In[32]:


byCountryDate = week3_wLatLong.groupby(['Country_Region','Date']).sum().reset_index()
fig = px.line(byCountryDate,
              x='Date',
              y='ConfirmedCases',
              color='Country_Region')
fig.update_layout(title = 'Timeline of Confirmed Cases by Country')
fig.show()

fig = px.line(byCountryDate,
              x='Date',
              y='Fatalities',
              color='Country_Region')
fig.update_layout(title = 'Timeline of Fatalities by Country')
fig.show()


# In[33]:


f.write('Problem 2-2 Visuals Generated \n');


# # Problem 2-3

# In[34]:


week1_data.describe()


# Looking at the timeline of confirmed cases by country, every country except the US and France appear to be experiencing leveling off in daily confirmed cases – these countries’ growths are beginning to take a logistic shape instead of exponential. With the addition of the United Kingdom, similar trends can be seen when looking at daily fatalities.
# 
# Analyzing the first order statistics for week 1 training data, most of the confirmed cases don’t appear until the 3rd quartile because the data for week 1 begins tracking since late January 2020, when the large majority of documented confirmed cases were in China. Since most of the data will fall in the last quartile and each country was infected at different time periods, it’d be best to look at countries isolated since virus spread can be modeled as exponential growth in the early stages, and logistic later on. We can expect the US, UK, and France to follow exponential growth, and the rest of the world to follow logistic growth based on the line graphs.

# In[35]:


f.write('Problem 2-3 First order statistics and previously generated graphs analyzed \n');


# # Problem 2-4

# In[36]:


def groupbyDate(df, country):
    """
    Group dataframe by date and filtered by country
    Parameters:
        df (Dataframe) - dataframe to be grouped
        country (String) - the country to filter day by
    Return:
        country_byDate (Dataframe) - a dataframe grouped by date and filtered for country parameter
    """
    
    byCountry = df[df['Country_Region']==country].drop(['Id'],axis=1)
    country_byDate = byCountry.groupby('Date').sum()
    country_byDate['Days'] = range(len(country_byDate)) 
    return country_byDate

def linear_reg(df):
    """
    Run linear regression from statsmodel package
    Parameters:
        df (Dataframe) - dataframe containing the data to run the linear regression off of
    Return:
        result (RegressionResults) - the resulting linear regression model
    """
    
    log_cases = np.log1p(df['ConfirmedCases'])
    X = range(len(log_cases))
    X = sm.add_constant(X)
    X[:,1] += min(df['Days'])

    y = log_cases
    mod = sm.OLS(y,X)
    result = mod.fit()
    print(result.summary())
    return result

def get_future_pred(df):
    """
    Gets the predictions for days outside of the modeled dates
    Parameters:
        df (Dataframe) - dataframe of the original data that the linear regression was ran on
    Return:
        future_pred (Dataframe) - dataframe of the future days and predictions
    """
    
    future = range(len(df))
    future = sm.add_constant(future)
    future[:,1] += min(df['Days'])
    future_pred = np.exp(result.predict(future))
    future_pred=pd.DataFrame(future_pred, columns=['Prediction'],index=df.index)
    return future_pred

def graph_compare(df, future_df, title):
    """
    Graphs the observed and predictions
    Parameters:
        df (Dataframe) - dataframe of the following week's observed data
        future_df (Dataframe) - dataframe of the future predictions
        title (String) - the title of the plot
    Return:
        None
    """
    
    fig, ax = plt.subplots(1,1)

    plt.plot(future_df['Prediction'], label = 'Prediction')
    ax.plot(df['ConfirmedCases'], label = 'Observed Week 2')
    plt.xticks(rotation=45)
    plt.xlabel('Dates')
    plt.ylabel('Confirmed Cases')
    plt.title(title)
    plt.legend()

    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.show()


# In[37]:


#group US data by date and genereate predictions
usa_byDate_wk1 = groupbyDate(week1_data,'US')
result = linear_reg(usa_byDate_wk1)

pred = np.exp(result.predict())
train_rmse = rmse(usa_byDate_wk1['ConfirmedCases'],pred)
print('RMSE: ' + str(round(train_rmse,2)))


# Analyzing the first order statistics for week 1 training data, most of the confirmed cases don’t appear until the 3rd quartile because the data for week 1 begins tracking since late January 2020, when the large majority of documented confirmed cases were in China. Since most of the data will fall in the last quartile and each country was infected at different time periods, it’d be best to look at countries isolated since virus spread can be modeled as exponential growth in the early stages, and logistic later on. We can expect the US, UK, and France to follow exponential growth, and the rest of the world to follow logistic growth based on the line graphs.
# 
# A linear regression was performed on week 1 US data after log transforming the y-axis: confirmed cases of COVID19. Initially this regression was performed on data for all available dates even if there were no recorded cases yet. This resulted in the following model: 

# In[38]:


#group US data by date and genereate predictions
usa_byDate_wk2 = groupbyDate(week2_data,'US')
future_pred = get_future_pred(usa_byDate_wk2)
graph_compare(usa_byDate_wk2, future_pred, 'Linear Regression of Week 1 Data')


# Despite the high R2 value, the linear regression seems to be a poor predictor for COVID19’s spread in the United States; this could be attributed to the significant amount of time that the data remained at zero.  In an attempt to improve the resulting linear regression, the data can be cut to only preserve data starting from when confirmed COVID19 cases began appearing. 

# In[39]:


# filter for where there are confirmed cases
usa_byDate_wk1 = usa_byDate_wk1[usa_byDate_wk1['ConfirmedCases']>0]
result = linear_reg(usa_byDate_wk1)

pred = np.exp(result.predict())
train_rmse = rmse(usa_byDate_wk1['ConfirmedCases'],pred)
print('RMSE: ' + str(round(train_rmse,2)))


# Looking at the regression results, there are improvements across the board with a higher coefficient of determination, as well as an RSME that’s almost half the pervious iteration’s RSME.

# In[40]:


# filter for where there are confirmed cases
usa_byDate_wk2 = usa_byDate_wk2[usa_byDate_wk2['ConfirmedCases']>0]
future_pred = get_future_pred(usa_byDate_wk2)

graph_compare(usa_byDate_wk2, future_pred, 'Linear Regression of Week 1 Data w/o Zeros')


# Based on our improved linear regression, the actual rate of confirmed COVID19 cases appears to be less than what the model is predicting by March 31st. The decreased confirmation rate also begins shortly after quarantine measures were taken on March 19th, however week 3 should be observed to confirm that the number of confirmed cases is continuing to flatten.

# In[41]:


f.write('Problem 2-4 Week 1 Regressions complete: \n')
f.write(str(result.summary()))
f.write('\n');


# # Problem 2-5

# In[42]:


# group US data by date and filter for where there are confirmed cases
usa_byDate_wk2 = groupbyDate(week2_data,'US')
usa_byDate_wk2 = usa_byDate_wk2[usa_byDate_wk2['ConfirmedCases']>0]
result = linear_reg(usa_byDate_wk2)

pred = np.exp(result.predict())
train_rmse = rmse(usa_byDate_wk2['ConfirmedCases'],pred)
print('RMSE: ' + str(round(train_rmse,2)))


# In[43]:


# group US data by date and filter for where there are confirmed cases
usa_byDate_wk3 = groupbyDate(week3_data,'US')
usa_byDate_wk3 = usa_byDate_wk3[usa_byDate_wk3['ConfirmedCases']>0]
future_pred = get_future_pred(usa_byDate_wk3)

graph_compare(usa_byDate_wk3, future_pred, 'Linear Regression of Week 2 Data')


# Comparing the predictions for week 3 with the improved regression versus the actual observed data for week 3, social distancing policies seem effective since the actual spread of Coronavirus is no longer following exponential growth.

# In[44]:


x_range = np.linspace(1, 1000000, 20000) 

#mortality rate annnounced on March 3rd from WHO
y = 0.034*x_range
who_val = pd.DataFrame({'x':x_range, 'y':y})

byCountry = week3_data.groupby(['Country_Region']).sum().reset_index()
mod = sm.OLS(byCountry['Fatalities'], byCountry['ConfirmedCases'])
res = mod.fit()
print(res.summary())
param = res.params

y_pred = param['ConfirmedCases']*x_range
val_pred = pd.DataFrame({'x':x_range, 'y':y_pred})

scatter_wk = week3_data.groupby(['Date','Country_Region']).sum().reset_index()
scatter_wk['death ratio'] = scatter_wk['Fatalities']/scatter_wk['ConfirmedCases']
scatter_wk['death ratio'].fillna(0,inplace=True)

scatter = px.scatter(scatter_wk,
                     x='ConfirmedCases', y='Fatalities', 
                     size='death ratio', size_max=100,
                     animation_frame='Date',
                     hover_name='Country_Region',
                     log_x=True,
                     range_x=[1,1000000], range_y=[1,20000])

pred = px.line(val_pred,
               x='x',y='y')

who_rate = px.line(who_val,
                   x='x',y='y')

pred.update_traces(line_color='#32CD32',
                   name = 'Model (' + str(round(param['ConfirmedCases'],3))+')',
                   showlegend=True)

who_rate.update_traces(line_color='#FF0000',
                       name = 'WHO (0.034)', 
                       showlegend=True)

scatter.add_trace(who_rate.data[0])
scatter.add_trace(pred.data[0])

scatter.update_layout(margin={"r":0,"t":25,"l":0,"b":0},
                  title = 'Forecasting Current Rate of Fatalities')

scatter.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
scatter.layout.sliders[0].active = 1

scatter.show()


# According to WHO Director-General, it was reported on March 3rd that, “globally, about 3.4% of reported COVID-19 cases have died”. Graphing out the log of confirmed cases versus fatalities, it can be observed that while the majority of countries fall fairly close to that 3.4% mortality line, by April, a better fit would be 4.3% mortality. An article published by the New York Times on April 17th cited a 4.3% mortality rate.
# 
# Around March 16th is the timeframe where the global data no longer follows a 3.4% mortality rate. Looking at the animation, it’s observed that Italy, Spain, France, and the United Kingdom began to report higher mortality rates.
# 
# Sources: 
# - https://www.who.int/dg/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---3-march-2020
# - https://www.nytimes.com/2020/04/17/us/coronavirus-death-rate.html

# In[45]:


f.write('Problem 2-5 Week 2 Regressions completed \n')
f.write(str(result.summary()))
f.write('\n');


# # Problem 2-6

# In[46]:


def dateToDays(df):
    """
    Calculates Days since the earliest date available in the Date column
    Parameters:
        df (Dataframe) - dataframe containing the Date-column in question
    Return:
        df (Dataframe) - same input dataframe but with the 'Days' column appended
    """
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - min(df['Date'])).dt.days
    return df

def create_YesNoCol(df,col):
    """
    Calculates Days since the earliest date available in the Date column
    Parameters:
        df (Dataframe) - dataframe containing the Date-column in question
        col (Series) - Column within dataframe to base the 1/0 column off of
    Return:
        df (Dataframe) - same input dataframe but with the 'Yes_No' column appended
    """
    
    col_val = np.array(df[col])
    df['Yes_No'] = np.where(col_val > 0,1,0)
    return df

def create_SVM(X_train,y_train,X_test,y_test, kernel, f):
    """
    Creates an SVM model, calculates its score, and prints the confusion matrix
    Parameters:
        X_train (Dataframe) - dataframe containing all independent variables to train model on
        y_train (Series) - series containing the corresponding classes to train the model on
        X_test (Dataframe) - dataframe containing all independent variables to test
        y_test (Series) - series containing the corresponding classes to test
        f (object) - text tile to be written to
    Return:
        result (object) - resulting model after fitting
    """
    
    clf = SVC(kernel=kernel)
    result = clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    pred_class_SVM = clf.predict(X_test)
    
    print('Training Score: ' + str(round(train_score,3)))
    print('Test Score: ' + str(round(test_score,3)))
    
    f1 = f1_score(y_test,pred_class_SVM,average='weighted')
    prec = precision_score(y_test,pred_class_SVM,average='weighted')

    print('F1 Score: ' + str(round(f1,3)))
    print('Precision Score: ' + str(round(prec,3)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,pred_class_SVM))
    print()
    
    plt.scatter(X_test['birth_year'],X_test['days_since_confirm'],c=pred_class_SVM)
    plt.show()
    
    return result

def create_LDA(X_train,y_train,X_test,y_test,solver, f):
    """
    Creates an LDA model, calculates its score, and prints the confusion matrix
    Parameters:
        X_train (Dataframe) - dataframe containing all independent variables to train model on
        y_train (Series) - series containing the corresponding classes to train the model on
        X_test (Dataframe) - dataframe containing all independent variables to test
        y_test (Series) - series containing the corresponding classes to test
        f (object) - text tile to be written to
    Return:
        result (object) - resulting model after fitting
    """
    
    clf = LDA(solver=solver)
    result = clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    pred_class_LDA = clf.predict(X_test)
    
    print('Training Score: ' + str(round(train_score,3)))
    print('Test Score: ' + str(round(test_score,3)))
    
    f1 = f1_score(y_test,pred_class_LDA,average='weighted')
    prec = precision_score(y_test,pred_class_LDA,average='weighted')

    print('F1 Score: ' + str(round(f1,3)))
    print('Precision Score: ' + str(round(prec,3)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,pred_class_LDA))
    print()
    
    plt.scatter(X_test['birth_year'],X_test['days_since_confirm'],c=pred_class_LDA)
    plt.show()
    
    return result

def cross_val_svm(X,y,kernel):
    clf = SVC(kernel=kernel)
    scores = cross_val_score(clf, X, y, cv=10, scoring='f1_weighted')
    print("F1 Weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores

def cross_val_lda(X,y,solver):
    clf = LDA(solver=solver)
    scores = cross_val_score(clf, X, y, cv=10, scoring='f1_weighted')
    print("F1 Weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores


# # Patient Classification

# In[47]:


def generate_synth_data(df,n):
    col = np.shape(df)[1]
    r = np.random.rand(n,col)
    cov = np.cov(df.T)
    r_Cov = np.dot(r,cov)
    
    featureMin = np.amin(r_Cov,axis=0)
    featureMax = np.amax(r_Cov,axis=0)
    maxminDiff = featureMax-featureMin
    
    df_min = np.amin(df,axis=0)
    df_max = np.amax(df,axis=0)
    
    orig_avg = np.mean(df,axis=0)
    
    synth = ((r_Cov-featureMin)/maxminDiff*(df_max-df_min)+df_min)
    synth_avg = np.mean(synth,axis=0)
    
    #synth_centered = synth - (synth_avg-orig_avg)
    synth_centered = synth
    return synth_centered

def mahalanobis(irisClass):
    
    mu = np.mean(irisClass, axis=0)
    diff = irisClass - mu
    cov = np.cov(irisClass.T)
    inv_cov = np.linalg.inv(cov)
    mah_mtx = np.dot(np.dot(diff, inv_cov),diff.T)
    d2 = mah_mtx.diagonal()
    return d2

def criticalValue(alpha,n,numVars):
    
    p = 1-alpha
    dfn = numVars
    dfd = n - numVars - 1
    inv_f = scipy.stats.f.ppf(p,dfn,dfd)
    num = dfn * (n-1)**2
    den = n*(dfd) + inv_f
    cv = num/den
    return cv

def removeOutliers(irisClass, alpha, numVars):
    
    d2 = mahalanobis(irisClass)
    cv = criticalValue(alpha,len(irisClass),numVars)
    outliers = np.where(d2 > cv)[0]
    leftover = np.delete(irisClass,outliers, 0)
    return leftover 


# In[48]:


conf_vs_dec_region['state'] = conf_vs_dec_region.state.map(dict(released=1, deceased=0))


# In[112]:


noNull = conf_vs_dec_region[~conf_vs_dec_region['birth_year'].isnull()]

#col = ['birth_year','days_since_confirm','latitude','longitude']
col = ['birth_year','days_since_confirm']
dead = noNull[noNull['state']==0][col]

plt.figure(figsize=(15,10))
plt.title('COVID19 Fatalities in South Korea')
plt.xlabel('Birth Year')
plt.ylabel('Days Since Confirmation')
plt.scatter(dead['birth_year'],dead['days_since_confirm'],label = 'Outliers')
plt.ylim(-5, 50)
plt.xlim(1920, 1990)

dead = dead.values
leftover = removeOutliers(dead,0.05,4)

plt.scatter(leftover[:,0],leftover[:,1],label='Non-Outliers')

X = noNull[col]
y = noNull['state']

synth = generate_synth_data(leftover,350)
synth_noZero = synth[synth[:,1]>=0]
synth_noZero = synth_noZero[synth_noZero[:,0]>1928]
X_syn = pd.DataFrame(synth_noZero,columns = col)

synth_size = len(X_syn)
print(synth_size)
synth_class = np.zeros(synth_size)
y_syn = pd.Series(synth_class)

plt.scatter(X_syn['birth_year'],X_syn['days_since_confirm'],label='Synthetic')
plt.legend()


# In[113]:


X = X.append(X_syn, ignore_index=True)
y = y.append(y_syn, ignore_index=True)

print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[114]:


print('SVM - Linear')
create_SVM(X_train,y_train,X_test,y_test,'linear',f)
print('SVM - RBF')
create_SVM(X_train,y_train,X_test,y_test,'rbf',f)


# In[115]:


print('SVM - Linear')
cross_val_svm(X_train,y_train,'linear');
print('SVM - RBF')
cross_val_svm(X_train,y_train,'rbf');


# In[116]:


print('LDA - SVD')
create_LDA(X_train,y_train,X_test,y_test,'svd',f)

print('LDA - Eigen')
create_LDA(X_train,y_train,X_test,y_test,'eigen',f)


# In[117]:


print('LDA - SVD')
cross_val_lda(X_train,y_train,'svd');

print('LDA - Eigen')
cross_val_lda(X_train,y_train,'eigen');


# # Problem 3-1

# In[55]:


def melt_frame(df, col_name):
    """
    Melts the dataframe
    Parameters:
        df (Dataframe) - dataframe containing the column to melt
        col_name (Series) - the column to melt
    Return:
        df (Dataframe) - dataframe containing the melted column
    """
    
    df = pd.melt(df, id_vars=['Province/State','Country/Region','Lat','Long'], var_name = 'Date', value_name = col_name)
    df.fillna('',inplace=True)
    df.loc[df['Province/State'] == '','Province/State'] =df['Country/Region']
    df['Combined_Key'] = df['Province/State']+ ', '+df['Country/Region']
    return df


# In[56]:


path = user+'\PA2\\novel-corona-virus-2019-dataset\\time_series_covid_19_'

confirmed = pd.read_csv(path +'confirmed.csv')
confirmed_us = pd.read_csv(path +'confirmed_US.csv')
deceased = pd.read_csv(path +'deaths.csv')
deceased_us = pd.read_csv(path +'deaths_US.csv')
recovered = pd.read_csv(path +'recovered.csv')


# In[57]:


#fix canada because it's grouped in one data source but broken out by province/state in another
canada_conf = confirmed[confirmed['Country/Region']=='Canada']
canada_dec = deceased[deceased['Country/Region']=='Canada']
canada_rec = recovered[recovered['Country/Region']=='Canada']

canada_conf_group = canada_conf.groupby('Country/Region').sum().reset_index()
canada_dec_group = canada_dec.groupby('Country/Region').sum().reset_index()

canada_conf_group[['Lat','Long']] = canada_rec[['Lat','Long']].values
canada_dec_group[['Lat','Long']] = canada_rec[['Lat','Long']].values

confirmed_woCanada = confirmed[confirmed['Country/Region']!='Canada']
deceased_woCanada = deceased[deceased['Country/Region']!='Canada']

confirmed = confirmed_woCanada.append(canada_conf_group, ignore_index=True, sort=False)
deceased = deceased_woCanada.append(canada_dec_group, ignore_index=True, sort=False)


# In[58]:


#preparation for graphing
confirmed = melt_frame(confirmed,'confirmed').sort_values(['Date','Lat','Long']).reset_index(drop=True)
deceased = melt_frame(deceased,'deceased').sort_values(['Date','Lat','Long']).reset_index(drop=True)
recovered = melt_frame(recovered,'recovered').sort_values(['Date','Lat','Long']).reset_index(drop=True)

confirmed_byCountry = confirmed.groupby(['Country/Region','Date']).sum()
deceased_byCountry = deceased.groupby(['Country/Region','Date']).sum()
recovered_byCountry = recovered.groupby(['Country/Region','Date']).sum()

latlong_byCountry = confirmed.groupby(['Country/Region']).mean().drop('confirmed',axis=1)

allStats_byCountry = confirmed_byCountry.join(deceased_byCountry['deceased']).join(recovered_byCountry['recovered']).reset_index()
allStats_byCountry.drop(['Lat','Long'],axis=1,inplace=True)
allStats_byCountry=allStats_byCountry.merge(latlong_byCountry,left_on='Country/Region', right_on=latlong_byCountry.index)

#precent composition calculations
allStats_byCountry['Date'] = allStats_byCountry['Date'].astype('datetime64[ns]')
allStats_byCountry['active_confirmed'] = allStats_byCountry['confirmed']-allStats_byCountry['deceased']-allStats_byCountry['recovered']
allStats_byCountry['perc_deceased'] = allStats_byCountry['deceased']/allStats_byCountry['confirmed']*100
allStats_byCountry['perc_recovered'] = allStats_byCountry['recovered']/allStats_byCountry['confirmed']*100
allStats_byCountry['perc_active'] = allStats_byCountry['active_confirmed']/allStats_byCountry['confirmed']*100

allStats_perc = pd.melt(allStats_byCountry.drop(['confirmed','Lat','Long','active_confirmed','deceased','recovered'],axis=1),
                        id_vars=['Date','Country/Region'],var_name='status',value_name='percent')

allStats_num = pd.melt(allStats_byCountry.drop(['confirmed','Lat','Long','perc_active','perc_deceased','perc_recovered'],axis=1),
                       id_vars=['Date','Country/Region'],var_name='status',value_name='count')
allStats_melt = allStats_num.join(allStats_perc['percent']).fillna(0)


# In[59]:


allStats_bylatlong = confirmed.join(deceased['deceased']).join(recovered['recovered'])
allStats_bylatlong['Date'] = allStats_bylatlong['Date'].astype('datetime64[ns]')
allStats_bylatlong.sort_values(['Date','Lat','Long'], inplace=True)

allStats_latlong_melt =  pd.melt(allStats_bylatlong.drop(['Province/State','Country/Region'],axis=1),
                                 id_vars=['Date','Combined_Key','Lat','Long'],
                                 var_name='status',
                                 value_name='count')

fig = px.scatter_mapbox(allStats_latlong_melt,
                        lat='Lat', lon='Long', 
                        size='count', size_max=30,
                        color = 'status',
                        opacity=0.7,
                        animation_frame=allStats_latlong_melt['Date'].astype(str),
                        zoom=0.1,
                        hover_name='Combined_Key')

fig.update_layout(mapbox_style= "carto-positron",
                 margin={"r":0,"t":0,"l":0,"b":0})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
fig.show()


# In[60]:


f.write('Problem 3-1 Visualizations completed \n');


# # Problem 3-2

# In[61]:


#calculate top infected countires
most_cases = confirmed_byCountry.groupby('Country/Region').sum().sort_values('confirmed', ascending=False).reset_index()[:10]
topStats_melt = allStats_melt[allStats_melt['Country/Region'].isin(most_cases['Country/Region'])].sort_values(['Country/Region','Date','status'])

topStats_viz = px.bar(topStats_melt,
                      x='Country/Region',
                      y='percent', 
                      text='count', 
                      color='status', 
                      animation_frame=topStats_melt['Date'].astype(str))

topStats_viz.update_layout(
    title="Case Status of Top 10 Infected Countries by Time",
    xaxis_title="Country",
    yaxis_title="Percent",
    uniformtext_mode="hide")

topStats_viz.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
topStats_viz.update_traces(textposition='inside', textfont_size=10)
topStats_viz.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
topStats_viz.layout.sliders[0].active = 1
topStats_viz.show()


# In[62]:


allStats_melt = allStats_melt.sort_values('Date')
active_case_graph = px.line(allStats_melt[allStats_melt['status']=='active_confirmed'],
              x='Date',
              y='count',
              color = 'Country/Region')

active_case_graph.update_layout(title = 'Active Cases Over Time by Country')
active_case_graph.show()


# In[63]:


f.write('Problem 3-2 Visualizations completed \n');


# # Problem 3-3

# In[64]:


# filter for US only
novel_US = allStats_byCountry[allStats_byCountry['Country/Region']=='US'].sort_values('Date').reset_index(drop=True)
novel_US['Date'] = novel_US['Date'].astype(str)
novel_US.set_index('Date', inplace=True)

# Predict 75 days worth of data from regression create in problem 2
future = range(75)
future = sm.add_constant(future)
future_pred = np.exp(result.predict(future))
future_pred=pd.DataFrame(future_pred, columns=['Prediction'],index=future[:,1])

fig, ax = plt.subplots(1,1)

plt.plot(future_pred['Prediction'], label = 'Model')
ax.plot(novel_US['confirmed'], label = 'Novel Data')
plt.xticks(rotation=45)
plt.xlabel('Dates')
plt.ylabel('Confirmed Cases')
plt.title('Linear Regression of Week 2 Data vs Novel Data')
plt.legend()

tick_spacing = 7
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.show()


# In comparison to the linear regression prediction generated from problem 2 week 2 data, the actual growth of confirmed Coronavirus cases within the United States is significantly slower - possibly because of the social distancing policies implemented to flatten the curve.

# In[65]:


f.write('Problem 3-3 Results have been compared. \n');


# # Problem 3-4 and 3-5

# In[66]:


def calc_rate(df,feature,window):
    """
    Calculate the rate of change per window period
    Parameters:
        df (Dataframe) - dataframe containing the features in question
        feature (String) - the feature name that should have its rate calculated
        window (int) - how far the window should stretch when calculating rate of change
    Return:
        df_rate (Dataframe) - a dataframe that contains the newly calculated rate of change
    """
    
    start = 0
    end = window-1 
    rate_list = []
    date_list = []
    while(end<len(df)):
        diff = df[feature][end]-df[feature][start]
        rate = diff/window
        date_list.append(df['Date'][start])
        start+=window
        end+=window
        rate_list.append(rate)
    df_rate = pd.DataFrame(rate_list,columns=[feature+'_changeRate'],index=date_list)
    return df_rate

def graph_rates(df,country, window):
    """
    Graphs the rate of change per window period
    Parameters:
        df (Dataframe) - dataframe containing the change rate 
        country (String) - the country that should have its rate graphed
        window (int) - how far the window should stretch when calculating rate of change
    Return:
        None
    """
    
    countryStats = df[df['Country/Region']==country].sort_values('Date').reset_index(drop=True)
    confirmed_changeRate = calc_rate(countryStats,'confirmed', window)
    recovered_changeRate = calc_rate(countryStats,'recovered', window)
    changeRates = confirmed_changeRate.join(recovered_changeRate)
    
    fig1 = px.line(changeRates,
              x=changeRates.index,
              y='confirmed_changeRate')
    fig1.update_traces(line_color='#FF0000',name = 'Confirmed Rate', showlegend=True)

    fig2 = px.line(changeRates,
                  x=changeRates.index,
                  y='recovered_changeRate')
    fig2.update_traces(line_color='#32CD32',name = 'Recovery Rate', showlegend=True)
    fig1.add_traces(fig2.data)
    fig1.update_layout(
        title=str(window) +' Day Interval Growth & Recovery Rates - US',
        xaxis_title='Date',
        yaxis_title='Growth Per Day')
    fig1.show()
    


# In[67]:


changeRates = graph_rates(allStats_byCountry,'US',3)


# Filtering the time_series* data down to the US – the growth rate of confirmed cases is stagnating, albeit, still high. However, the recovery rate is steadily increasing. Looking at a country that is on the road to recovery, the recovery rate should overtake the confirmed rate. From a previous analysis, it was seen that in most countries, the number of confirmed cases per day was beginning to stabilize. 

# In[68]:


changeRates = graph_rates(allStats_byCountry,'China',3)


# Looking at China, in particular, the recovery rate crossed the confirmed rate of increase around February 15th and is now on its way towards zero since there are only a few hundred cases remaining.

# In[69]:


changeRates = graph_rates(allStats_byCountry,'Italy',3)


# Similarly, in Italy, the recovery rate has very recently crossed the confirmed growth rate with an experienced downward trend ever since March 19th.

# In[70]:


f.write('Problem 3-4 Visualizations completed. \n');
f.write('Problem 3-5 Visualizations completed. \n');
f.write('Script Completed.')
f.close();

