#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
data=pd.read_csv('E:\diabetes_binary_health_indicators_BRFSS2015 (2).csv')
data=pd.isnull(data)
data
data.head()


# In[ ]:


Goals=data['Goals']
µ_goals=np.mean(Goals)
goals=Goals.sum()
σ_goals=np.std(Goals)
print(σ_goals)
data.shape


# In[4]:


print('In a league of '+str(data.shape[0])+' registered players, there was a total of '+str(goals)+' goals scored in the 2020/2021 season with an average of a goal in every game. '+str(data['Penalty_Goals'].sum())+' out of such was scored by penalties.')


# In[5]:


Penalties=data['Penalty_Goals']
penalties_goals=Penalties.sum()
goals_without_penalties=goals-penalties_goals
print(penalties_goals,goals_without_penalties)


# In[6]:


plt.figure(figsize=(14,7))
datadraw=[goals_without_penalties,penalties_goals]
mylabels=['Goals scored without penalties','Goals scored with penalties']
plt.pie(datadraw,labels=mylabels,autopct='%.0f%%')
plt.show()


# In[7]:


penelties_not_scored=data['Penalty_Attempted'].sum()-penalties_goals
plt.figure(figsize=(14,7))
drawpenelaties=[penelties_not_scored,data['Penalty_Attempted'].sum()]
mylabels2=['Penalties missed','Penalties scored']
plt.pie(drawpenelaties,labels=mylabels2,autopct='%.0f%%')
plt.show()
print('Penalty_Attempted = '+str(data['Penalty_Attempted'].sum()))
print('Penalties missed = '+str(penelties_not_scored))
print('Penalties scored = '+str(penalties_goals))


# In[8]:


penelties_not_scored=data['Penalty_Attempted'].sum()-penalties_goals
plt.figure(figsize=(14,7))
drawpenelaties=[penelties_not_scored,data['Penalty_Attempted'].sum()]
mylabels2=['Penalties missed','Penalties scored']
plt.pie(drawpenelaties,labels=mylabels2,autopct='%.0f%%')
plt.show()
print('Penalty_Attempted = '+str(data['Penalty_Attempted'].sum()))
print('Penalties missed = '+str(penelties_not_scored))
print('Penalties scored = '+str(penalties_goals))


# In[9]:


foriegn_countries=data['Nationality'].unique()
print (str(len(foriegn_countries))+' foriegn countries in England')


# In[10]:


no_of_English_players=(data['Nationality']=='ENG').sum()
no_of_non_English_players=(data['Nationality']!='ENG').sum()


# In[11]:


plt.figure(figsize=(14,7))
data=[no_of_English_players,no_of_non_English_players]
mylabels3=['English players','Foreign players']
plt.pie(data, labels=mylabels3,autopct='%.0f%%')
plt.show()


# In[12]:


data=pd.read_csv('E:\EPL_20_21.csv')
cnties_wt_more_play=data.groupby('Nationality').filter(lambda x:len(x)>1) #countries with more than one player representing
eng_players=data.loc[data['Nationality']=='ENG'].index #england players
cnties_wt_more_play_exc_eng=cnties_wt_more_play.drop(eng_players)
plt.figure(figsize=(20,6))
plt.title("Countries excluding England with more than one player representing")
sns.countplot(x='Nationality', data=cnties_wt_more_play_exc_eng)
plt.ylabel('Players')


# In[13]:


data['Yellow_Cards'].sum()


# In[14]:


data=pd.read_csv('E:\EPL_20_21.csv')
data['Red_Cards'].sum()


# In[15]:


data.loc[(data['Yellow_Cards']==0) & (data['Red_Cards']==0)]['Name'].count()


# In[16]:


yellow_card=data.loc[(data['Yellow_Cards']>0)]
red_card=data.loc[(data['Red_Cards']>0)]
plt.figure(figsize=(20,6))
# epl_without_eng=epl_data.drop(eng_players)
# epl_wt_goals=epl_without_eng.loc[(epl_without_eng['Goals']>0)]
plt.title("Positions")
# fig, ax=plt.subplotd(1,2)
sns.countplot(x='Position', data=yellow_card, color='yellow', order=yellow_card['Position'].value_counts().index, label='yellow card')
sns.countplot(x='Position', data=red_card, color='red', order=yellow_card['Position'].value_counts().index, label='red card')
# sns.countplot(x='Position', data=red_card, color='red')
plt.ylabel('Count')


# In[17]:


data=pd.read_csv('E:\EPL_20_21.csv')
sns.histplot(x=data['Age'])
plt.figure(figsize=(14,6))
plt.axvline(np.median(data['Age']), color='b', linestyle='--',label = 'Median')
plt.axvline(np.mean(data['Age']), color='g', linestyle='-', label = 'Mean')
plt.legend()
plt.show()
array = data['Age']
mode = stats.mode(array)
print(mode[0])


# In[18]:


sns.distplot(data['Age'])
plt.axvline(np.median(data['Age']), color='r', linestyle='--',label = 'Median')
plt.axvline(np.mean(data['Age']), color='g', linestyle='-', label = 'Mean')
plt.legend()
plt.show()


# In[19]:


data=pd.read_csv('E:\EPL_20_21.csv')
goal_clubs=data.loc[(data['Goals']>0)]['Goals'].groupby(data['Club']).sum().sort_values(ascending=False).to_frame()
plt.figure(figsize=(20,6))
plt.title("Number of Goalscorers by Club")
c=sns.barplot(x=goal_clubs.index, y=goal_clubs['Goals'], label='Goalscorers')
plt.ylabel('SUM')
c.set_xticklabels(c.get_xticklabels(),rotation=45)
c


# In[20]:


yellowbar=data['Yellow_Cards'].groupby(data['Club']).sum().sort_values(ascending=False).to_frame()
red_bar=data['Red_Cards'].groupby(data['Club']).sum().to_frame()


# In[21]:


plt.figure(figsize=(20,6))
plt.title("Cummulative number of cards by Club")
c=sns.barplot(x=yellowbar.index, y=yellowbar['Yellow_Cards'], color='yellow', label='Yellow')
sns.barplot(x=yellowbar.index, y=red_bar['Red_Cards'], order=yellowbar.index, color='red', label='Red')
# plt.ylabel('Arrival delay(in mins)')
plt.ylabel('Sum')
plt.legend(loc=1,prop={'size':15})
c.set_xticklabels(c.get_xticklabels(),rotation=45)
c


# In[22]:


sns.boxplot(data['Goals'])
plt.axvline(np.median(data['Goals']), color='b', linestyle='--',label = 'Median')


# In[23]:


data = data[data['Goals'] > 15]
plt.figure(figsize=(20,6))
ax = sns.barplot(x=data['Name'], y=data['Goals'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()


# In[24]:


import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('E:\EPL_20_21.csv')
xx=np.array(data['Age'])
yy=np.array(data['Mins'])
slope, intercept, r, p, stderr = scipy.stats.linregress(xx, yy)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
fig, ax = plt.subplots()
ax.plot(xx, yy, linewidth=0, marker='s', label='Data points')
ax.plot(xx, intercept + slope * xx, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y'\][-]
ax.legend(facecolor='white')
plt.show()
      


# In[25]:


sns.boxplot(data['Matches'])
plt.axvline(np.median(data['Matches']), color='b', linestyle='--',label = 'Median')
plt.show()


# In[32]:


import scipy.stats as stats
import scipy.stats
from scipy.stats import norm
data=pd.read_csv('E:\EPL_20_21.csv')
sample = data.sample(90)
confidence_level1 =0.97
alphaon2=1-((1-confidence_level1)/2)
norm.ppf(alphaon2)
sigma=np.std(data['Age'])
z_critical=stats.norm.ppf(q=alphaon2)
standard_error=sigma/np.sqrt(len(sample))
X_BAR=np.mean(data['Age'])
CI_LOWER=X_BAR-z_critical*standard_error
CL_UPPER=X_BAR+z_critical*standard_error
print(CI_LOWER)
print(CL_UPPER)


# In[27]:


import scipy.stats
from scipy.stats import t
sample = data.sample(10)
confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_mean = np.mean(sample['Age'])
sample_stdr=np.std(sample['Age'])
t_crit = np.abs(t.ppf((1-confidence_level)/2,degrees_freedom))
print((sample_mean-sample_stdr*t_crit/np.sqrt(len(sample )), sample_mean+sample_stdr*t_crit/np.sqrt(len(sample))))
print(sample_mean)
print(sample_stdr)


# In[28]:


data=pd.read_csv('E:\EPL_20_21.csv')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X = data['Matches']
y = data['Goals']
X = X.values.reshape(-1,1)
y = y.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
lR = LinearRegression()
lR.fit(X_train,y_train)
predictions = lR.predict(X_test)
plt.scatter(X_test, y_test,  color='purple')
plt.xlabel('Matches')
plt.ylabel('Goals')
plt.plot(X_test, predictions, color='green', linewidth=3)
plt.show()
print(lR.predict([[30]]))
print(mean_absolute_error( y_test,predictions ))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




