#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import datetime
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import interact_manual
plt.style.use('fivethirtyeight')


# In[3]:


get_ipython().run_line_magic('time', "data=pd.read_csv('data.csv')")
print(data.shape)


# In[4]:


data.columns


# In[5]:


pd.set_option('max_columns',100)
data.head()


# In[6]:


pd.set_option('max_columns',100)
data.iloc[:, 2:].describe().style.background_gradient(cmap='cividis')


# In[7]:


data.iloc[:,13:].describe(include='object')


# In[8]:


mno.bar(data.iloc[:, :40],color='orange',sort='ascending')
plt.title('Checking Missing Values Heat Map for first half of the data',fontsize=15)
plt.show()


# In[9]:


mno.bar(data.iloc[:,40:])
plt.title('Checking Missing Values Heat Map for the second half of the data')
plt.show()


# In[10]:


data['ShortPassing'].fillna(data['ShortPassing'].mean(),inplace=True)
data['Volleys'].fillna(data['Volleys'].mean(),inplace=True)
data['Dribbling'].fillna(data['Dribbling'].mean(),inplace=True)
data['Curve'].fillna(data['Curve'].mean(),inplace=True)
data['FKAccuracy'].fillna(data['FKAccuracy'].mean(),inplace=True)
data['LongPassing'].fillna(data['LongPassing'].mean(),inplace=True)
data['BallControl'].fillna(data['BallControl'].mean(),inplace=True)
data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(),inplace=True)
data['Finishing'].fillna(data['Finishing'].mean(),inplace=True)
data['Crossing'].fillna(data['Crossing'].mean(),inplace=True)
data['Weight'].fillna('200lbs',inplace=True)
data['Contract Valid Until'].fillna(2019,inplace=True)
data['Height'].fillna("5'11",inplace=True)
data['Loaned From'].fillna('None',inplace=True)
data['Joined'].fillna('Jul 1,2018',inplace=True)
data['Jersey Number'].fillna(8,inplace=True)
data['Body Type'].fillna('Normal',inplace=True)
data['Position'].fillna('ST',inplace=True)
data['Club'].fillna('No Club',inplace=True)
data['Work Rate'].fillna("Medium/Medium",inplace=True)
data['Skill Moves'].fillna('Skill Moves',inplace=True)
data['Weak Foot'].fillna(3,inplace=True)
data['Preferred Foot'].fillna('Right',inplace=True)
data['International Reputation'].fillna(1,inplace=True)
data['Wage'].fillna('€200k',inplace=True)
data.fillna(0,inplace=True)

data.isnull().sum().sum()


# In[11]:


def defending(data):
    return int(round((data[['Marking', 'StandingTackle', 'SlidingTackle']].mean()).mean()))
def general(data):
    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 'BallControl']].mean()).mean()))
def mental(data):
    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 'Vision', 'Composure']].mean()).mean()))
def passing(data):
    return int(round((data[['Crossing','ShortPassing','LongPassing']].mean()).mean()))
def mobility(data):
    return int(round((data[['Acceleration', 'SprintSpeed','Agility','Reactions']].mean()).mean()))
def power(data):
    return int(round((data[['Balance','Jumping','Stamina','Strength']].mean()).mean()))
def rating(data):
    return int(round((data[['Potential','Overall']].mean()).mean()))
def shooting(data):
    return int(round((data[['Finishing','Volleys','FKAccuracy','ShotPower','LongShots','Penalties']].mean()).mean()))


# In[12]:


data['Defending'] = data.apply(defending,axis=1)
data['General'] = data.apply(general,axis=1)
data['Mental'] = data.apply(mental,axis=1)
data['Passing'] = data.apply(passing,axis=1)
data['Mobility'] = data.apply(mobility,axis=1)
data['Power'] = data.apply(power,axis=1)
data['Rating'] = data.apply(rating,axis=1)
data['Shooting'] = data.apply(shooting,axis=1)

data.columns


# In[13]:


plt.rcParams['figure.figsize']=(18,8)
plt.subplot(2,4,1)
sns.distplot(data['Defending'],color='red')
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['General'],color='black')
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['Mental'],color='red')
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['Passing'],color='black')
plt.grid()

plt.subplot(2,4,5)
sns.distplot(data['Mobility'],color='red')
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['Power'],color='black')
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['Shooting'],color='red')
plt.grid()

plt.subplot(2,4,8)
sns.distplot(data['Rating'],color='black')
plt.grid()

plt.suptitle('Source Distributions for Different Abilities')
plt.show()


# In[14]:


plt.rcParams['figure.figsize']=(10,5)
sns.countplot(data['Preferred Foot'],palette='pink')
plt.title('Most Preferred Foot of the Players',fontsize=20)
plt.show()


# In[15]:


labels=['1','2','3','4','5']
sizes=data['International Reputation'].value_counts()
colors=plt.cm.copper(np.linspace(0,1,5))
explode=[0.1,0.1,0.2,0.5,0.9]

plt.rcParams['figure.figsize']=(9,9)
plt.pie(sizes,labels=labels,colors=colors,explode=explode,shadow=True)
plt.title('International Reputation for the Football Players',fontsize=20)
plt.legend()
plt.show()


# In[17]:


data[data['International Reputation']==5][['Name','Nationality','Overall']].sort_values(by='Overall',ascending=False).style.background_gradient(cmap='magma')


# In[18]:


labels=['5','4','3','2','1']
size=data['Weak Foot'].value_counts()
colors=plt.cm.Wistia(np.linspace(0,1,5))
explode=[0,0,0,0,0.1]

plt.pie(size,labels=labels,colors=colors,explode=explode,shadow=True,startangle=90)
plt.title('Distribution of Week Foot among Players',fontsize=25)
plt.legend()
plt.show()


# In[19]:


plt.figure(figsize=(13,15))
plt.style.use('fivethirtyeight')
ax=sns.countplot(y='Position',data=data,palette='bone')
ax.set_xlabel(xlabel='Different Positions in Football',fontsize=16)
ax.set_ylabel(ylabel='Count of players',fontsize=16)
ax.set_title(label='Comparision of Positions and Players',fontsize=20)
plt.show()


# In[20]:


def extract_value_from(value):
    out=value.replace('lbs','')
    return float(out)

data['Weight']=data['Weight'].apply(lambda x:extract_value_from(x))

sns.distplot(data['Weight'],color='black')
plt.title("Distribution of Players Weight",fontsize=15)
plt.show()


# In[21]:


def extract_value_from(column):
    out=column.replace('€','')
    if 'M' in out:
        out=float(out.replace('M', ''))*1000000
    elif 'K' in column:
        out=float(out.replace('K', ''))*1000
    return float(out)


# In[22]:


data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))
data['Wage']=data['Wage'].apply(lambda x: extract_value_from(x))

plt.rcParams['figure.figsize']=(16,5)
plt.subplot(1,2,1)
sns.distplot(data['Value'],color='violet')
plt.title('Distribution of Value of the Players',fontsize=15)

plt.subplot(1,2,2)
sns.distplot(data['Wage'],color='purple')
plt.title('Distribution of Wages of the Players',fontsize=15)
plt.show()


# In[23]:


plt.figure(figsize=(10,6))
ax=sns.countplot(x='Skill Moves',data=data,palette='pastel')
ax.set_title(label='Count of players on basis of their skill moves',fontsize=20)
ax.set_xlabel(xlabel='Number of Skill Moves',fontsize=20)
ax.set_ylabel(ylabel='Count',fontsize=16)
plt.show()


# In[24]:


plt.figure(figsize=(15,5))
plt.style.use('fivethirtyeight')

sns.countplot(x='Work Rate',data = data,palette='hls')
plt.title('Different work rates of the Players Participating in the FIFA 2019',fontsize=20)
plt.xlabel('Work rates associated with the players',fontsize=16)
plt.ylabel('count of Players',fontsize=16)
plt.xticks(rotation=90)
plt.show()


# In[25]:


plt.figure(figsize=(16,4))
plt.style.use('seaborn-paper')

plt.subplot(1,2,1)
x=data.Potential
ax=sns.distplot(x, bins=58,kde=False,color='y')
ax.set_xlabel(xlabel="Player's Potential Scores",fontsize=10)
ax.set_ylabel(ylabel='Number of players',fontsize=10)
ax.set_title(label='Histogram of players Potential Scores',fontsize=15)

plt.subplot(1,2,2)
y=data.Overall
ax=sns.distplot(y,bins=58,kde=False,color='y')
ax.set_xlabel(xlabel="Player's Overall Scores",fontsize=10)
ax.set_ylabel(ylabel='Number of players',fontsize=10)
ax.set_title(label='Histogram of players Overall Scores',fontsize=15)
plt.show()


# In[26]:


plt.rcParams['figure.figsize']=(20,7)
plt.style.use('seaborn-dark-palette')

sns.boxplot(data['Overall'],data['Age'],hue=data['Preferred Foot'],palette='Greys')
plt.title('Comparision of Overall Scores and age wrt Preferred foot',fontsize=20)
plt.show()


# In[27]:


data['Nationality'].value_counts().head(10).plot(kind='pie',cmap='inferno',startangle=90,explode=[0,0,0,0,0,0,0,0,0.1,0])
plt.title('Countries having Highest Number of players',fontsize=15)
plt.axis('off')
plt.show()


# In[28]:


some_countries=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia')
data_countries=data.loc[data['Nationality'].isin(some_countries)&data['Weight']]

plt.rcParams['figure.figsize']=(15,7)
ax=sns.violinplot(x=data_countries['Nationality'],y=data_countries['Weight'],palette='Reds')
ax.set_xlabel(xlabel='Countries',fontsize=9)
ax.set_ylabel(ylabel='Weight in lbs',fontsize=9)
ax.set_title(label='Distribution of Weight of players from different countries',fontsize=20)
plt.show()


# In[29]:


some_countries=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia')
data_countries=data.loc[data['Nationality'].isin(some_countries)&data['Overall']]

plt.rcParams['figure.figsize']=(15,7)
ax=sns.barplot(x=data_countries['Nationality'],y=data_countries['Overall'],palette='spring')
ax.set_xlabel(xlabel='Countries',fontsize=9)
ax.set_ylabel(ylabel='Overall Scores',fontsize=9)
ax.set_title(label='Distribution of overall scores of players from different countries',fontsize=20)
plt.show()


# In[30]:


some_countries=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia')
data_countries=data.loc[data['Nationality'].isin(some_countries)&data['Wage']]

plt.rcParams['figure.figsize']=(15,7)
ax=sns.barplot(x=data_countries['Nationality'],y=data_countries['Wage'],palette='Purples')
ax.set_xlabel(xlabel='Countries',fontsize=9)
ax.set_ylabel(ylabel='Wage',fontsize=9)
ax.set_title(label='Distribution of Wages of players from different countries',fontsize=20)
plt.show()


# In[31]:


some_countries=('England','Germany','Spain','Argentina','France','Brazil','Italy','Columbia')
data_countries=data.loc[data['Nationality'].isin(some_countries)&data['International Reputation']]

plt.rcParams['figure.figsize']=(15,7)
ax=sns.boxenplot(x=data_countries['Nationality'],y=data_countries['International Reputation'],palette='autumn')
ax.set_xlabel(xlabel='Countries',fontsize=9)
ax.set_ylabel(ylabel='Distribution of reputation',fontsize=9)
ax.set_title(label='Distribution of International Repuatation of players from different countries',fontsize=20)
plt.show()


# In[32]:


some_clubs=('CD Leganes','Southampton','RC Celta','Empoli','Fortuna Dusseldorf','Manchestar City','Tottenham Hotspur','FC Barcelona','Valencia CF','Real Madrid')
data_clubs=data.loc[data['Club'].isin(some_clubs)&data['Overall']]

plt.rcParams['figure.figsize']=(15,8)
ax=sns.boxplot(x=data_clubs['Club'], y=data_clubs['Overall'],palette='inferno')
ax.set_xlabel(xlabel='Some Popular Clubs',fontsize=9)
ax.set_ylabel(ylabel='Overall Score',fontsize=9)
ax.set_title(label='Distribution of Overall Score in Different popular Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[33]:


some_clubs=('CD Leganes','Southampton','RC Celta','Empoli','Fortuna Dusseldorf','Manchestar City','Tottenham Hotspur','FC Barcelona','Valencia CF','Real Madrid')
data_club=data.loc[data['Club'].isin(some_clubs)&data['Wage']]

plt.rcParams['figure.figsize']=(15,8)
ax=sns.boxenplot(x='Club', y='Age', data = data_club, palette='magma')
ax.set_xlabel(xlabel='Names of some popular Clubs',fontsize=10)
ax.set_ylabel(ylabel='Distribution',fontsize=10)
ax.set_title(label='Distribution of Ages in some popular Clubs',fontsize=20)
plt.grid()
plt.show()


# In[49]:


some_clubs=('CD Leganes','Southampton','RC Celta','Empoli','Fortuna Dusseldorf','Manchestar City','Tottenham Hotspur','FC Barcelona','Valencia CF','Real Madrid')
data_club=data.loc[data['Club'].isin(some_clubs)&data['Wage']]

plt.rcParams['figure.figsize']=(15,8)
ax=sns.boxplot(x='Club', y='Wage', data = data_club, palette='magma')
ax.set_xlabel(xlabel='Names of some popular Clubs',fontsize=10)
ax.set_ylabel(ylabel='Distribution',fontsize=10)
ax.set_title(label='Distribution of Wages in some popular Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.show()


# In[50]:


some_clubs=('CD Leganes','Southampton','RC Celta','Empoli','Fortuna Dusseldorf','Manchestar City','Tottenham Hotspur','FC Barcelona','Valencia CF','Real Madrid')
data_club=data.loc[data['Club'].isin(some_clubs)&data['International Reputation']]

plt.rcParams['figure.figsize']=(16,8)
ax=sns.boxenplot(x='Club', y='International Reputation', data = data_club, palette='copper')
ax.set_xlabel(xlabel='Names of some popular Clubs',fontsize=10)
ax.set_ylabel(ylabel='Distribution od Reputation',fontsize=10)
ax.set_title(label='Distribution of International Reputation in some popular Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[48]:


some_clubs=('CD Leganes','Southampton','RC Celta','Empoli','Fortuna Dusseldorf','Manchestar City','Tottenham Hotspur','FC Barcelona','Valencia CF','Chelsea','Real Madrid')
data_club=data.loc[data['Club'].isin(some_clubs)&data['Wage']]

plt.rcParams['figure.figsize']=(15,8)
ax=sns.boxenplot(x='Club', y='Age', data = data_club, palette='magma')
ax.set_xlabel(xlabel='Names of some popular Clubs',fontsize=10)
ax.set_ylabel(ylabel='Distribution',fontsize=10)
ax.set_title(label='Distribution of Ages in some popular Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[36]:


sns.lmplot(x='BallControl',y='Dribbling',data=data,col='Preferred Foot')
plt.show()


# In[37]:


data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Position','Name','Age','Club','Nationality','Overall']].sort_values(by='Overall', ascending=False).style.background_gradient(cmap='pink')


# In[38]:


@interact
def skill(skills=['Defending','General','Mental','Passing','Mobility','Power','Rating','Shooting'],score=75):
    return data[data[skills]>score][['Name','Nationality','Club','Overall',skills]].sort_values(by=skills,ascending=False).head(20).style.background_gradient(cmap='Blues')


# In[39]:


@interact
def country(country=list(data['Nationality'].value_counts().index)):
    return data[data['Nationality']==country][['Name','Position','Overall','Potential']].sort_values(by='Overall',ascending=False).head(15).style.background_gradient(cmap='magma')


# In[40]:


@interact
def club(club=list(data['Club'].value_counts().index[1:])):
    return data[data['Club']==club][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage','Value','Contract Valid Until']].sort_values(by='Overall',ascending=False).head(15).style.background_gradient(cmap='inferno')


# In[41]:


youngest=data[data['Age']==16][['Name','Age','Club','Nationality','Overall']]
youngest.sort_values(by='Overall',ascending=False).head().style.background_gradient(cmap='magma')


# In[42]:


data.sort_values('Age',ascending=False)[['Name','Age','Club','Nationality','Overall']].head(15).style.background_gradient(cmap='Wistia')


# In[43]:


data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club',
         'Nationality', 'Overall']].sort_values(by = 'Overall',
            ascending = False).head(10).style.background_gradient(cmap = 'bone')


# In[44]:


player_features = ('Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 'Composure', 'Crossing', 'Dribbling', 'FKAccuracy', 'Finishing', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'HeadingAccuracy', 'Interceptions', 'Jumping', 'LongPassing', 'LongShots', 'Marking', 'Penalties')
for i, val in data.groupby(data['Position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))


# In[45]:


data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club','Nationality', 'Overall']].sort_values(by = 'Overall',ascending = False).head(10).style.background_gradient(cmap = 'copper')


# In[52]:


sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')
plt.show()


# In[ ]:




