#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import pandas as pd
import time
from datetime import date


# In[ ]:


#parametros = {'place.fields': ',Mexico,,,,,,'}
headers = {"Authorization": "Bearer Aqui_Pones_Llave"}


# In[ ]:


r = requests.get('https://api.twitter.com/2/tweets/search/recent?query=%23seguros&max_results=10', headers=headers)


# In[ ]:


print(r.status_code)


# In[ ]:


print(r.json())


# In[ ]:


lista = r.json()


# In[ ]:


#Dentro del Json que contesta, existe el diccionario de Meta que da datos generales de la consulta
#Validar el campo de Id, https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user

search_params = []
Tweet_Id = []
Text = []
Company = []

for tweet in r.json()['data']:
    Tweet_Id.append(tweet['id'])
    Text.append(tweet['text'])


# In[ ]:


df = pd.DataFrame(list(zip(Tweet_Id, Text)), columns=['Id', 'Text'])
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## End to End PipeLine

# In[ ]:


search_params = {'%23seguros': 'Insurance', 'Chubb_Seguros':'Chubb', 'ABASEGUROS':'Chubb', 'ABA_SEGUROS':'Chubb', 'ANASeguros':'Ana',                  'AXXA Seguros':'Axxa', 'GNPSeguros':'GNP', 'Qualitas':'Qualitas', 'Mapfre':'Mapfre', 'Seguros_Atlas':'Atlas',                  'Zurich': 'Zurich', 'Inbursa': 'Inbursa', 'HDISeguros': 'HDI', 'AIG': 'AIG'}


# In[2]:


headers = {"Authorization": "Bearer "}
search_params = {'%23seguros': 'Insurance', 'Asegurado':'Insurance', 'Chubb_Seguros':'Chubb', 'ABASEGUROS':'Chubb', 'ABA_SEGUROS':'Chubb', 'ANASeguros':'Ana',                  'AXXA Seguros':'Axxa', 'GNPSeguros':'GNP', 'Qualitas':'Qualitas', 'Mapfre':'Mapfre', 'Seguros_Atlas':'Atlas',                  'Zurich': 'Zurich', 'Inbursa': 'Inbursa', 'HDISeguros': 'HDI', 'AIG': 'AIG'}
Tweet_Id = []
Text = []
Company = []


for key, value in search_params.items():
    r = requests.get('https://api.twitter.com/2/tweets/search/recent?query={}&max_results=100'.format(key), headers=headers)
    print(r)
    print(key)
    try:
        for tweet in r.json()['data']:
            Tweet_Id.append(tweet['id'])
            Text.append(tweet['text'])
            Company.append(value)
        time.sleep(900)
    except:
        print('No hay tweets para {}'.format(value))


# In[3]:


df = pd.DataFrame(list(zip(Tweet_Id, Text, Company)), columns =['Id', 'Text', 'Company'])
df


# In[4]:


df.to_csv('tweets_{}.csv'.format(date.today()), index=False)


# In[ ]:




