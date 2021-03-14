#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sklearn.metrics
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


#df1 = pd.read_csv('tweets_12_01_2020.csv')
#df2 = pd.read_csv('tweets_2021-01-18.csv')
#df3 = pd.read_csv('tweets_2021-01-26.csv')


# In[3]:


#df = pd.concat([df1, df2, df3])
#df.head()


# In[4]:


#df.drop_duplicates()


# In[5]:


#df.to_csv('Tweets.csv', index=False)


# In[6]:


pd.set_option('display.max_columns', None)


# In[7]:


def preprocesar(texto):
  #convierte a minúsculas
  texto = texto.lower()

  #elimina stopwords
  stop = re.compile(r'\b(' + r'|'.join(stopwords.words('spanish')) + r')\b\s*')
  texto = stop.sub('', texto) 

  texto = re.sub(r"http\S+", "", texto)
    
  texto = re.sub(r"@\S+", "", texto)
  texto = re.sub(r"#\S+", "", texto)
    
  #quita direcciones html
  borrar = re.compile('<.*?>')
  texto= re.sub(borrar, '', texto)

  #quita puntuaciones y todo lo que no sea letra y números
  #texto = re.sub('[^A-ZÜÖÄãüáéíóúa-z0-9]+', ' ', texto)

  #quita numeros
  texto = re.sub(" \d+", " ", texto)
  
  texto = texto.replace('á', 'a')
  texto = texto.replace('é', 'e')
  texto = texto.replace('í', 'i')
  texto = texto.replace('ó', 'o')
  texto = texto.replace('ú', 'u')
  texto = texto.replace('ü', 'u')
  texto = texto.replace('ã', 'a')
  texto = texto.replace('\n', '')
  texto = texto.replace('qualitas', '')
  texto = texto.replace('chubb', '')
  texto = texto.replace('zurich', '')
  texto = texto.replace('aba', '')
  texto = texto.replace('inbursa', '')
  texto = texto.replace('gnp', '')
  texto = texto.replace('ana', '')
  
  return(texto)


# ## Clustering

# In[8]:


def preprocesar_clu(texto):
  #convierte a minúsculas
  texto = texto.lower()

  #elimina stopwords
  stop = re.compile(r'\b(' + r'|'.join(stopwords.words('spanish')) + r')\b\s*')
  texto = stop.sub('', texto)

  texto = re.sub(r"@\S+", "", texto)
  texto = re.sub(r"#\S+", "", texto)

  texto = texto.replace('á', 'a')
  texto = texto.replace('é', 'e')
  texto = texto.replace('í', 'i')
  texto = texto.replace('ó', 'o')
  texto = texto.replace('ú', 'u')
  texto = texto.replace('ü', 'u')
  texto = texto.replace('ã', 'a')
  
  return(texto)


# In[9]:


df_clu = pd.read_csv('Tweets.csv')
df_clu.head()


# In[10]:


df_clu = df_clu[df_clu['Company'] != 'Insurance']


# In[11]:


df_clu = df_clu[['Text']]


# In[12]:


df_clu['Text'] = df_clu['Text'].apply(preprocesar_clu)


# In[13]:


df_clu.head()


# In[14]:


cv = CountVectorizer()
mdt_frec = cv.fit_transform(df_clu['Text']) 
terminos= cv.get_feature_names()
X = pd.DataFrame(mdt_frec.todense(), 
                              index=df_clu.index, 
                              columns=terminos)
print(X.shape)
X.head()


# In[15]:


ms = MeanShift()
ms.fit(X)
cluster_centers = ms.cluster_centers_


# In[16]:


cluster_centers


# In[17]:


labels = ms.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


# In[18]:


labels_unique


# In[19]:


df_clu['label'] = labels


# In[20]:


df_clu.head()


# In[21]:


df_clu.to_csv('test_cluster.csv', index=False)


# ## Latent Semantic Analysis

# In[22]:


from sklearn.decomposition import TruncatedSVD


# In[23]:


df_lsa = pd.read_csv('Tweets.csv')
df_lsa.head()


# In[24]:


df_lsa = df_lsa[df_lsa['Company'] != 'Insurance']


# In[25]:


df_lsa = df_lsa[['Text']]


# In[26]:


df_lsa['Text'] = df_lsa['Text'].apply(preprocesar_clu)


# In[27]:


cv = CountVectorizer()
mdt_frec = cv.fit_transform(df_lsa['Text']) 
terminos= cv.get_feature_names()
X = pd.DataFrame(mdt_frec.todense(), 
                              index=df_lsa.index, 
                              columns=terminos)
print(X.shape)
X.head()


# In[28]:


#30 topics trae sweet spot, 1000 iteraciones
svd_model = TruncatedSVD(n_components=30, algorithm='randomized', n_iter=1000, random_state=122)


# In[29]:


svd_model.fit(X)


# In[30]:


terms = cv.get_feature_names()
spam_words = []

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    temas = []
    for t in sorted_terms:
        temas.append(t[0])
    print("Topic "+str(i)+": ", temas)
    if i in (0, 1, 3, 4, 5, 6, 7, 10, 11, 16, 21, 22, 24, 30):
        spam_words.append(temas)


# In[31]:


words = []
for x in spam_words:
    for y in x:
        words.append(y)


# In[32]:


len(words)


# ## Naive Bayes

# In[33]:


df = pd.read_csv('Tweets.csv')
df.head()


# In[34]:


df = df[df['Company'] != 'Insurance']


# In[35]:


df = df[['Text', 'Company']]


# In[36]:


df['Text'] = df['Text'].apply(preprocesar)


# In[37]:


df.head()


# In[38]:


spam_score = []
word_len = []

for index, row in df.iterrows():
    contador = 0
    largo = 0
    for x in str(row[0]).split():
        if x in words:
            contador = contador + 1
        largo = largo + 1
    spam_score.append(contador)
    word_len.append(largo)
    
df['spam_score'] = spam_score
df['word_len'] = word_len
df['spam'] = df['spam_score'] / df['word_len']
df = df[df['spam'] < 0.20]


# In[39]:


df = df[['Text', 'Company']]


# In[40]:


cv = CountVectorizer()
mdt_frec = cv.fit_transform(df['Text']) 
terminos= cv.get_feature_names()
X = pd.DataFrame(mdt_frec.todense(), 
                              index=df.index, 
                              columns=terminos)
print(X.shape)
X.head()


# In[41]:


y = df['Company']


# In[42]:


X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=3, shuffle=True)


# In[43]:


y_train.value_counts(normalize=True)


# In[44]:


y_test.value_counts(normalize=True)


# In[45]:


naive_bayes = MultinomialNB()   
naive_bayes.fit(X_train, y_train)


# In[46]:


naive_bayes.classes_


# In[47]:


y_pred = naive_bayes.predict(X_test)
y_pred


# In[48]:


data = {'Y_Real':  y_test,
        'Y_Prediccion':y_pred
        }

df = pd.DataFrame(data, columns=['Y_Real','Y_Prediccion'])
confusion_matrix = pd.crosstab(df['Y_Real'], df['Y_Prediccion'], rownames=['Real'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, fmt='g')
plt.show()


# In[49]:


df.head(50)


# In[50]:


print('Exactitud: ', format(sklearn.metrics.accuracy_score(y_test, y_pred)))

#print('Exactitud average: ', format(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))


#print('\nPrecisión: ', format(sklearn.metrics.precision_score(y_test, y_pred, average='weighted')))  

#print('\nSensibilidad: ', format(sklearn.metrics.recall_score(y_test, y_pred, average='weighted')))

#print('\nF1 score: ', format(sklearn.metrics.f1_score(y_test, y_pred, average='weighted')))


# ## Lexicon Based Sentiment Analysis

# In[51]:


from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import spacy
import nltk
from textblob import TextBlob, Word
from langdetect import detect
import string
from googletrans import Translator
import googletrans
from deep_translator import GoogleTranslator


# In[52]:


def preprocesar2(texto):
  #convierte a minúsculas
  texto = texto.lower()

  #elimina stopwords
  stop = re.compile(r'\b(' + r'|'.join(stopwords.words('spanish')) + r')\b\s*')
  texto = stop.sub('', texto) 

  texto = re.sub(r"http\S+", "", texto)
    
  texto = re.sub(r"@\S+", "", texto)
  texto = re.sub(r"#\S+", "", texto)
    
  #quita direcciones html
  borrar = re.compile('<.*?>')
  texto= re.sub(borrar, '', texto)

  #quita puntuaciones y todo lo que no sea letra y números
  texto = re.sub('[^A-ZÜÖÄãüáéíóúa-z0-9]+', ' ', texto)

  #quita numeros
  texto = re.sub(" \d+", " ", texto)
  
  texto = texto.replace(',', '')
  texto = texto.replace('á', 'a')
  texto = texto.replace('é', 'e')
  texto = texto.replace('í', 'i')
  texto = texto.replace('ó', 'o')
  texto = texto.replace('ú', 'u')
  texto = texto.replace('ü', 'u')
  texto = texto.replace('ã', 'a')
  texto = texto.replace('\n', '')
  
  return(texto)


# In[53]:


df_lex = pd.read_csv('Tweets.csv')
df_lex.head()


# In[54]:


#df_lex = df_lex[df_lex['Company'] != 'Insurance']
df_lex = df_lex[df_lex['Company'] == 'Chubb']


# In[55]:


spam_score = []
word_len = []

for index, row in df_lex.iterrows():
    contador = 0
    largo = 0
    for x in str(row[0]).split():
        if x in words:
            contador = contador + 1
        largo = largo + 1
    spam_score.append(contador)
    word_len.append(largo)
    
df_lex['spam_score'] = spam_score
df_lex['word_len'] = word_len
df_lex['spam'] = df_lex['spam_score'] / df_lex['word_len']
df_lex = df_lex[df_lex['spam'] < 0.20]


# In[56]:


df_lex = df_lex[['Text', 'Company']]


# In[57]:


df_lex.head()


# In[58]:


df_lex['Text'] = df_lex['Text'].apply(preprocesar2)


# In[59]:


#stemmer = SnowballStemmer('spanish')


# In[60]:


#df_lex['Text'] = df_lex['Text'].apply(lambda x: [stemmer.stem(x) for y in x])


# In[61]:


df_lex.head()


# In[62]:


sentimiento = []
empresa = []
subjetividad = []

for index, row in df_lex.iterrows():
    try:
        tweets_trad = GoogleTranslator(source='auto', target='en').translate(row[0])
        fila = str(tweets_trad)
        textFinal = TextBlob(fila)
        sentimiento.append(float(textFinal.sentiment.polarity))
        subjetividad.append(float(textFinal.sentiment.subjectivity))
        empresa.append(row[1])
    except:
        None


# In[63]:


df_sen = pd.DataFrame(list(zip(empresa, sentimiento, subjetividad)), columns=['Empresa', 'Sentimiento', 'Subjetividad'])
df_sen.head()


# In[64]:


len(sentimiento)


# In[65]:


df_lex.shape


# In[66]:


df_lex['Sentimiento'] = sentimiento
df_lex['Subjetividad'] = subjetividad
df_lex.head()


# In[68]:


df_senti = pd.read_csv('Tweets.csv')
df_senti = df_senti[df_senti['Company'] == 'Chubb']
pd.merge(df_senti, df_lex, left_index=True, right_index=True).to_csv('Research_DB.csv', index=False)


# In[ ]:


df_sen.groupby('Empresa').sum()


# ## Transformer for Sentiment Analysis

# In[ ]:


from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer


# In[ ]:


def preprocesar_nlp(texto):
  #convierte a minúsculas
  texto = texto.lower()

  #elimina stopwords
  stop = re.compile(r'\b(' + r'|'.join(stopwords.words('spanish')) + r')\b\s*')
  texto = stop.sub('', texto) 

  texto = re.sub(r"http\S+", "", texto)
    
  texto = re.sub(r"@\S+", "", texto)
  texto = re.sub(r"#\S+", "", texto)
    
  #quita direcciones html
  borrar = re.compile('<.*?>')
  texto= re.sub(borrar, '', texto)

  #quita puntuaciones y todo lo que no sea letra y números
  texto = re.sub('[^A-ZÜÖÄãüáéíóúa-z0-9]+', ' ', texto)

  #quita numeros
  texto = re.sub(" \d+", " ", texto)
  
  texto = texto.replace('á', 'a')
  texto = texto.replace('é', 'e')
  texto = texto.replace('í', 'i')
  texto = texto.replace('ó', 'o')
  texto = texto.replace('ú', 'u')
  texto = texto.replace('ü', 'u')
  texto = texto.replace('ã', 'a')
  texto = texto.replace('\n', '')
  
  return(texto)


# In[ ]:


df_nlp = pd.read_csv('Tweets.csv')
df_nlp.head()


# In[ ]:


df_nlp = df_nlp[df_nlp['Company'] == 'Chubb']
df_nlp.head()


# In[ ]:


df_nlp = df_nlp[['Text']]


# In[ ]:


df_nlp['Text'] = df_nlp['Text'].apply(preprocesar_nlp)


# In[ ]:


df_nlp.head()


# In[ ]:


df_nlp.to_csv('data.csv', index=False)


# In[ ]:


LANG_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = Tokenizer.load(
    LANG_MODEL,
    do_lower_case=False)


# In[ ]:


LABEL_LIST = ["negative", "neutral", "positive"]
processor = TextClassificationProcessor(
    tokenizer=tokenizer,
    max_seq_len=128,
    data_dir="data.csv",
    label_list=LABEL_LIST,
    label_column_name="sentiment",
    metric="acc")


# In[ ]:


data_silo = DataSilo(
    processor=processor,
    batch_size=90)


# ## Regresión Múltiple

# In[ ]:


df_reg = pd.read_csv('Tweets.csv')
df_reg.head()


# In[ ]:


df_reg = df_reg[df_reg['Company'] != 'Insurance']


# In[ ]:


spam_score = []
word_len = []

for index, row in df_reg.iterrows():
    contador = 0
    largo = 0
    for x in str(row[0]).split():
        if x in words:
            contador = contador + 1
        largo = largo + 1
    spam_score.append(contador)
    word_len.append(largo)
    
df_reg['spam_score'] = spam_score
df_reg['word_len'] = word_len
df_reg['spam'] = df_reg['spam_score'] / df_reg['word_len']
df_reg = df_reg[df_reg['spam'] < 0.20]


# In[ ]:


df_reg = df_reg[['Text', 'Company']]


# In[ ]:


df_reg['Company'] = df_reg['Company'].str.upper()


# In[ ]:


df_reg.head()


# In[ ]:


list(df_reg.Company.unique())


# In[ ]:


df_pricing = pd.read_csv('anualizada_Onix_2021_Female.csv')
df_pricing.head()


# In[ ]:


df_pricing['Empresa'] = df_pricing['Empresa'].str.replace('Á', 'A')


# In[ ]:


df_top = df_pricing.head(9)


# In[ ]:


del df_top['Anio_vehiculo']


# In[ ]:


del df_top['Vehiculo']


# In[ ]:


del df_top['Sexo']


# In[ ]:


del df_top['edad']


# In[ ]:


del df_top['Coberturas']


# In[ ]:


del df_top['Sexo_']


# In[ ]:


df_top.head()


# In[ ]:


df_pricing['Valor_Comercial_de_Indemnización_daños'].unique()


# In[ ]:


df_pricing['Deducible_Danio'].unique()


# In[ ]:


df_pricing['Valor_Comercial_de_Indemnización_Robo'].unique()


# In[ ]:


df_pricing['Deducible_Robo'].unique()


# In[ ]:


df_pricing['RC_Daños_a_Terceros_Fallecimiento_Accidental_LUC1'].unique()


# In[ ]:


df_pricing['Gastos_Médicos_a_Ocupantes'].unique()


# In[ ]:


df_pricing['Asistencia_Legal_Jurídica'].unique()


# In[ ]:


df_pricing['Servicios_de_Asistencia_Vial_y_en_Viajes'].unique()


# In[ ]:


df_pricing['Responsabilidad_Civil_Obligatorio_RCO'].unique()


# In[ ]:


df_pricing['Extensión_de_RC_por_daños_a_terceros'].unique()


# In[ ]:


df_pricing['Cobertura_RC_en_USA'].unique()


# In[ ]:


df_pricing['Tiempo_promedio_de_arribo_del_Ajustador1'].unique()


# In[ ]:


df_pricing['Envío_de_reparación_en_agencia_antigüedad_del_auto2'].unique()


# In[ ]:


df_pricing['Servicio_de_Grúa6'].unique()


# In[ ]:


df_pricing['Tiempo_de_pago_de_indemnización_Robo_Total3'].unique()


# In[ ]:


df_pricing['Pago_Valor_Factura'].unique()


# In[ ]:


df_pricing['Responsabilidad_Civil_en_el_Extranjero'].unique()


# In[ ]:


df_pricing['Extensión_de_Responsabilidad_Civil'].unique()


# In[ ]:


df_pricing['Auxilio_Vial'].unique()


# In[ ]:


df_pricing['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'].unique()


# In[ ]:





# In[ ]:


def solo_numeros(texto):
    texto = re.sub('[^0-9]', '', texto)
    
    return (texto)


# In[ ]:


df_top['Tiempo_promedio_de_arribo_del_Ajustador1'] = df_top['Tiempo_promedio_de_arribo_del_Ajustador1'].apply(solo_numeros).str[:2]


# In[ ]:


df_top['Tiempo_de_pago_de_indemnización_Robo_Total3'] = df_top['Tiempo_de_pago_de_indemnización_Robo_Total3'].apply(solo_numeros)


# In[ ]:


df_top['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'] = df_top['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'].apply(solo_numeros)


# In[ ]:


df_top['Servicio_de_Grúa6'] = df_top['Servicio_de_Grúa6'].apply(solo_numeros)


# In[ ]:


df_top['Envío_de_reparación_en_agencia_antigüedad_del_auto2'] = df_top['Envío_de_reparación_en_agencia_antigüedad_del_auto2'].apply(solo_numeros)


# In[ ]:


df_top['Tiempo_de_pago_de_indemnización_Robo_Total3'] = df_top['Tiempo_de_pago_de_indemnización_Robo_Total3'].str.replace('5', '120')


# In[ ]:


df_top['Envío_de_reparación_en_agencia_antigüedad_del_auto2'] = df_top['Envío_de_reparación_en_agencia_antigüedad_del_auto2'].str.replace('', '0')


# In[ ]:


df_top['Servicio_de_Grúa6'] = df_top['Servicio_de_Grúa6'].str.replace('', '0')


# In[ ]:


df_top['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'] = df_top['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'].str.replace('', '0')


# In[ ]:


df_top = df_top[['Empresa', 'Precio', 'Tiempo_promedio_de_arribo_del_Ajustador1', 'Envío_de_reparación_en_agencia_antigüedad_del_auto2', 'Servicio_de_Grúa6', 'Tiempo_de_pago_de_indemnización_Robo_Total3', 'Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo']]


# In[ ]:


df_top


# In[ ]:


df_final = pd.merge(df_reg, df_top, left_on='Company', right_on='Empresa', how='left')
df_final.head()


# In[ ]:


df_final['Text'] = df_final['Text'].apply(preprocesar)


# In[ ]:


df_final = df_final.dropna()


# In[ ]:


df_final


# In[ ]:





# In[ ]:


del df_final['Empresa']
del df_final['Company']


# In[ ]:


df_final.head()


# In[ ]:


df_final.dtypes


# In[ ]:


df_final


# In[ ]:


df_final['Precio'] = df_final['Precio'].str.replace('$', '').str.replace(',', '')


# In[ ]:


df_final['Precio'] = df_final['Precio'].astype(float)
df_final['Tiempo_promedio_de_arribo_del_Ajustador1'] = df_final['Tiempo_promedio_de_arribo_del_Ajustador1'].astype(int)
df_final['Envío_de_reparación_en_agencia_antigüedad_del_auto2'] = df_final['Envío_de_reparación_en_agencia_antigüedad_del_auto2'].astype(int)
df_final['Servicio_de_Grúa6'] = df_final['Servicio_de_Grúa6'].astype(int)
df_final['Tiempo_de_pago_de_indemnización_Robo_Total3'] = df_final['Tiempo_de_pago_de_indemnización_Robo_Total3'].astype(int)
df_final['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'] = df_final['Opción_de_Pago_de_Daños_Materiales_Pago_en_Efectivo'].astype(int)


# In[ ]:





# In[ ]:





# In[ ]:


y = df_final['Precio']


# In[ ]:


del df_final['Precio']


# In[ ]:


cv = CountVectorizer()
mdt_frec = cv.fit_transform(df_final['Text']) 
terminos= cv.get_feature_names()
X = pd.DataFrame(mdt_frec.todense(), 
                             index=df_final.index, 
                             columns=terminos)


print(X.shape)
X.head()


# In[ ]:


###########################Aquí estoy agregando el pricing strategy

#X = pd.merge(df_final, X, left_index=True, right_index=True)


# In[ ]:


#del X['Text']


# In[ ]:


X.head()


# In[ ]:





# In[ ]:


X.dtypes


# In[ ]:





# In[ ]:


X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=3, shuffle=True)


# In[ ]:


model = ElasticNet(alpha=1.0, l1_ratio=0.5)


# In[ ]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[ ]:


scores_mae = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores_r = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)


# In[ ]:


scores_mae = absolute(scores_mae)
scores_r = absolute(scores_r)


# In[ ]:


print('Mean MAE: %.3f (%.3f)' % (mean(scores_mae), std(scores_mae)))
print('Mean r2: %.3f (%.3f)' % (mean(scores_r), std(scores_r)))


# ## Random Forest

# In[ ]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


predictions = rf.predict(X_test)


# In[ ]:


errors = abs(predictions - y_test)


# In[ ]:


print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[ ]:


mape = 100 * (errors / y_test)


# In[ ]:


accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:





# In[ ]:


feature_list = list(X_train.columns)


# In[ ]:


# Import tools needed for visualization
#from sklearn.tree import export_graphviz
#import pydot# Pull out one tree from the forest
#tree = rf.estimators_[5]# Import tools needed for visualization
#from sklearn.tree import export_graphviz
#import pydot# Pull out one tree from the forest
#tree = rf.estimators_[5]# Export the image to a dot file
#export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)# Use dot file to create a graph
#(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('tree.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


importances = list(rf.feature_importances_)


# In[ ]:


feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[ ]:


import matplotlib.pyplot# Set the style
plt.style.use('fivethirtyeight')# list of x locations for plotting
x_values = list(range(len(importances)))# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[ ]:





# In[ ]:




