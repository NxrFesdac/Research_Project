#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import pyodbc
import time
import zipfile
import os
from os import walk
import datetime


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


#Parametros iniciales

sexo = 0 #Hombre 0, Mujer 1
edad = []
nombre = 'Carlos'
#cp = '03100'
cp = ['01000', 
'56600', 
'44100', 
'64000', 
'91700', 
'37000', 
'72000', 
'76000', 
'31000', 
'25000']
email = ''
telefono = 


#Scrapper
Coberturas_sucio = []
Coberturas_limpio = []
Empresa = []
Precio = []
Vehiculo = []
Anio_Vehiculo = []


# ## New test

# In[ ]:


#Parametros iniciales

sexo = 0 #Hombre 0, Mujer 1
edad = []
nombre = 'Carlos'
#cp = '03100'
cp = ['01000', 
'56600', 
'44100', 
'64000', 
'91700', 
'37000', 
'72000', 
'76000', 
'31000', 
'25000']

email = ''
telefono = 
coches = {
    'JETTA MK VI TRENDLINE TIPTRONIC 2.5L 5CI' : '2016', 
    'AVEO LS STD 1.6L 4CIL 4PTAS' : '2016', 
    'SENTRA ADVANCE AUT 1.8L 4CIL' : '2017', 
    'VERSA SENSE STD 1.6L 4CIL' : '2017', 
    'CR-V EX AUT 2.4L 4CIL' : '2014', 
    'TIIDA SEDAN CONFORT STD 1.8L A/AC' : '2011', 
    'FIESTA SE STD 1.6L 4CIL 4PTAS' : '2014', 
    'VERSA SENSE STD 1.6L 4CIL' : '2017', 
    'JETTA MK VI SPORTLINE TIPTRONIC 2.5L 5CIL' : '2016', 
    'TSURU GSII STD 1.6L 4CIL A/A' : '2013', 
    'CHEVY 4 PTAS. B STD DH' : '2007', 
    'MAZDA 3 SEDAN I TOURING AUT 2.0L 4CIL 4PTAS' : '2015', 
    'X-TRAIL ADVANCE 2 ROW CVT 2.5L 4CIL' : '2016', 
    'ESCAPE S AUT 2.5L 4CIL' : '2014', 
    'CIVIC EX STD L4 2.0 4PTAS' : '2017', 
    'GOL CL STD 1.6L 4CIL A/A 5PTAS' : '2016', 
    'POLO STD 1.6L 4CIL' : '2018', 
    'SPARK CLASSIC LT  B STD 1.2L 4CIL' : '2017', 
    'FOCUS SE AUT 2.0L 4CIL 4PTAS' : '2013', 
    'MARCH ADVANCE AUT 1.6L 4CIL' : '2018', 
    'MARCH SENSE AUT 1.6L 4CIL' : '2018', 
    'JETTA MK VI COMFORTLINE STD 2.5L 5CIL' : '2017', 
    'ACCORD EXL NAVI SEDAN CVT 2.4L 4CIL 4PTAS' : '2016', 
    'SONIC LT D STD 1.6L 4CIL 4PTAS' : '2016', 
    'VENTO STYLE STD 1.6L 4CIL 4PTAS' : '2014', 
    'TRAX LT B AUT 1.8L 4CIL' : '2015', 
    'RAV4 XLE AUT AWD 2.5L 4CIL' : '2015', 
    'COROLLA LE AUT 1.8L 4CIL' : '2014', 
    'BEAT SEDAN LT 1.2L 4CIL' : '2018', 
    'VENTO ACTIVE STD 1.6L 4CIL 4PTAS' : '2014', 
    'VENTO HIGHLINE STD 1.6L 4CIL 4PTAS' : '2014', 
    'ECOSPORT SE TREND STD 2.0L 4CIL' : '2014', 
    'AVANZA PREMIUM AUT 1.5L 4CIL' : '2016' }

#Scrapper
Coberturas_sucio = []
Coberturas_limpio = []
Empresa = []
Precio = []
Vehiculo = []
Anio_Vehiculo = []


# In[ ]:


Coberturas_limpio = []
Empresa = []
Precio = []
vehiculo_cot = []
anio_cot = []
edad = []
sexo_cot = []

for x in range(len(cp)):
    for key, value in coches.items():
        Vehiculo = []
        Anio_Vehiculo = []
        #print(x)
        #print('-----------------------------------------------------------------------------------------------------------------')
        driver.get('https://www.autocompara.com/')
        time.sleep(10)
        
        try:
            intro = driver.find_element_by_xpath("//a[@class='vtw-close cybXButton']")
            intro.click()
        except:
            print("No clic")

        anio = driver.find_element_by_xpath("//div[@class='select-button-home year select-custom']")
        anio.click()
        anio_drop = driver.find_elements_by_xpath("//div[@class='ng-dropdown-panel-items scroll-host']")
        for anio in anio_drop:
            years =  anio.find_elements_by_xpath(".//div[@class='ng-option ng-option-marked']") +  anio.find_elements_by_xpath(".//div[@class='ng-option']")
            for year in years:
                year_boton = year.find_element_by_xpath("./span[@class='ng-option-label']")
                #print(year)
                #print(year.text)
                if year.text == value:
                    Anio_Vehiculo.append(year.text)
                    year_boton.click()
                    #print(year_boton)
                    break
        
        
        #try:
        #    intro = driver.find_element_by_xpath("//a[@class='vtw-close cybXButton']")
        #    intro.click()
        #except:
        #    print("No clic")
        
    
        time.sleep(10)
        modelos1 = driver.find_element_by_xpath("//div[@class='ng-select-container']")
        modelos1.click()
        time.sleep(1)
        modelo2 = driver.find_element_by_xpath("//div[@class='ng-select-container']//div[@class='ng-input']/input[@role='combobox']")
        modelo2.send_keys(key)
        time.sleep(1)
        modelos_drop = driver.find_elements_by_xpath("//div[@class='ng-dropdown-panel-items scroll-host']")
        for modelo in modelos_drop:
            models = modelo.find_elements_by_xpath(".//div[@class='ng-option ng-option-child ng-option-marked']") + modelo.find_elements_by_xpath(".//div[@class='ng-option ng-option-child']")
            for model in models:
                #print(model)
                print(model.text)
                Vehiculo.append(model.text)
                model.click()
                break
    
        #try:
        #    intro = driver.find_element_by_xpath("//a[@class='vtw-close cybXButton']")
        #    intro.click()
        #except:
        #    print("No clic")
        
    
    
        time.sleep(5)
        
        #try:
        #    intro = driver.find_element_by_xpath("//a[@class='vtw-close cybXButton']")
        #    intro.click()
        #except:
        #    print("No clic")
        
        
        cotiza_boton = driver.find_element_by_xpath("//div[@class='button-cotiza']")
        cotiza_boton.click()
        time.sleep(0.5)
        try:
            cotiza_boton.click()
        except:
            print('Un solo boton cotiza')
    
    
        time.sleep(2)
        if sexo == 1:
            sexo_boton = driver.find_element_by_xpath("//label[@class='btn icon-mujer']").click()
    
    
        driver.find_element_by_xpath("//input[@id='date']").click()
        #Año
        time.sleep(2)
        driver.find_element_by_css_selector("[title^='Select year']").click()
        select_year = Select(driver.find_element_by_css_selector("[title^='Select year']"))
        #Cambiar el año para que loopee
                                #select_year.select_by_value(str(2004 - x))
        select_year.select_by_value(str(1985))
        #Cambiar el año para que loopee
        #driver.find_element_by_css_selector("[title^='Select year']").click()
    
    
        #Selección de día
        time.sleep(2)
        dias = driver.find_elements_by_xpath("//div[@class='btn-light']")
        for dia in dias:
            if dia.text == '1':
                dia.click()
                break
            
    
        #Selección de Nombre
        time.sleep(2)
        name = driver.find_element_by_xpath("//input[@id='name']")
        name.clear
        name.send_keys(nombre)
    
    
        #Selección de CP
        time.sleep(2)
        codigo_postal = driver.find_element_by_xpath("//input[@id='txtPostalCode']")
        codigo_postal.clear
        codigo_postal.send_keys(cp[x])
    
    
        #Selección Correo
        time.sleep(2)
        correo = driver.find_element_by_xpath("//input[@id='email']")
        correo.clear
        correo.send_keys(email)
    
    
        #Selección Teléfono
        time.sleep(2)
        phone = driver.find_element_by_xpath("//input[@id='phone']")
        phone.clear
        phone.send_keys(telefono)
    
    
        time.sleep(2)
        driver.find_element_by_xpath("//button[@class='btn btn-red']").click()
    
    
        time.sleep(35)
        detalles = driver.find_elements_by_xpath("//button[@class='btn btn-lnk details']")
    
    
        #DB = {'Auto': ['Empresa':[], 'Coberturas':[]]}

        for detalle in detalles:
            Coberturas_sucio = []
            detalle.click()
            cobertura_actual = detalle.find_element_by_xpath("//div[@class='col-12']")
            coberturas = cobertura_actual.find_elements_by_xpath("//div[@class='row']")
            for cobertura in coberturas:
                Coberturas_sucio.append(cobertura.text)
    
            Coberturas_limpio.append(Coberturas_sucio[3])
            Empresa.append(cobertura_actual.find_element_by_xpath("//span[@class='ng-value-label']").text)
            Precio.append(detalle.find_element_by_xpath("//div[@class='price mb-2']/p[@class='price']").text)
        
            vehiculo_cot.append(Vehiculo)
            anio_cot.append(Anio_Vehiculo)
            sexo_cot.append(sexo)
            edad.append(36)
    
            cobertura_actual.find_element_by_xpath("//button[@id='btn-close-modal']").click()
            time.sleep(5)
    
        #break
        time.sleep(5)
    print(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Acaba el Test

# In[4]:


caps = DesiredCapabilities.FIREFOX

fp = webdriver.FirefoxProfile()
fp.set_preference("browser.preferences.instantApply",True)
fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain, text/csv, text/x-csv, text/comma-separated-values, text/xml, text/x-comma-separated-values, text/tab-separated-values, application/octet-stream, application/xml, application/zip, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel.sheet.binary.macroenabled.12, application/vnd.ms-excel.addin.macroenabled.12, application/vnd.ms-excel.template.macroenabled.12, application/vnd.ms-excel.sheet.macroenabled.12, application/x-download")
fp.set_preference("browser.helperApps.alwaysAsk.force",False)
fp.set_preference("browser.download.manager.showWhenStarting",False)
# 0 means to download to the desktop, 1 means to download to the default "Downloads" directory, 2 means to use the directory
fp.set_preference("browser.download.folderList", 2)
fp.set_preference("browser.download.manager.showAlertOnComplete", False)
fp.set_preference("dom.popup_maximum", 50)
fp.set_preference("pdfjs.disabled", True)


# In[5]:


driver = webdriver.Firefox(firefox_profile=fp, capabilities=caps)


# In[ ]:


driver.get('https://www.autocompara.com/')
time.sleep(3)


# In[ ]:


try:
    intro = driver.find_element_by_xpath("//a[@class='vtw-close cybXButton']")
    intro.click()
except:
    print("No clic")


# In[ ]:


anio = driver.find_element_by_xpath("//div[@class='select-button-home year select-custom']")
anio.click()


# In[ ]:


anio_drop = driver.find_elements_by_xpath("//div[@class='ng-dropdown-panel-items scroll-host']")


# In[ ]:


#Aquí se cambia el año a cotizar, si se remueve el break, se pueden ver todos los años

for anio in anio_drop:
    years =  anio.find_elements_by_xpath(".//div[@class='ng-option ng-option-marked']") +  anio.find_elements_by_xpath(".//div[@class='ng-option']")
    for year in years:
        year_boton = year.find_element_by_xpath("./span[@class='ng-option-label']")
        print(year)
        print(year.text)
        Anio_Vehiculo.append(year.text)
        year_boton.click()
        print(year_boton)
        break


# In[ ]:


time.sleep(3)
driver.find_element_by_xpath("//div[@class='ng-select-container']").click()


# In[ ]:


modelos_drop = driver.find_elements_by_xpath("//div[@class='ng-dropdown-panel-items scroll-host']")


# In[ ]:


for modelo in modelos_drop:
    models = modelo.find_elements_by_xpath(".//div[@class='ng-option ng-option-child ng-option-marked']") + modelo.find_elements_by_xpath(".//div[@class='ng-option ng-option-child']")
    for model in models:
        print(model)
        print(model.text)
        Vehiculo.append(model.text)
        model.click()
        break


# In[ ]:


time.sleep(5)
cotiza_boton = driver.find_element_by_xpath("//div[@class='button-cotiza']")
cotiza_boton.click()
time.sleep(0.5)
cotiza_boton.click()


# ## Página de Datos de Conductor

# In[ ]:


#Solo si se requiere mujer
time.sleep(2)
if sexo == 1:
    sexo_boton = driver.find_element_by_xpath("//label[@class='btn icon-mujer']").click()


# In[ ]:


#Clic en calendario
driver.find_element_by_xpath("//input[@id='date']").click()


# In[ ]:


#Año
time.sleep(2)
driver.find_element_by_css_selector("[title^='Select year']").click()
select_year = Select(driver.find_element_by_css_selector("[title^='Select year']"))
#Cambiar el año para que loopee
select_year.select_by_value('2000')
#Cambiar el año para que loopee
#driver.find_element_by_css_selector("[title^='Select year']").click()


# In[ ]:


#Selección de día
time.sleep(2)
dias = driver.find_elements_by_xpath("//div[@class='btn-light']")
for dia in dias:
    if dia.text == '1':
        dia.click()
        break


# In[ ]:


#Selección de Nombre
time.sleep(2)
name = driver.find_element_by_xpath("//input[@id='name']")
name.clear
name.send_keys(nombre)


# In[ ]:


#Selección de CP
time.sleep(2)
codigo_postal = driver.find_element_by_xpath("//input[@id='txtPostalCode']")
codigo_postal.clear
codigo_postal.send_keys(cp)


# In[ ]:


#Selección Correo
time.sleep(2)
correo = driver.find_element_by_xpath("//input[@id='email']")
correo.clear
correo.send_keys(email)


# In[ ]:


#Selección Teléfono
time.sleep(2)
phone = driver.find_element_by_xpath("//input[@id='phone']")
phone.clear
phone.send_keys(telefono)


# In[ ]:


time.sleep(2)
driver.find_element_by_xpath("//button[@class='btn btn-red']").click()


# ## Cotizaciones

# In[ ]:


time.sleep(10)
detalles = driver.find_elements_by_xpath("//button[@class='btn btn-lnk details']")


# In[ ]:


#DB = {'Auto': ['Empresa':[], 'Coberturas':[]]}
Coberturas_sucio = []
Coberturas_limpio = []
Empresa = []
Precio = []

for detalle in detalles:
    detalle.click()
    cobertura_actual = detalle.find_element_by_xpath("//div[@class='col-12']")
    coberturas = cobertura_actual.find_elements_by_xpath("//div[@class='row']")
    for cobertura in coberturas:
        Coberturas_sucio.append(cobertura.text)
    
    Coberturas_limpio.append(Coberturas_sucio[3])
    Empresa.append(cobertura_actual.find_element_by_xpath("//span[@class='ng-value-label']").text)
    Precio.append(detalle.find_element_by_xpath("//div[@class='price mb-2']/p[@class='price']").text)
    
    
    cobertura_actual.find_element_by_xpath("//button[@id='btn-close-modal']").click()
    
    #break


# In[ ]:





# In[ ]:


print(detalles)


# In[ ]:


#DB = {'Auto': ['Empresa':[], 'Coberturas':[]]}
#Coberturas_sucio = []
Coberturas_limpio = []
Empresa = []
Precio = []

for detalle in detalles:
    Coberturas_sucio = []
    detalle.click()
    cobertura_actual = detalle.find_element_by_xpath("//div[@class='col-12']")
    coberturas = cobertura_actual.find_elements_by_xpath("//div[@class='row']")
    for cobertura in coberturas:
        Coberturas_sucio.append(cobertura.text)
    
    Coberturas_limpio.append(Coberturas_sucio[3])
    Empresa.append(cobertura_actual.find_element_by_xpath("//span[@class='ng-value-label']").text)
    Precio.append(detalle.find_element_by_xpath("//div[@class='price mb-2']/p[@class='price']").text)
    
    
    cobertura_actual.find_element_by_xpath("//button[@id='btn-close-modal']").click()
    
    #break


# In[ ]:


print(len(Coberturas_sucio))


# In[ ]:





# In[ ]:


df = pd.DataFrame(list(zip(Empresa, Precio, Coberturas_limpio)), columns=['Empresa', 'Precio', 'Coberturas'])
df.head()


# In[ ]:


print(df['Coberturas'].str.split('\n')[1])


# In[ ]:


print(Anio_Vehiculo)


# In[ ]:


print(Vehiculo)


# In[ ]:


Cotizacion_Auto = {'Empresa': Empresa, 'Precio': Precio, 'Coberturas': Coberturas_limpio}
#Cotizacion_Auto


# In[ ]:


lista = []

for key, value in Cotizacion_Auto.items():
    if key == 'Coberturas':
        for lis in value:
            for x in lis.split('\n'):
                if len(x.replace('-', '')) != 0 and x not in ('Daños Materiales', 'Robo Total', 'Asistencias', 'Responsabilidad Civil por daños a terceros(LUC1)'):
                    lista.append(x)
                       


# In[ ]:


#lista


# In[ ]:


lista = [i for i in Coberturas_limpio[0].split('\n') if len(i.replace('-', '')) != 0]


# In[ ]:


for x in lista:
    if x in ('Daños Materiales', 'Robo Total', 'Asistencias', 'Responsabilidad Civil por daños a terceros(LUC1)'):
        lista.remove(x)


# In[ ]:


columnas = [i.replace('(', '').replace(')', '').replace('.', '').replace(' ', '_').replace('/', '').replace('__', '_') for i in lista[::2] ]


# In[ ]:


datos = [i for i in lista if i in (lista[::2])]


# In[ ]:


res = [] 
for i in datos: 
    if i not in res: 
        res.append(i) 


# In[ ]:


lista = [i for i in Coberturas_limpio[0].split('\n') if len(i.replace('-', '')) != 0]

for x in lista:
    if x in ('Daños Materiales', 'Robo Total', 'Asistencias', 'Responsabilidad Civil por daños a terceros(LUC1)'):
        lista.remove(x)
        
columnas = [i.replace('(', '').replace(')', '').replace('.', '').replace(' ', '_').replace('/', '').replace('__', '_') for i in lista[::2] ]
datos = [i for i in lista if i in (lista[::2])]
res = [] 
for i in datos: 
    if i not in res: 
        res.append(i) 


# In[ ]:





# In[ ]:


res


# In[ ]:


datos


# In[ ]:





# In[ ]:





# In[ ]:


df.head()


# In[ ]:


df.head()


# In[ ]:


df_cober = df['Coberturas'].str.split('\n', expand=True).copy()


# In[ ]:


df_cober.head()


# In[ ]:


df_cober.shape


# In[ ]:


df_cober[df_cober.isin(['-'])]


# In[ ]:


df_cober.columns = df_cober.iloc[1]


# In[ ]:


df_cober.drop(columns = ['-', 'Daños Materiales', 'Robo Total', 'Asistencias', 'Responsabilidad Civil por daños a terceros(LUC1)'], axis = 1, inplace=True)


# In[ ]:


df_cober.head()


# In[ ]:


df_cober2 = df_cober[~df_cober.isin(res)]
df_cober2.head()


# In[ ]:


df_cober2.dropna(axis=1, how='all', inplace=True)


# In[ ]:


df_cober2.head()


# In[ ]:


columnas_df = list(df_cober2.columns)
columnas_df


# In[ ]:


cols=pd.Series(df_cober2.columns)
for dup in df_cober2.columns[df_cober2.columns.duplicated(keep=False)]: 
    cols[df_cober2.columns.get_loc(dup)] = ([dup + '.' + str(d_idx) 
                                     if d_idx != 0 
                                     else dup 
                                     for d_idx in range(df_cober2.columns.get_loc(dup).sum())]
                                    )
df_cober2.columns=cols


# In[ ]:


df_cober2


# In[ ]:


df_cober2.rename(columns={df_cober2.columns[0]:'Valor_Comercial_de_Indemnización_daños', df_cober2.columns[1]:'Deducible_Danio', df_cober2.columns[2]:'Valor_Comercial_de_Indemnización_Robo', df_cober2.columns[3]:'Deducible_Robo', df_cober2.columns[4]:columnas[4],                          df_cober2.columns[5]:columnas[5], df_cober2.columns[6]:columnas[6], df_cober2.columns[7]:columnas[7], df_cober2.columns[8]:columnas[8], df_cober2.columns[9]:columnas[9],                          df_cober2.columns[10]:columnas[10], df_cober2.columns[11]:columnas[11], df_cober2.columns[12]:columnas[12], df_cober2.columns[13]:columnas[13], df_cober2.columns[14]:columnas[14],                          df_cober2.columns[15]:columnas[15], df_cober2.columns[16]:columnas[16], df_cober2.columns[17]:columnas[17], df_cober2.columns[18]:columnas[18], df_cober2.columns[19]:columnas[19]}, inplace=True)


# In[ ]:





# In[ ]:





# In[27]:


Coberturas_limpio = []
Empresa = []
Precio = []
vehiculo_cot = []
anio_cot = []
edad = []
sexo_cot = []
cp_lis = []


for x in range(len(cp)):
    Vehiculo = []
    Anio_Vehiculo = []
    print(x)
    print('-----------------------------------------------------------------------------------------------------------------')
    driver.get('https://www.autocompara.com/')
    time.sleep(15)
    
    try:
        intro = driver.find_element_by_xpath("//a[@class='vtw-close cybXButton']")
        intro.click()
    except:
        print("No clic")
    
    

    anio = driver.find_element_by_xpath("//div[@class='select-button-home year select-custom']")
    anio.click()
    anio_drop = driver.find_elements_by_xpath("//div[@class='ng-dropdown-panel-items scroll-host']")
    for anio in anio_drop:
        years =  anio.find_elements_by_xpath(".//div[@class='ng-option ng-option-marked']") +  anio.find_elements_by_xpath(".//div[@class='ng-option']")
        for year in years:
            year_boton = year.find_element_by_xpath("./span[@class='ng-option-label']")
            print(year)
            print(year.text)
            Anio_Vehiculo.append(year.text)
            year_boton.click()
            print(year_boton)
            break
    
    
    time.sleep(10)
    driver.find_element_by_xpath("//div[@class='ng-select-container']").click()
    modelos_drop = driver.find_elements_by_xpath("//div[@class='ng-dropdown-panel-items scroll-host']")
    for modelo in modelos_drop:
        models = modelo.find_elements_by_xpath(".//div[@class='ng-option ng-option-child ng-option-marked']") + modelo.find_elements_by_xpath(".//div[@class='ng-option ng-option-child']")
        for model in models:
            print(model)
            print(model.text)
            Vehiculo.append(model.text)
            model.click()
            break
    
    
    time.sleep(5)
    cotiza_boton = driver.find_element_by_xpath("//div[@class='button-cotiza']")
    cotiza_boton.click()
    time.sleep(0.5)
    try:
        cotiza_boton.click()
    except:
        print('Un solo boton cotiza')
    
    
    time.sleep(2)
    if sexo == 1:
        sexo_boton = driver.find_element_by_xpath("//label[@class='btn icon-mujer']").click()
    
    
    driver.find_element_by_xpath("//input[@id='date']").click()
    #Año
    time.sleep(2)
    driver.find_element_by_css_selector("[title^='Select year']").click()
    select_year = Select(driver.find_element_by_css_selector("[title^='Select year']"))
    #Cambiar el año para que loopee
                                #select_year.select_by_value(str(2004 - x))
    select_year.select_by_value(str(2003))
    #Cambiar el año para que loopee
    #driver.find_element_by_css_selector("[title^='Select year']").click()
    
    
    #Selección de día
    time.sleep(2)
    dias = driver.find_elements_by_xpath("//div[@class='btn-light']")
    for dia in dias:
        if dia.text == '1':
            dia.click()
            break
            
    
    #Selección de Nombre
    time.sleep(2)
    name = driver.find_element_by_xpath("//input[@id='name']")
    name.clear
    name.send_keys(nombre)
    
    
    #Selección de CP
    time.sleep(2)
    codigo_postal = driver.find_element_by_xpath("//input[@id='txtPostalCode']")
    codigo_postal.clear
    codigo_postal.send_keys(cp[x])
    
    
    #Selección Correo
    time.sleep(2)
    correo = driver.find_element_by_xpath("//input[@id='email']")
    correo.clear
    correo.send_keys(email)
    
    
    #Selección Teléfono
    time.sleep(2)
    phone = driver.find_element_by_xpath("//input[@id='phone']")
    phone.clear
    phone.send_keys(telefono)
    
    
    time.sleep(2)
    driver.find_element_by_xpath("//button[@class='btn btn-red']").click()
    
    
    time.sleep(35)
    detalles = driver.find_elements_by_xpath("//button[@class='btn btn-lnk details']")
    
    
    #DB = {'Auto': ['Empresa':[], 'Coberturas':[]]}

    for detalle in detalles:
        Coberturas_sucio = []
        detalle.click()
        cobertura_actual = detalle.find_element_by_xpath("//div[@class='col-12']")
        coberturas = cobertura_actual.find_elements_by_xpath("//div[@class='row']")
        for cobertura in coberturas:
            Coberturas_sucio.append(cobertura.text)
    
        Coberturas_limpio.append(Coberturas_sucio[3])
        Empresa.append(cobertura_actual.find_element_by_xpath("//span[@class='ng-value-label']").text)
        Precio.append(detalle.find_element_by_xpath("//div[@class='price mb-2']/p[@class='price']").text)
        
        vehiculo_cot.append(Vehiculo)
        anio_cot.append(Anio_Vehiculo)
        sexo_cot.append(sexo)
        edad.append(str(2004 - x))
        
        cp_lis.append(cp[x])
    
        cobertura_actual.find_element_by_xpath("//button[@id='btn-close-modal']").click()
        time.sleep(5)
    
        #break
    time.sleep(5)


# In[28]:


df = pd.DataFrame(list(zip(cp_lis, anio_cot, vehiculo_cot, sexo_cot, edad, Empresa, Precio, Coberturas_limpio)), columns=['CP', 'Anio_vehiculo', 'Vehiculo', 'Sexo', 'edad', 'Empresa', 'Precio', 'Coberturas'])
df.head()


# In[29]:


lista = [i for i in Coberturas_limpio[0].split('\n') if len(i.replace('-', '')) != 0]

for x in lista:
    if x in ('Daños Materiales', 'Robo Total', 'Asistencias', 'Responsabilidad Civil por daños a terceros(LUC1)'):
        lista.remove(x)
        
columnas = [i.replace('(', '').replace(')', '').replace('.', '').replace(' ', '_').replace('/', '').replace('__', '_') for i in lista[::2] ]
datos = [i for i in lista if i in (lista[::2])]
res = [] 
for i in datos: 
    if i not in res: 
        res.append(i) 


# In[30]:


df_cober = df['Coberturas'].str.split('\n', expand=True).copy()
df_cober.head()


# In[31]:


df_cober.columns = df_cober.iloc[1]
df_cober.drop(columns = ['-', 'Daños Materiales', 'Robo Total', 'Asistencias', 'Responsabilidad Civil por daños a terceros(LUC1)'], axis = 1, inplace=True)
df_cober.head()


# In[32]:


df_cober2 = df_cober[~df_cober.isin(res)]
df_cober2.head()


# In[33]:


df_cober2.dropna(axis=1, how='all', inplace=True)


# In[34]:


cols=pd.Series(df_cober2.columns)
for dup in df_cober2.columns[df_cober2.columns.duplicated(keep=False)]: 
    cols[df_cober2.columns.get_loc(dup)] = ([dup + '.' + str(d_idx) 
                                     if d_idx != 0 
                                     else dup 
                                     for d_idx in range(df_cober2.columns.get_loc(dup).sum())]
                                    )
df_cober2.columns=cols


# In[35]:


df_cober2.rename(columns={df_cober2.columns[0]:'Valor_Comercial_de_Indemnización_daños', df_cober2.columns[1]:'Deducible_Danio', df_cober2.columns[2]:'Valor_Comercial_de_Indemnización_Robo', df_cober2.columns[3]:'Deducible_Robo', df_cober2.columns[4]:columnas[4],                          df_cober2.columns[5]:columnas[5], df_cober2.columns[6]:columnas[6], df_cober2.columns[7]:columnas[7], df_cober2.columns[8]:columnas[8], df_cober2.columns[9]:columnas[9],                          df_cober2.columns[10]:columnas[10], df_cober2.columns[11]:columnas[11], df_cober2.columns[12]:columnas[12], df_cober2.columns[13]:columnas[13], df_cober2.columns[14]:columnas[14],                          df_cober2.columns[15]:columnas[15], df_cober2.columns[16]:columnas[16], df_cober2.columns[17]:columnas[17], df_cober2.columns[18]:columnas[18], df_cober2.columns[19]:columnas[19]}, inplace=True)


# In[36]:


df_cober2.head()


# In[37]:


df_final = pd.concat([df, df_cober2], axis=1)
df_final.head()


# In[38]:


df_final['edad'] = datetime.date.today().year - df_final['edad'].astype(int)


# In[39]:


df_final['Sexo'] = df_final['Sexo'].astype(str)


# In[40]:


df_final.loc[df_final['Sexo'] == '0', 'Sexo_'] = 'Masculino'
df_final.loc[df_final['Sexo'] == '1', 'Sexo_'] = 'Femenino'


# In[41]:


df_final.head()


# In[42]:


df_final.to_csv('anualizada_cp.csv', index=False)


# In[ ]:




