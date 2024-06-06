#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import sklearn


while True:
    try:
        typeRandom = int(input('tipo random para reducir 1:choice 2: D.normal 3: D.standard_normal 4: choice Prob 5: Sample-> ')) 
        if 1 <= typeRandom <= 5:
            # La entrada está dentro del rango
            print("¡Bien hecho! Has introducido un número válido.")
            break  # Salir del bucle while ya que tenemos un valor válido
        else:
            print(f"El número debe estar entre 1 y 3. Inténtalo de nuevo.")
    except ValueError:
        print("¡Oops! Parece que no ingresaste un número entero válido. Inténtalo de nuevo.")

print()

while True:
    try:
        SampleSize = int(input('El tamaño de la muestra -> '))
        if SampleSize >0:
            # La entrada está dentro del rango
            #print("¡Bien hecho! Has introducido un número válido.")
            break  # Salir del bucle while ya que tenemos un valor válido
        else:
            print(f"El número debe ser mayor a 0. Inténtalo de nuevo.")
    except ValueError:
        print("¡Oops! Parece que no ingresaste el tamaño de la muestra. Inténtalo de nuevo.")      
        
        
     

csv_path = "D:/pythonProjects/JupyterProjects/recbole/dataset/bgg/csv/bgg_ratings.csv"
df = pd.read_csv(csv_path)

n_users = df.username.unique().shape[0]
n_items = df.gameId.unique().shape[0]
n_rating = df.rating.unique().shape[0]
print (str(n_users) + ' users')
print (str(n_items) + ' items')
print (str(n_rating) + ' rating')

df2 = df.sort_values(by='username').drop_duplicates()

if typeRandom == 1: 
    type='choice'
    # Realizar el muestreo aleatorio utilizando la función `choice` de NumPy
    ind_muestra = np.random.choice(df2.index, size=SampleSize, replace=False)

    # Seleccionar las filas correspondientes a los índices de la muestra
    df_muestra = df2.loc[ind_muestra]
    


elif typeRandom == 2:
    type='normal'
    # Generar más muestras que el tamaño deseado para evitar duplicados
    num_muestras_ad = SampleSize * 2

    # Realizar el muestreo aleatorio utilizando una distribución normal estándar
    ind_muestra = np.random.normal(loc=0, scale=1, size=num_muestras_ad) * len(df2)

    # Redondear los índices y asegurarse de que estén dentro del rango del DataFrame
    ind_muestra = np.round(ind_muestra).astype(int)
    ind_muestra = np.clip(ind_muestra, 0, len(df2) - 1)

    # Eliminar valores duplicados y seleccionar las primeras muestras únicas
    ind_muestra = np.unique(ind_muestra)[:SampleSize]

    # Seleccionar las filas correspondientes a los índices de la muestra
    df_muestra = df2.iloc[ind_muestra]
    

elif typeRandom == 3:
    type='standard_normal'
    # Generar índices únicos utilizando np.random.standard_normal sin valores duplicados
    ind_unicos = set()
    while len(ind_unicos) < SampleSize:
        new_ind = np.random.standard_normal(SampleSize) * len(df)
        new_ind = np.round(new_ind).astype(int)
        new_ind = np.unique(new_ind)
        new_ind = np.clip(new_ind, 0, len(df) - 1)
        ind_unicos.update(new_ind)

    # Convertir los índices únicos a una lista y tomar los primeros "SampleSize" índices
    ind_muestra = list(ind_unicos)[:SampleSize]

    # Seleccionar las filas correspondientes a los índices de la muestra
    df_muestra = df2.iloc[ind_muestra]
    


elif typeRandom == 4:
    type='choiceProb'
    # Calcular las probabilidades para cada fila
    probabilidades = np.random.standard_normal(len(df2))
    probabilidades = np.abs(probabilidades)  # Tomamos el valor absoluto para asegurarnos de que las probabilidades sean positivas
    probabilidades = probabilidades / probabilidades.sum()  # Normalizamos las probabilidades para que sumen 1

    # Generar índices únicos utilizando np.random.choice con probabilidades
    ind_muestra = np.random.choice(df2.index, size=SampleSize, replace=False, p=probabilidades)

    # Seleccionar las filas correspondientes a los índices de la muestra
    df_muestra = df2.loc[ind_muestra]


elif typeRandom == 5:
    type='Sample'
    # Verificar si el tamaño de la muestra deseada es mayor que el tamaño del DataFrame
    if SampleSize > len(df2):
        SampleSize = len(df2)

    # Generar índices aleatorios utilizando np.random.sample
    ind_aleatorios = np.random.sample(df2.shape[0]) < SampleSize / df2.shape[0]

    # Seleccionar las filas correspondientes a los índices generados
    df_muestra = df2[ind_aleatorios]
 
 
 
 
# Guardar el DataFrame reducido en un nuevo archivo CSV
df_muestra.to_csv(f"D:/pythonProjects/JupyterProjects/data/bgg/datos_reducidos_{type}_{SampleSize}.csv", index=False)

# Mostrar las primeras filas del DataFrame reducido
print("\nDataFrame reducido:")
print(df_muestra.head())


