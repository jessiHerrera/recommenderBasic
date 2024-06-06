#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

dir = 'D:/pythonProjects/JupyterProjects/data/bgg/'
typeFile = str(input('tipo de fichero atomic: ')) #inter,item, user
filename_rating = str(input('Fichero csv a convertir: '))
ffile = dir + "" + filename_rating


#filename_game = str(input('Si el tipo es ITEM introducir el Fichero game: ')) 


#df = pd.read_csv(filename_rating)
#df.head()


#**** Convirtiendo el file ratings CSV to atomic File inter*******
if typeFile == 'inter':
    #df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/bgg_ratings.csv")
    df = pd.read_csv(ffile)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    #Convert DATE to TIMESTAMP
    df['timestamp'] = df.date.values.astype(np.int64) // 10 ** 9
    df.head()
    #sort,drop duplicate row
    df2 = df.drop_duplicates().sort_values(by='username')
    
    #choice rows - para reducir el número de registros
    #vc=df.value_counts('username').reset_index(name='count')
    #df_u = vc[vc['count'] > 75]
    #df3 =df2.merge(df_u, on="username")

    # change column name 
    temp = df2[['username', 'gameId', 'rating','timestamp']].rename(
        columns={'username': 'user_id:token', 'gameId': 'item_id:token', 'rating': 'rating:float','timestamp': 'timestamp:float'})
    temp
    # save atomic file in dataset format for using with recbole
    temp.to_csv('D:/pythonProjects/JupyterProjects/data/bgg/bgg.inter', index=False, sep='\t')

elif typeFile == 'item':
    #**** Convirtiendo el file games CSV to atomic File item*******
    #filename_game = 'r' + dir + filename_game
    df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/bgg_games.csv")
    
    #df.head()

    # change column name 
    #'id', 'name', 'expansion','year','minplayers','maxplayers','age','numratings','numcomments','views','wishlist'
    df2 = df.sort_values(by='id')
    itemp = df2[['id', 'name', 'year']].rename(
        columns={'id': 'item_id:token', 'name': 'movie_title:token_seq', 'year': 'release_year:token'})
    itemp

    # save atomic file in dataset format for using with recbole
    itemp.to_csv('D:/pythonProjects/JupyterProjects/data/bgg/bgg.item', index=False, sep='\t')


elif typeFile == 'user':
    #**** Convirtiendo el file ratings CSV to atomic File user*******
    df = pd.read_csv(r"D:/pythonProjects/JupyterProjects/data/bgg/bgg_ratings.csv")

    utemp = df[['username']].drop_duplicates().rename(columns={'username': 'user_id:token'})
    utemp

    # save atomic file in dataset format for using with recbole
    utemp.to_csv('D:/pythonProjects/JupyterProjects/data/bgg/bgg.user', index=False, sep='\t')


