from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import location
from django.http import JsonResponse

import json

from myapp.models import Car

def index(request):
    response = json.dumps([{}])
    return HttpResponse(response, content_type='text/json')

def get_County(request, County):
    if request.method == 'GET':
        try:
            import numpy
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score
            import pandas as pd
            #import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.cluster import KMeans
            #print(County)
            df = pd.read_csv(r'C:\Users\mnzava\Downloads\popstreet.csv', low_memory=False)
            dd = pd.read_csv(r'C:\Users\mnzava\Downloads\ppdns.csv', low_memory=False)
            de = pd.read_csv(r'C:\Users\mnzava\Downloads\popub.csv', low_memory=False)
            de['County'] = de['County'].str.upper()
            dd['County'] = de['County'].str.upper()
            dff = pd.merge(df, de, 
                   on='County', 
                   how='inner')
            dfff = pd.merge(dff, dd,
                on='County',
                how='inner')
            dg=dfff
            from sklearn.preprocessing import StandardScaler
            dg['Total_y'] = dg.Total_y.astype(str).str.replace(',', '').astype(float)
            dg['Total_x'] = dg.Total_x.astype(str).str.replace(',', '')
            dg['Total_x'] = dg.Total_x.astype(str).str.replace('-', '')
            dg['Male_x'] = dg.Male_x.astype(str).str.replace(',', '')
            dg['Male_x'] = dg.Male_x.astype(str).str.replace('-', '')
            dg['Male_y'] = dg.Male_y.astype(str).str.replace(',', '')
            dg['Male_y'] = dg.Male_y.astype(str).str.replace('-', '')
            dg['Female_x'] = dg.Female_x.astype(str).str.replace(',', '')
            dg['Female_x'] = dg.Female_x.astype(str).str.replace('-', '')
            dg['Female_y'] = dg.Female_y.astype(str).str.replace(',', '')
            dg['Female_y'] = dg.Female_y.astype(str).str.replace('-', '')
            dg['Intersex'] = dg.Intersex.astype(str).str.replace(',', '')
            dg['Intersex'] = dg.Intersex.astype(str).str.replace('-', '')
            dg.rename(columns={'Population Density (No. per Sq. Km)': 'ppdens'}, inplace=True)
            dg.rename(columns={'Land Area (Sq. Km)': 'Land'}, inplace=True)
            dg["Intersex"] = pd.to_numeric(dg["Intersex"], downcast="float")
            dg["Total_x"] = pd.to_numeric(dg["Total_x"], downcast="float")
            dg["Male_x"] = pd.to_numeric(dg["Male_x"], downcast="float")
            dg["Female_x"] = pd.to_numeric(dg["Female_x"], downcast="float")
            dg["Male_y"] = pd.to_numeric(dg["Male_y"], downcast="float")
            dg["Female_y"] = pd.to_numeric(dg["Female_y"], downcast="float")
            dg["Total_y"] = pd.to_numeric(dg["Total_y"], downcast="float")
            dg["Population"] = pd.to_numeric(dg["Population"], downcast="float")
            dg["Land"] = pd.to_numeric(dg["Land"], downcast="float")
            dg["ppdens"] = pd.to_numeric(dg["ppdens"], downcast="float")
            dg['Total_x'] = dg['Total_x'].fillna(0)
            dg['Total_y'] = dg['Total_y'].fillna(0)
            dg['ppdens'] = dg['ppdens'].fillna(0)
            dg['Female_x'] = dg['Female_x'].fillna(0)
            dg['Female_y'] = dg['Female_y'].fillna(0)
            dg['Male_x'] = dg['Male_x'].fillna(0)
            dg['Male_y'] = dg['Male_y'].fillna(0)
            dg['Land'] = dg['Land'].fillna(0)
            dg['Population'] = dg['Population'].fillna(0)
            dg['Intersex'] = dg['Intersex'].fillna(0)
            dh = dg.drop('County', axis=1)    
            X = dg.values[:, 1:]
            X = np.nan_to_num(X)
            Clus_dataSet = StandardScaler().fit_transform(X)
            Clus_dataSet
            X.shape
            clusterNum = 5
            k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
            k_means.fit(X)
            labels = k_means.labels_
            labels = k_means.labels_
            #print(labels)
            dg["Clus_km"] = labels
            dg.head(5)
            dg.groupby('Clus_km').mean()
            phone_id = np.where(dg.County == County)[0][0]
            suggestion = dh.iloc[phone_id].values.reshape(1, -1)
            suggestions = np.array(suggestion, dtype=np.float)
            prediction = k_means.predict(suggestions)
            prediction_cls = int(prediction)
            temp_df = dg.loc[dg['Clus_km'] == prediction_cls]
            temp_df = temp_df.sample(3)
            temp_df
            list1 = numpy.array(temp_df['County'])
            #print(list1)
            list2=list1.tostring()
            #print(list2)
            for z in list1:
                loc = location.objects.get(County=z)
                #print(loc)
                response = json.dumps([{ 'County': loc.County, 'Population': loc.Population, 'Land': loc.Land}])
            #print(response)
        except:
            response = json.dumps([{ 'Error': 'No County with that name'}])
    return HttpResponse(response, content_type='text/json')    

@csrf_exempt
def add_car(request):
    if request.method == 'POST':
        payload = json.loads(request.body)
        car_name = payload['car_name']
        top_speed = payload['top_speed']
        car = Car(name=car_name, top_speed=top_speed)
        try:
            car.save()
            response = json.dumps([{ 'Success': 'Car added successfully!'}])
        except:
            response = json.dumps([{ 'Error': 'Car could not be added!'}])
    return HttpResponse(response, content_type='text/json')
