"""
Created on  2/26/2019
@author: Jingchao Yang

code sample: https://data.geotab.com/weather/temperature
"""
# import datalab.bigquery as bq
from google.cloud import bigquery
import pandas as pd
import folium
import branca.colormap as cm
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "..\\geotab-intelligence-4a49636c730a.json"
client = bigquery.Client()

SQL = """
SELECT Latitude_SW as sw_lat, Longitude_SW as sw_lon, Latitude_NE as ne_lat, Longitude_NE as ne_lon, Temperature_C as Temperature
  FROM `geotab-intelligence.Weather.Temperature`
 WHERE (State LIKE 'Illinois')
   AND (City LIKE 'Chicago')
   AND LocalHour = '14'
"""

query_job = client.query(
    SQL,
    # Location must match that of the dataset(s) referenced in the query.
    # location="US",
)
# print(query_job)
df_geo = query_job.to_dataframe()
# df_geo = pd.read_csv('../dataSample/GA17-20190311-123402.csv')
# print(df_geo)

magnitudes = pd.DataFrame(df_geo['Temperature'], columns=['Temperature'])
polygons_out = {'type': 'FeatureCollection', 'features': []}

for index, row in df_geo.iterrows():
    sw_lon_temp = row['sw_lon']
    sw_lat_temp = row['sw_lat']

    se_lon_temp = row['ne_lon']
    se_lat_temp = row['sw_lat']

    ne_lon_temp = row['ne_lon']
    ne_lat_temp = row['ne_lat']

    nw_lon_temp = row['sw_lon']
    nw_lat_temp = row['ne_lat']

    poly_points = [[sw_lon_temp, sw_lat_temp], [se_lon_temp, se_lat_temp], [ne_lon_temp, ne_lat_temp],
                   [nw_lon_temp, nw_lat_temp]]

    feature = {'type': 'Feature',
               'properties': {'Temperature': row['Temperature']},
               'id': index,
               'geometry': {'type': 'Polygon',
                            'coordinates': [poly_points]}}

    polygons_out['features'].append(feature)

magnitudes.to_csv('../output/chicago_magnitudes_polygons14.csv', index_label='id')

loclat = df_geo['sw_lat'].mean()
loclon = df_geo['sw_lon'].mean()

colorList = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5']
colorList.reverse()
linear = cm.LinearColormap(colorList, vmin=df_geo['Temperature'].min(), vmax=df_geo['Temperature'].max())
linear = linear.to_step(17)

magnitudes = pd.read_csv('../output/LA_magnitudes_polygons14.csv')
magnitudes['Temperature'] = magnitudes['Temperature'].apply(lambda x: round(x * 2) / 2)
mags_dict = magnitudes.set_index('id')['Temperature']

m = folium.Map(location=[loclat + 0.02, loclon + 0.1], zoom_start=12, width=1500, height=500, tiles='cartoDB Positron')

folium.GeoJson(
    polygons_out,
    style_function=lambda feature: {
        'fillColor': linear(mags_dict[feature['id']]),
        'color': 'black',
        'weight': 0.2,
        'fillOpacity': 0.8
    }
).add_to(m)

m.save('../output/Chicago14.html')
