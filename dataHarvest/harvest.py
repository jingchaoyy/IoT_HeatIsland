"""
Created on  4/1/2019
@author: Jingchao Yang

Data harvest every day at 1:00 A.M.
https://stackoverflow.com/questions/15088037/python-script-to-do-something-at-the-same-time-every-day
"""
# import schedule
from schedule import *
import schedule
import time
from google.cloud import bigquery
import os
from datetime import date

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "..\\geotab-intelligence-4a49636c730a.json"
client = bigquery.Client()


def job(t):
    today = str(date.today())
    print('start', today)  # '2017-12-26'

    SQL = """
            SELECT *
            FROM `geotab-intelligence.Weather.Temperature`
            WHERE State LIKE 'Georgia' 
            AND City LIKE 'Atlanta' 
    """
    query_job = client.query(
        SQL,
    )

    df = query_job.to_dataframe()
    file_name = today + '.csv'
    df.to_csv(file_name, encoding='utf-8', index=False)
    print('success')

    return


schedule.every().day.at("12:30").do(job, 'It is 12:30')

while True:
    schedule.run_pending()
    time.sleep(60)  # wait one minute
