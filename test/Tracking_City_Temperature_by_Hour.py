"""
Created on  2/25/2019
@author: Jingchao Yang
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import datalab.bigquery as bq
import pandas as pd
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "..\\geotab-intelligence-4a49636c730a.json"

# SQL = """
#       SELECT LocalDate, LocalHour, AVG(Temperature_C) as AvgTemperature
#         FROM `geotab-public-intelligence.Weather.Temperature`
#        WHERE State LIKE 'North Carolina'
#          AND City LIKE 'Charlotte'
#        GROUP BY LocalDate, LocalHour
#        ORDER BY LocalDate, LocalHour
# """
#
# df = bq.Query(SQL).to_dataframe(dialect='standard')
df = pd.read_csv('../dataSample/Atlanta-20190311-122351.csv')
print(df)


def temperature_plot(date, hour, temperature):
    dates = [' '.join(x) for x in zip(date, [str(v).zfill(2) for v in hour])]
    dates = [datetime.datetime.strptime(x, "%Y-%m-%d %H") for x in dates]
    locator = mdates.HourLocator()
    formatter = mdates.DateFormatter('%Y-%m-%d - %H:00')

    fig, ax = plt.subplots(ncols=1, figsize=(30, 5))
    ax.scatter(dates, temperature, s=300, c=temperature, cmap='plasma', zorder=2)
    ax.plot(dates, temperature, '--k', zorder=1)

    ax.set_ylim([np.floor(min(temperature)) - 1, np.ceil(max(temperature)) + 1])
    ax.set_xlim([min(dates), max(dates)])
    ax2 = ax.twinx()
    ax2.set_ylim([ax.get_ylim()[0] / 0.5556 + 32, ax.get_ylim()[1] / 0.5556 + 32])

    ylabel_left = 'Temperature ($^\circ$C)'
    ylabel_right = 'Temperature ($^\circ$F)'
    ax.set_xlabel('Date and Time', fontsize=14, labelpad=20)
    axlab = ax.set_ylabel(ylabel_left, fontsize=14, labelpad=10)
    ax2lab = ax2.set_ylabel(ylabel_right, fontsize=14, labelpad=10)
    ax2.grid(None)

    fig.autofmt_xdate()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    return plt


a = temperature_plot(df['LocalDate'], df['LocalHour'], df['AvgTemperature'])
a.show()
