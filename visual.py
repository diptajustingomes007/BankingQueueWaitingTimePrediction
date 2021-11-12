from typing import Any, Union

import pandas as pd
import matplotlib.pyplot as plt


# read dataset file
# dataset = pd.read_csv('dataset.csv')
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

r = pd.read_csv("results_knn.csv")

# weeekday_column = dataset.iloc[:,2].values
# waiting_time_column = dataset.iloc[:,3].values
# service_time_column = dataset.iloc[:,4].values
# number_customers_column = dataset.iloc[:,5].values

position = r.iloc[:, 0].values
wt = r.iloc[:, 2].values

#Visualization
_, ax = plt.subplots()

# #scatter plot service time with respect to waiting Time
# ax.scatter(service_time_column, waiting_time_column, s = 10, color = "red", alpha = 0.75)
# ax.set_title("Service TIme with respect to Waiting time")
# ax.set_xlabel("Service Time")
# ax.set_ylabel("Waiting Time")

# Barplot service time with respect to position
ax.bar(position, wt, color = '#539caf', align = 'center')
ax.set_ylabel("Waiting Time")
ax.set_xlabel("Position")
ax.set_title('Waiting time with respect to Position KNN')


#Output
plt.show()