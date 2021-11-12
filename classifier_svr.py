import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVR
from sklearn.svm import SVC
from datetime import date
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# read dataset file
x = pd.read_csv('dataset.csv')
dataset = pd.read_csv('dataset.csv')
a=np.array(x)

# input position 
weekday = input("What's the day of the week(sun=1....saturday=7)? ")
total_customers_served = input("What's the total customers served? ")

# predict service time from dataset based on position and branch
c = np.column_stack((x.weekday, x.number_customers))
c.shape
servicetime_column= dataset.iloc[:,4].values
svr_lin1 = SVR(kernel='linear')
svr_lin1.fit(c, servicetime_column)
predicted_service_time = svr_lin1.predict([[weekday,total_customers_served]])


#convert predicted_service_time from list to float
pst = list(predicted_service_time)
pst2 = float("".join(map(str, pst)))

# predict waiting time based on position and predicted service time3
y = np.array(x['waiting_time'])
x = np.column_stack((x.weekday, x.service_time))
x.shape
svr_lin = SVR(kernel='linear')
svr_lin.fit(x, y)
prediction = svr_lin.predict([[weekday, predicted_service_time]])


pwt = list(prediction)
pwt2 = float("".join(map(str, pwt)))

position = int(input("What's your position in the queue? "))
waiting_time = (position-1)*pwt2

print("Predicted Waiting Time")
print(waiting_time)