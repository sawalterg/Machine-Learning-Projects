
### PART 1


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logins_df = pd.read_json("logins.json")


# Turn dataframe index into timeseries data

login_edited = logins_df.reindex(pd.DatetimeIndex(logins_df['login_time']))

# Turn login time to a single event, represented by 1

login_edited['count'] = 1

del logins_edited['login_time']

# Resample the data into 15 minutes and sum the total data

fifteen_minute_window = logins_df.resample("15Min").sum()

fifteen_minute_window_null = fifteen_minute_window[fifteen_minute_window['count'].isnull()]
# Sort values 

fifteen_minbute_window = logins_df.sort_values(by = ['login_time'], ascending = False)

# Give the top values 

logins_df[:5]


# Plot data

view = fifteen_minute_window['1970-01':'1970-04']

_ = view.plot()
_ = plt.xticks(rotation=45)

plt.show()



# Look at the mean hourly breakout of the data


hourly_breakout_mean = view['count'].groupby(view.index.hour).mean()
hourly_breakout_median = view['count'].groupby(view.index.hour).median()
_ = plt.legend(loc = 'upper left')
_ = hourly_breakout_mean.plot(color = 'blue')
_ = hourly_breakout_median.plot(color = 'brown')
_ = plt.title('Logins by hour of the day')
_ = plt.xlabel('Login Hour')
_ = plt.ylabel('Number of events')
plt.show()
# Plot this data

_ = hourly_breakout.plot()
plt.show()

# Compute summary statistics

hourly_breakout.describe()


# Look at weekly breakout of data

weekly_breakout_mean = view['count'].groupby(view.index.weekday).mean()
weekly_breakout_median = view['count'].groupby(view.index.weekday).median()
_ = plt.legend(loc = 'upper left')
_ = weekly_breakout_mean.plot(color = 'blue')
_ = weekly_breakout_median.plot(color = 'brown')
_ = plt.title('Logins by day of the week')
_ = plt.xlabel('Login Day')
_ = plt.ylabel('Number of events')
plt.show()

# Compute summary statistics

weekly_breakout.describe()



### PART 2

## 1. A metric for determining the success of Ultimate's experiment would be the proportion of trips in Ultimate Gotham vs. Metropolis.

## 2a. This experiment could be implemented by offering a statistically significant portion of drivers the promotion and leaving the rest of the 
# drivers with no special promotion to pay their toll expenses. When selecting riders for the promotion, it would be important to select riders who
# are average in terms of their habits and truly random, lest we introduce bias to the experiment.

# 2b. We could could determine if our experiment caused a significant change using a two proportion z test ( assume p1 = p2). This would be done by assuming the null hypothesis (no difference between proportions) is
# true and if our p value exceeds a significance level of 0.05, the alternative hypothesis (the promotion is effective) would be true.

# 3c. If we rejected the null hypothesis, the team's could be considered provisionally effective, and likely in need of another experiment. If the null
# hypothesis was confirmed, it wouldn't necessarily mean the promotion was not effective and the drivers selected for the promotion would need to be examined
# as well as the time period and other external factors to rule out any confounding variables.




### PART 3


# Open as JSON

import json

with open('ultimate_data_challenge.json', 'r') as df:
    data = json.load(df)
    
rider_df = pd.DataFrame(data)


# Create target variable

rider_df['retention_positive'] = pd.Timedelta(60, unit = 'd') < pd.to_datetime(rider_df['last_trip_date']) - pd.to_datetime(rider_df['signup_date'])

rider_df['signup_date'] = pd.to_datetime(rider_df['signup_date'])
rider_df['last_trip_date'] = pd.to_datetime(rider_df['last_trip_date'])


del rider_df['phone'], rider_df['avg_rating_by_driver'], rider_df['avg_rating_of_driver']


# Split into test and training set for analysis

x = rider_df.iloc[:, 1:9].values
y = rider_df.iloc[:,9].values
# We have some variables that need to be encoded for processing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 2] = labelencoder_X_1.fit_transform(x[:,2])
labelencoder_X_2 = LabelEncoder()
x[:, 7] = labelencoder_X_2.fit_transform(x[:,7])




# Create dummy variables for just the countries. To avoid trap (subtract one dummy variable), you take out one encoder
onehotencoder = OneHotEncoder(categorical_features = [2])
x = onehotencoder.fit_transform(x).toarray()

# Remove one dummy variable from the countries one hot columns
np.delete(x, [2], axis=1)



                
# Splitting Training Sheets to Test Sheets
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) 




# See if the target varifable is balanced

rider_df.retention_positive[rider_df.retention_positive != True].count()



# Explore the data

rider_df.head()

rider_df.describe()


rider_df.isnull().sum()


# Looks like there are a lot of missing ratings, lets first look athe 'avg_rating_of_driver' column

rider_df[rider_df['avg_rating_of_driver'].isnull()].head(15)

# It doesn't look like this variable has much predictive value and is worth



# Check data types

rider_df.dtypes





from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()                                               
x_train = sc_x.fit_transform(x_train)                        
x_test = sc_x.transform(x_test)     


# Fitting Logistic Regression to the training set 
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)       
classifier.fit(x_train, y_train)        

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, y_pred)

# 1. 


### PART 4 - Using PCA for feature extraction




# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()                                               
x_train = sc_x.fit_transform(x_train)                        
x_test = sc_x.transform(x_test)          




# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf') 
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)



# Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
