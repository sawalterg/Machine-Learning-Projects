import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataframes from the csvs

df_users = pd.read_csv("takehome_users.csv", encoding='latin-1')

df_engagement = pd.read_csv("takehome_user_engagement.csv")

df_users.head()
df_engagement.head()


df_engagement.time_stamp = pd.to_datetime(df_engagement.time_stamp)


"""df_user_grouped = pd.DataFrame(df_user_engagement['visited'].groupby(df_user_engagement.user_id).sum())

df_user_grouped['user_id'] = df_user_grouped.index

df_user_engagement_merged = df_users.merge(df_user_grouped, left_on = 'object_id', right_on = 'user_id', how = 'inner'"""

def week_rolling_count(grp, freq):
    return grp.rolling(freq, on='time_stamp')['user_id'].count() 
df_engagement['seven_day_visits'] = df_engagement.groupby('user_id', as_index=False, group_keys=False).apply(week_rolling_count, '7D') 
df_engagement['adopted_user'] = 3 <= df_engagement['seven_day_visits']


grouped_user = df_engagement.adopted_user.groupby(df_engagement.user_id).sum()

grouped_user_df = pd.DataFrame({'user_id':grouped_user.index, 'is_adopted_user': 1 <= grouped_user})



df_user_merged = df_users.merge(grouped_user_df, left_on = 'object_id', right_on = 'user_id', how = 'inner') 

df_user_merged = df_user_merged.set_index('object_id').sort_index()

df_user_merged = df_user_merged.drop(columns = ['user_id', 'name', 'email', 'creation_time', 'last_session_creation_time'])

df_user_merged.isnull().sum()
df_user_merged.describe()
df_user_merged.info()
df_user_merged.dtypes



df_user_merged['invited_by_user_id'] = df_user_merged['invited_by_user_id'].fillna(0)
df_user_merged['invited_by_user_id'] = df_user_merged['invited_by_user_id'].apply(
        lambda x: 1 if x!=0 else 0)

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
df_user_merged['creation_source'] = labelencoder_X_1.fit_transform(df_user_merged['creation_source'])





onehotencoder = OneHotEncoder(categorical_features = [0])
df_user_merged = onehotencoder.fit_transform(df_user_merged).toarray()
df = x[: , 1:] # Take away one dummy variable to avoid multicolinearity
"""
org = df_user_merged.groupby('org_id').sum()
org_reduced = []
for obs in df_user_merged['org_id']:
    red = org.loc[obs,'is_adopted_user']
    org_reduced.append(red)
df_user_merged['org_id'] = org_reduced

df_user_merged_dum = pd.get_dummies(df_user_merged, columns = ['creation_source', 'org_id'], drop_first = True)



x = df_user_merged_dum.drop(['is_adopted_user'], axis = 1).values
y = df_user_merged_dum.pop('is_adopted_user').values




"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 0] = labelencoder_X_1.fit_transform(x[:,0])





onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x = x[: , 1:] # Take away one dummy variable to avoid multicolinearity


""""


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy




# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)


# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(x, y)

# check selected features - first 5 features are selected
feat_selector.support_

# check selected features - first 5 features are selected
feat_selector.support_weak_

# check ranking of features
ranked_vec = np.array(feat_selector.ranking_)

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(x)

selected_feature_vec = 

selected_feature_vec = []
for i,j in enumerate(ranked_vec):
    if j == 1:
        selected_feature_vec.append(i)
    else:
        continue

important_feature_list = []
for idx, col in enumerate(list(df_user_merged_dum.columns)):
    if idx in selected_feature_vec:
        print(col)
        important_feature_list.append(col)
    else:
        continue
    
    
    
    
### The most important features were the creation source_sign_up column and org_id 1, 2, 6, 11. It is important to note that
## all of the columns were the result of one-hot encoding variables, which shows the importance of dealing with variables 
# in the correct data format. 
        
## The importance of the variables was determined using the Boruta package, which rates the importance of variables in a 
## classification setting. This algorithm uses variable importance measure to perform a downward search for relevenet variables by
## comparing the original attributes imporance with the importance derived from random, permuted copies and disposing of unimportant variables.
    








