import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataframes from the csvs

df_users = pd.read_csv("takehome_users.csv", encoding='latin-1')

df_engagement = pd.read_csv("takehome_user_engagement.csv")

df_engagement.time_stamp = pd.to_datetime(df_engagement.time_stamp)


"""df_user_grouped = pd.DataFrame(df_user_engagement['visited'].groupby(df_user_engagement.user_id).sum())

df_user_grouped['user_id'] = df_user_grouped.index

df_user_engagement_merged = df_users.merge(df_user_grouped, left_on = 'object_id', right_on = 'user_id', how = 'inner'"""

def seven_day_count(grp, freq):
    return grp.rolling(freq, on='time_stamp')['user_id'].count() 
df_engagement['seven_day_visits'] = df_engagement.groupby('user_id', as_index=False, group_keys=False).apply(seven_day_count, '7D') 
df_engagement['adopted_user'] = 3 <= df_engagement['seven_day_visits']


grouped_user = df_engagement.adopted_user.groupby(df_engagement.user_id).sum()

grouped_user_df = pd.DataFrame({'user_id':grouped_user.index,'7_day_int': grouped_user, 'is_adopted_user': 1 <= grouped_user})



df_user_merged = df_users.merge(grouped_user_df, left_on = 'object_id', right_on = 'user_id', how = 'inner') 

df_user_merged = df_user_merged.drop(columns = ['object_id', 'invited_by_user_id','user_id', '7_day_int', 'name', 'email', 'creation_time', 'last_session_creation_time'])




x = df_user_merged.iloc[:,:-1].values
y = df_user_merged.iloc[:,-1].values




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 0] = labelencoder_X_1.fit_transform(x[:,0])





onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x = x[: , 1:] # Take away one dummy variable to avoid multicolinearity





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

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(x)





