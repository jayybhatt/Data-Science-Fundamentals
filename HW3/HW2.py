
# coding: utf-8

# The Zillow challenge is about predicting the prices of real estate in 2017. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import gc

get_ipython().magic(u'matplotlib inline')


# In[2]:


prop_data = pd.read_csv("properties_2016.csv")
all_columns = prop_data.columns.values
print prop_data.shape
prop_data.head()


# In[3]:


train_df = pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
train_data = train_df
print train_data.shape
print train_data.head()


# ## Plotting the logerror

# In[4]:


logerror = train_data["logerror"]
hist, bins = np.histogram(logerror, bins=50)
center = (bins[:-1] + bins[1:]) / 2
width = 0.7 * (bins[1]-bins[0])
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.xlabel("index")
plt.ylabel("logerror")
plt.scatter(range(train_data.shape[0]), logerror)
plt.subplot(122)
plt.scatter(range(train_data.shape[0]), np.sort(logerror))
plt.show()
# plt.subplot(133)
ulimit = np.percentile(train_data.logerror.values, 99)
llimit = np.percentile(train_data.logerror.values, 1)
train_data['logerror'].loc[train_data['logerror']>ulimit] = ulimit
train_data['logerror'].loc[train_data['logerror']<llimit] = llimit

plt.figure(figsize=(12,8))
sb.distplot(train_data.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()

del logerror
del hist
del bins


# ## Plotting the missing value count

# In[5]:


missing_val = prop_data.isnull().sum().reset_index()
missing_val.columns = ['column_name', 'missing_count']
missing_val = missing_val.loc[missing_val['missing_count']>0]
missing_val = missing_val.sort_values(by='missing_count')

ind = np.arange(missing_val.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_val.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_val.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()

del ind


# ## Performing minimal cleaning

# In[6]:


missing_val['missing_ratio'] = missing_val["missing_count"]/prop_data.shape[0]
missing_val = missing_val.loc[missing_val["missing_ratio"]>0.995]
missing_val


# In[7]:


prop_data.drop(missing_val.column_name.values, axis=1, inplace=True)


# In[8]:


missing_val_rows = prop_data.isnull().sum(axis=1).reset_index()
missing_val_rows.columns = ["row_index", "null_count"]
missing_val_rows["missing_ratio"] = missing_val_rows["null_count"]/prop_data.shape[1]
missing_val_rows["missing_ratio"]


# In[9]:


# rows_to_delete = missing_val_rows.loc[missing_val_rows["missing_ratio"]>0.95]
# rows_to_delete


# In[10]:


# prop_data.drop(prop_data.index[rows_to_delete["row_index"]], inplace=True)
prop_data.shape


# In[11]:


del missing_val
del missing_val_rows
gc.collect()

plt.figure(figsize=(12,12))
sb.jointplot(x=prop_data.latitude.values, y=prop_data.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# ## Baseline Model

# In[12]:


train_data = pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
train_data = train_data.merge(prop_data, on='parcelid', how='left')

sample = pd.read_csv("./sample_submission.csv")
sample['parcelid'] = sample['ParcelId']
test_months = [k for k in sample.columns.values if k not in ["parcelid"]]

for c in train_data.columns:
    print c, train_data[c].dtype


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

train_data.fillna(-1, inplace = True)

non_float_cols = ["propertycountylandusecode", "taxdelinquencyflag", "hashottuborspa"]
cat_cols = ["airconditioningtypeid", "heatingorsystemtypeid", "propertylandusetypeid"]


X_train = train_data.drop(["parcelid", "logerror", "transactiondate", "propertyzoningdesc"]+non_float_cols, axis=1)
y_train = train_data["logerror"].values

X_test = sample.merge(prop_data, on='parcelid', how='left')
print X_test.shape

X_test = X_test[X_train.columns]
X_test.fillna(-1, inplace=True)
for c in X_test.dtypes[X_test.dtypes == object].index.values:
    X_test[c] = (X_test[c] == True)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

    
from datetime import datetime

for date in test_months:
    sample[date] = y_pred

sample["ParcelId"] = sample["parcelid"]
print sample.head()
sample.drop(["parcelid"], axis=1, inplace=True)

print( "\nWriting results to disk ..." )
sample.to_csv('baseline_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")

print("\nWorking on local data")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))


# In[13]:


missing_val = prop_data.isnull().sum().reset_index()
missing_val.columns = ['column_name', 'missing_count']
missing_val = missing_val.loc[missing_val['missing_count']>0]
missing_val = missing_val.sort_values(by='missing_count')
missing_val['missing_ratio'] = missing_val["missing_count"]/train_data.shape[0]
missing_val = missing_val.loc[missing_val["missing_ratio"]>0.80]
missing_val


# In[14]:


prop_data.drop(missing_val.column_name.values, axis=1, inplace=True)
prop_data.head()


# In[15]:


missing_val_rows = prop_data.isnull().sum(axis=1).reset_index()
missing_val_rows.columns = ["row_index", "null_count"]
missing_val_rows["missing_ratio"] = missing_val_rows["null_count"]/prop_data.shape[1]
missing_val_rows["missing_ratio"]


# In[16]:


print "Saving this modified data to prop_data_mod.csv"

prop_data_new = prop_data
prop_data_new.fillna(-1).to_csv("./prop_data_mod.csv")
prop_data.head()

del sample
del missing_val
del missing_val_rows
del prop_data_new
del train_data
gc.collect()


# In[17]:


dtype_df = prop_data.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df
# dtype_df.groupby("Column Type").aggregate("count").reset_index()


# ## Checking the value count for each feature

# In[18]:


for column in prop_data:
    print column, "\n",prop_data[column].value_counts(dropna=False), "\n\n\n********"


# ## FInding the discarded columns

# In[19]:


columns = [
# 'logerror', 
'transactiondate',
'airconditioningtypeid', 'architecturalstyletypeid',
'basementsqft', 'bathroomcnt',
'bedroomcnt', 'buildingqualitytypeid',
'buildingclasstypeid', 'calculatedbathnbr',
'decktypeid', 'threequarterbathnbr',
'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
'finishedsquarefeet6', 'finishedsquarefeet12',
'finishedsquarefeet13', 'finishedsquarefeet15',
'finishedsquarefeet50', 'fips',
'fireplacecnt', 'fireplaceflag',
'fullbathcnt', 'garagecarcnt',
'garagetotalsqft', 'hashottuborspa',
'heatingorsystemtypeid', 'latitude',
'longitude', 'lotsizesquarefeet',
'numberofstories', 'parcelid',
'poolcnt', 'poolsizesum',
'pooltypeid10', 'pooltypeid2',
'pooltypeid7', 'propertycountylandusecode',
'propertylandusetypeid', 'propertyzoningdesc',
'rawcensustractandblock', 'censustractandblock',
'regionidcounty', 'regionidcity',
'regionidzip', 'regionidneighborhood',
'roomcnt', 'storytypeid',
'typeconstructiontypeid', 'unitcnt',
'yardbuildingsqft17', 'yardbuildingsqft26',
'yearbuilt','taxvaluedollarcnt',
'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt',
'taxamount', 'assessmentyear', 
'taxdelinquencyflag', 'taxdelinquencyyear'
]

for c in columns:
    if c not in prop_data.columns.values:
        print c


# ## Finding the correlation among the features

# In[ ]:


# mean_values = prop_data.mean(axis=0)
prop_data_new = prop_data.fillna(-1)

x_cols = [col for col in prop_data_new.columns if col not in ['logerror'] if prop_data_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(prop_data_new[col].values, prop_data_new.logerror.values)[0,1])
corr_data = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_data = corr_data.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.5
fig, ax = plt.subplots(figsize=(12,25))
rects = ax.barh(ind, np.array(corr_data.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_data.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# In[ ]:


corr_data_sel = corr_data.loc[(corr_data['corr_values']>0.02) | (corr_data['corr_values'] < -0.01)]
corr_data_sel


# In[ ]:


cols_to_use = corr_data_sel.col_labels.tolist()

temp_df = prop_data[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sb.heatmap(corrmat, vmax=1., annot=True, square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[20]:


print "saving clean data to prop_data_clean.csv"
prop_data.to_csv("./prop_data_clean.csv")
prop_data.head()
del prop_data
gc.collect()


# In[ ]:


prop_data = pd.read_csv("./prop_data_clean.csv", index_col=0)
prop_data_new = prop_data.fillna(-1)

# Now let us look at the correlation coefficient of each of these variables #
cols_to_use = [col for col in prop_data_new.columns if col in ['airconditioningtypeid', 'heatingorsystemtypeid']]

# cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = prop_data[cols_to_use]
corrmat = temp_df.corr(method="spearman")
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sb.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[22]:


prop_data.shape


# In[ ]:


train_data = pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
train_data = train_data.merge(prop_data, on='parcelid', how='left')

sample = pd.read_csv("./sample_submission.csv")
sample['parcelid'] = sample['ParcelId']
test_months = [k for k in sample.columns.values if k not in ["parcelid"]]

for c in train_data.columns:
    print c, train_data[c].dtype


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

train_data.fillna(-1, inplace = True)

non_float_cols = ["propertycountylandusecode"]#,"propertyzoningdesc", "taxdelinquencyflag", "hashottuborspa"]
cat_cols = ["airconditioningtypeid", "heatingorsystemtypeid", "propertylandusetypeid"]


X_train = train_data.drop(["parcelid", "logerror", "transactiondate"]+non_float_cols, axis=1)
y_train = train_data["logerror"].values

X_test = sample.merge(prop_data, on='parcelid', how='left')
print X_test.shape

X_test = X_test[X_train.columns]
X_test.fillna(-1, inplace=True)
for c in X_test.dtypes[X_test.dtypes == object].index.values:
    X_test[c] = (X_test[c] == True)

# Create linear regression object
regr = linear_model.LinearRegression(normalize=True)

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

    
from datetime import datetime

for date in test_months:
    sample[date] = y_pred

sample["ParcelId"] = sample["parcelid"]
print sample.head()
sample.drop(["parcelid"], axis=1, inplace=True)

print( "\nWriting results to disk ..." )
sample.to_csv('baseline_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")

print("\nWorking on local data")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

regr = linear_model.LinearRegression(normalize=True)

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))


# In[ ]:


prop_data[["airconditioningtypeid", "heatingorsystemtypeid"]]


# ## Imputing values

# In[ ]:


#imputing airconditioningtypeid, making some NaN to 1.0 where heatingorsystemtypeid == 2

prop_data.loc[(prop_data["heatingorsystemtypeid"]==2.0) & (pd.isnull(prop_data["airconditioningtypeid"])), "airconditioningtypeid"] = 1.0

prop_data["airconditioningtypeid"].fillna(-1, inplace=True)
print prop_data["airconditioningtypeid"].value_counts()

prop_data[["airconditioningtypeid", "heatingorsystemtypeid"]].head()


# In[ ]:


duplicate_or_not_useful_cols = pd.Series(['calculatedbathnbr', 'assessmentyear', 'fullbathcnt', 
                    'regionidneighborhood', 'propertyzoningdesc', 'censustractandblock'])#,'finishedsquarefeet12'])
prop_data.drop(duplicate_or_not_useful_cols, axis=1, inplace=True)


# In[ ]:


prop_data["buildingqualitytypeid"].fillna(prop_data["buildingqualitytypeid"].mean(), inplace=True)
prop_data["calculatedfinishedsquarefeet"].interpolate(inplace=True)
prop_data["heatingorsystemtypeid"].fillna(-1, inplace=True)
prop_data["lotsizesquarefeet"].fillna(prop_data["lotsizesquarefeet"].median(), inplace=True)
prop_data.drop(["numberofstories"], axis=1, inplace=True)
#removing propertycountylandusecode because it is not in interpretable format
prop_data.drop(["propertycountylandusecode"], axis=1, inplace=True)
prop_data["regionidcity"].interpolate(inplace=True)
prop_data["regionidzip"].interpolate(inplace=True)
prop_data["yearbuilt"].fillna(prop_data["yearbuilt"].mean(), inplace=True)

#impute structuretaxvaluedollarcnt, taxvaluedollarcnt, landtaxvaluedollarcnt, taxamount by interpolation
cols_to_interpolate = ["structuretaxvaluedollarcnt", "taxvaluedollarcnt", "landtaxvaluedollarcnt", "taxamount"]
for c in cols_to_interpolate:
    prop_data[c].interpolate(inplace=True)


# In[ ]:


#imputing garagecarcnt on basis of propertylandusetypeid
#All the residential places have 1 or 2 garagecarcnt, hence using random filling for those values.

prop_data.loc[(prop_data["propertylandusetypeid"]==261) & (pd.isnull(prop_data["garagecarcnt"])), "garagecarcnt"] = np.random.randint(1,3)
prop_data.loc[(prop_data["propertylandusetypeid"]==266) & (pd.isnull(prop_data["garagecarcnt"])), "garagecarcnt"] = np.random.randint(1,3)

prop_data["garagecarcnt"].fillna(-1, inplace=True)

prop_data["garagecarcnt"].value_counts(dropna=False)


# In[ ]:


#imputing garagetotalsqft using the garagecarcnt

prop_data.loc[(prop_data["garagecarcnt"]==-1) & (pd.isnull(prop_data["garagetotalsqft"]) | (prop_data["garagetotalsqft"] == 0)), "garagetotalsqft"] = -1
prop_data.loc[(prop_data["garagecarcnt"]==1) & (pd.isnull(prop_data["garagetotalsqft"]) | (prop_data["garagetotalsqft"] == 0)), "garagetotalsqft"] = np.random.randint(180, 400)
prop_data.loc[(prop_data["garagecarcnt"]==2) & (pd.isnull(prop_data["garagetotalsqft"]) | (prop_data["garagetotalsqft"] == 0)), "garagetotalsqft"] = np.random.randint(400, 720)
prop_data.loc[(prop_data["garagecarcnt"]==3) & (pd.isnull(prop_data["garagetotalsqft"]) | (prop_data["garagetotalsqft"] == 0)), "garagetotalsqft"] = np.random.randint(720, 880)
prop_data.loc[(prop_data["garagecarcnt"]==4) & (pd.isnull(prop_data["garagetotalsqft"]) | (prop_data["garagetotalsqft"] == 0)), "garagetotalsqft"] = np.random.randint(880, 1200)

prop_data["garagetotalsqft"].interpolate(inplace=True)

prop_data["garagetotalsqft"].value_counts(dropna=False)


# In[ ]:


cols_to_use = [col for col in prop_data_new.columns if col in ['calculatedfinishedsquarefeet', 'bedroomcnt', 'lotsizesquarefeet']]

# cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = prop_data[cols_to_use]
corrmat = temp_df.corr(method="spearman")
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sb.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[ ]:


for column in prop_data:
    print column, "\n",prop_data[column].value_counts(dropna=False), "\n\n\n********"


# In[ ]:


#imputing unitcnt using propertylandusetypeid

prop_data.loc[(prop_data["garagecarcnt"]==4) & (pd.isnull(prop_data["unitcnt"])), "unitcnt"] = np.random.randint(880, 1200)
prop_data["unitcnt"].fillna(1, inplace=True)
prop_data["unitcnt"].value_counts(dropna=False)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


prop_data.fillna(-1, inplace=True)
prop_data.to_csv("./prop_data_clean_2.csv")

del prop_data
gc.collect()


# ## Creating New features

# In[ ]:


print "creating new features"
prop_data = pd.read_csv("./prop_data_clean_2.csv", index_col=0)

prop_data['age'] = 2017 - prop_data['yearbuilt']

#proportion of living area
prop_data['N-LivingAreaProp'] = prop_data['calculatedfinishedsquarefeet']/prop_data['lotsizesquarefeet']

#Ratio of the built structure value to land area
prop_data['N-ValueProp'] = prop_data['structuretaxvaluedollarcnt']/prop_data['landtaxvaluedollarcnt']

#Ratio of tax of property over parcel
prop_data['N-ValueRatio'] = prop_data['taxvaluedollarcnt']/prop_data['taxamount']


#Average structuretaxvaluedollarcnt by city
group = prop_data.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
prop_data['N-Avg-structuretaxvaluedollarcnt'] = prop_data['regionidcity'].map(group)

#Deviation away from average
prop_data['N-Dev-structuretaxvaluedollarcnt'] = abs((prop_data['structuretaxvaluedollarcnt'] - 
                        prop_data['N-Avg-structuretaxvaluedollarcnt']))/prop_data['N-Avg-structuretaxvaluedollarcnt']

#Number of properties in the zip
zip_count = prop_data['regionidzip'].value_counts().to_dict()
prop_data['N-zip_count'] = prop_data['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = prop_data['regionidcity'].value_counts().to_dict()
prop_data['N-city_count'] = prop_data['regionidcity'].map(city_count)

#Number of properties in the city
region_count = prop_data['regionidcounty'].value_counts().to_dict()
prop_data['N-county_count'] = prop_data['regionidcounty'].map(region_count)

prop_data.to_csv("./prop_data_with_new_features.csv")


# ## Plotting Some interesting bar chart

# In[ ]:


plt.figure(figsize=(22,8))
sb.countplot(x="regionidcounty", data=prop_data)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Number of real estate in counties", fontsize=15)
plt.show()


# In[ ]:



## Merging the train and prop data
print "Merging the train and prop data" 

train_data = pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])

train_data = pd.merge(train_data, prop_data, on='parcelid', how='left')
print "training data", train_data.head()


# In[ ]:


plt.figure(figsize=(30,26))
sb.set_context(font_scale=2.5)
corr_fig = sb.heatmap(prop_data.corr("pearson"), cmap="YlGnBu", annot=True, fmt=".3f", linewidths=1.0)
corr_fig = corr_fig.get_figure()
corr_fig.savefig("./correlation_plot_1.png")


# ### Analysing the above Heatmap
# 
# Here we can observe that the tax related variables are highly correlated, which means it is redundant to use all of these for further analysis
# Since, 'taxamount' is described as 'The total property tax assessed for that assessment year' it seem like a good option to keep and to discard the other tax variables.
# 
# There 3 more pairs of variables that are highly correlated and do not contribute to adding more information.
# These pairs are :
# 1. fips / rawcensustractandblock
# 2. structuretaxvaluedollarcnt/ landtaxvaluedollarcnt/ taxvaluedollarcnt
# 3. garagecarcnt / garagetotalsqft
# 
# After carefully analysing the data
# I decided to drop, rawcensustractandblock, structuretaxvaluedollarcnt, landtaxvaluedollarcnt, garagecarcnt

# In[ ]:


train_data.drop(["rawcensustractandblock", "structuretaxvaluedollarcnt", "landtaxvaluedollarcnt", "taxvaluedollarcnt", "garagecarcnt"],axis=1, inplace=True)#, "calculatedfinishedsquarefeet"], axis=1, inplace=True)


# ### Lets plot the correlation again

# In[ ]:


plt.figure(figsize=(30,26))
sb.set_context(font_scale=2.5)
corr_fig = sb.heatmap(train_data.corr("pearson"), cmap="YlGnBu", square=True, annot=True, fmt=".3f", linewidths=1.0)
corr_fig = corr_fig.get_figure()
corr_fig.savefig("./correlation_plot_2.png")


# ## Linear regression on the modified data

# In[ ]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split

sample = pd.read_csv("./sample_submission.csv")
sample['parcelid'] = sample['ParcelId']
test_months = [k for k in sample.columns.values if k not in ["parcelid", "ParcelId"]]
train_data.fillna(train_data.mean(), inplace=True)

cat_cols = []
# cat_cols = ["airconditioningtypeid", "heatingorsystemtypeid", "propertylandusetypeid"]

X_train = train_data.drop(["parcelid", "logerror", "transactiondate"]+cat_cols, axis=1)
y_train = train_data["logerror"].values
X_test = pd.merge(sample, prop_data, on='parcelid', how='left')
X_test = X_test[X_train.columns]
print "X_test shape"
print X_test.shape

# Create linear regression object
regr = linear_model.LinearRegression(normalize=True)
# Train the model using the training sets
regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)
    
from datetime import datetime

for date in test_months:
    sample[date] = y_pred

sample["ParcelId"] = sample["parcelid"]
sample.drop(["parcelid"], axis=1, inplace=True)

print( "\nWriting results to disk ..." )
sample.to_csv('LR_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")


X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# print y_pred.shape

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))

del train_data
del sample
del X_train
del y_train
del X_test
gc.collect()



# ## Let's see the feature importance

# In[ ]:


import xgboost as xgb

n_features = 20

# for c in prop_data.columns:
#     if prop_data[c].dtype == 'object':
#         lbl = LabelEncoder()
#         lbl.fit(list(prop_data[c].values))
#         prop_data_new[c] = lbl.transform(list(prop_data[c].values))

train_data = pd.read_csv("./train_2016_v2.csv")
sample = pd.read_csv("./sample_submission.csv")
sample['parcelid'] = sample['ParcelId']

train_data = train_data.merge(prop_data, on='parcelid', how='left')

y_train = train_data['logerror'].values
X_train = train_data.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
feat_names = X_train.columns.values
X_test = pd.merge(sample, prop_data, on='parcelid', how='left')
X_test = X_test[X_train.columns]

print "X train cols\n", X_train.columns.values
print "X_test shape"
print X_test.shape


xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=150)

thresholds = model.get_fscore()
L = [(k,v) for (k,v) in thresholds.iteritems()]
thresholds = sorted(L, key=lambda x: x[1] , reverse=True)[:n_features]
thresholds = dict(thresholds)

X_train_new = X_train.drop([k for k in feat_names if k not in thresholds], axis=1)
# X_train, X_test, y_train, y_test = train_test_split( X_train, train_y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train_new, y_train, feature_names=X_train_new.columns.values)
X_test = xgb.DMatrix(X_test[X_train_new.columns])
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=150)
    
y_pred = model.predict(X_test)

for date in test_months:
    sample[date] = y_pred

sample["ParcelId"] = sample["parcelid"]
sample.drop(["parcelid"], axis=1, inplace=True)

print( "\nWriting results to disk ..." )
sample.to_csv('XGB_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()
                

X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2, random_state=42)

print "X train cols\n", X_train.columns.values
print "X_test shape"
print X_test.shape

dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=150)

thresholds = model.get_fscore()
L = [(k,v) for (k,v) in thresholds.iteritems()]
thresholds = sorted(L, key=lambda x: x[1] , reverse=True)[:n_features]
thresholds = dict(thresholds)

X_train_new = X_train.drop([k for k in feat_names if k not in thresholds], axis=1)

dtrain = xgb.DMatrix(X_train_new, y_train, feature_names=X_train_new.columns.values)
X_test = xgb.DMatrix(X_test)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=150)
    
y_pred = model.predict(X_test)


print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %f" % mean_squared_error(y_test, y_pred))


# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()


# In[ ]:


not_to_drop = ['parcelid', 'logerror', 'transactiondate']
X_train = train_data.drop([k for k in train_data.columns.values if k not in thresholds.keys() + not_to_drop], axis=1)
print X_train.columns.values
X_train.to_csv("./important_features.csv", index=False)

del train_data
del sample
del X_train
del X_train_new
del y_train
del X_test
gc.collect()


# ## Ridge Regression

# In[ ]:


train_data = pd.read_csv("./important_features.csv")

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# cat_cols = ["airconditioningtypeid", "heatingorsystemtypeid", "propertylandusetypeid"]
y_train = train_data["logerror"].values
X_train = train_data.drop(["parcelid", "logerror", "transactiondate"], axis=1)

print "X train cols\n", X_train.columns.values

X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2, random_state=42)
# scorer = make_scorer(mean_sbsolute_error, greater_is_better=False)
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], normalize=True, gcv_mode='auto')
reg.fit( X_train, y_train)       
y_pred = reg.predict(X_test)


print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %f" % mean_squared_error(y_test, y_pred))

print reg.alpha_
print reg.coef_
print reg.intercept_


# ## Lasso Regression

# In[ ]:


X_train = train_data.drop(["parcelid", "logerror", "transactiondate"], axis=1)

X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2, random_state=42)

print "X train cols\n", X_train.columns.values

# scorer = make_scorer(mean_sbsolute_error, greater_is_better=False)
reg = linear_model.LassoCV(normalize=True, precompute='auto', n_jobs=-1, random_state=7,
                           selection='random')
reg.fit( X_train, y_train)       

y_pred = reg.predict(X_test)

print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %f" % mean_squared_error(y_test, y_pred))


print reg.alpha_
print reg.coef_
print reg.intercept_


# ## MLPRegressor

# In[ ]:


from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_train = train_data.drop(["parcelid", "logerror", "transactiondate"], axis=1)

print "X train cols\n", X_train.columns.values

X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2, random_state=42)

sc = StandardScaler(with_mean=True, with_std=True)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components=5, svd_solver='auto')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print pca.explained_variance_

# reg = MLPRegressor(hidden_layer_sizes=(50, 50), solver='sgd', learning_rate='adaptive', tol=1e-6, max_iter=10000, warm_start=True, verbose=True, early_stopping=True, validation_fraction=0.3)
reg = MLPRegressor(hidden_layer_sizes=(160, 64, 28), tol=1e-6, max_iter=10000, warm_start=True, verbose=True)#, early_stopping=True, validation_fraction=0.3)

for i in range(100):
    reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %f" % mean_squared_error(y_test, y_pred))



# In[ ]:


from sklearn.preprocessing import LabelEncoder
import datetime as dt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


print( "\n\nProcessing data for Neural Network ...")
print('\nLoading train, prop and sample data...')
train_data = pd.read_csv("./train_2016_v2.csv", parse_dates=["transactiondate"])
# prop = pd.read_csv('../input/properties_2016.csv')

prop = prop_data#pd.read_csv("./train_data_clean_2.csv")
sample = pd.read_csv('./sample_submission.csv')

print('Fitting Label Encoder on properties...')
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

print('Creating training set...')
df_train = train_data.merge(prop, how='left', on='parcelid')

df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
df_train["transactiondate"] = df_train["transactiondate"].dt.day

# print('Creating x_train and y_train from df_train...' )
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = df_train["logerror"]

train_columns = x_train.columns
print train_columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

print('Creating df_test...')
sample['parcelid'] = sample['ParcelId']
test_months = [k for k in sample.columns.values if k not in ["parcelid"]]

print("Merging Sample with property data...")
df_test = sample.merge(prop, on='parcelid', how='left')

# ## Preprocessing
print("\nPreprocessing neural network data...")
# imputer= Imputer()
# imputer.fit(x_train.iloc[:, :])
# x_train = imputer.transform(x_train.iloc[:, :])
# imputer.fit(x_test.iloc[:, :])
# x_test = imputer.transform(x_test.iloc[:, :])

# x_test.fillna(x_test.mean(), inplace=True)

df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version
df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
df_test["transactiondate"] = df_test["transactiondate"].dt.day     
x_test = df_test[train_columns]

print('Shape of x_test:', x_test.shape)
print("Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)

len_x=int(x_train.shape[1])
print("len_x is:",len_x)


# Neural Network
print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.63))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.45))
nn.add(Dense(units = 28, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

print("\nFitting neural network model...")
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']
nn_pred = {}

for i in range(len(test_columns)):

    df_test["transactiondate"] = pd.to_datetime(test_dates[i])
    df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
    df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
    df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
    df_test["transactiondate"] = df_test["transactiondate"].dt.day     
    x_test = df_test[train_columns]

    print('Shape of x_test:', x_test.shape)
    print("Preparing x_test...")
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)

    x_test = sc.transform(x_test)
    print("\nPredicting with neural network model...")
    #print("x_test.shape:",x_test.shape)
    y_pred_ann = nn.predict(x_test)

    print( "\nPreparing results for write..." )
    nn_pred[test_columns[i]] = y_pred_ann.flatten()
    
    print( "Type of nn_pred is ", type(nn_pred[test_columns[i]]) )
    print( "Shape of nn_pred is ", nn_pred[test_columns[i]].shape )

    print( "\nNeural Network predictions:" )
    print( pd.DataFrame(nn_pred[test_columns[i]]).head() )

from datetime import datetime

for date in test_columns:
    sample[date] = nn_pred[date]

sample["ParcelId"] = sample["parcelid"]
sample.drop(["parcelid"], axis=1, inplace=True)

print( "\nWriting results to disk ..." )
sample.to_csv('NN_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")


x_train, x_test, y_train, y_test = train_test_split( x_train, y_train, test_size=0.3, random_state=42)

# Neural Network
print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.63))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.45))
nn.add(Dense(units = 28, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

print("\nFitting neural network model...")
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)


print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
y_pred_ann = nn.predict(x_test)

print( "\nPreparing results for write..." )
nn_pred = y_pred_ann.flatten()
print( "Type of nn_pred is ", type(nn_pred) )
print( "Shape of nn_pred is ", nn_pred.shape )

print( "\nNeural Network predictions:" )
print( pd.DataFrame(nn_pred).head() )

print("Mean absolute error: % f" % mean_absolute_error(y_test, nn_pred))
print("Mean squared error: %f" % mean_squared_error(y_test, nn_pred))


# Cleanup
del train_data
del prop
# del sample
# del x_train
# del x_test
del df_train
# del df_test
del y_pred_ann
gc.collect()


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split( x_train, y_train, test_size=0.2, random_state=7)

neigh = KNeighborsRegressor(n_neighbors=5, weights='uniform', n_jobs=-1)
neigh.fit(x_train, y_train) 

y_pred = neigh.predict(x_test)

print("Mean absolute error: % f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %f" % mean_squared_error(y_test, y_pred))
# from datetime import datetime

# for date in test_months:
#     sample[date] = nn_pred

# sample["ParcelId"] = sample["parcelid"]
# sample.drop(["parcelid"], axis=1, inplace=True)


# print( "\nWriting results to disk ..." )
# sample.to_csv('k_NN_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

# print( "\nFinished ...")

