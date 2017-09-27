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

x_test.fillna(x_test.mean(), inplace=True)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

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


for date in test_months:


	df_test["transactiondate"] = pd.to_datetime(date)  # placeholder value for preliminary version
	df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
	df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
	df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
	df_test["transactiondate"] = df_test["transactiondate"].dt.day     
	x_test = df_test[train_columns]

	print('Shape of x_test:', x_test.shape)
	print("Preparing x_test...")
	for c in x_test.dtypes[x_test.dtypes == object].index.values:
	    x_test[c] = (x_test[c] == True)
	  
	print("\nPredicting with neural network model...")
	#print("x_test.shape:",x_test.shape)
	y_pred_ann = nn.predict(x_test)

	print( "\nPreparing results for write..." )
	nn_pred = y_pred_ann.flatten()
	print( "Type of nn_pred is ", type(nn_pred) )
	print( "Shape of nn_pred is ", nn_pred.shape )

	print( "\nNeural Network predictions:" )
	print( pd.DataFrame(nn_pred).head() )




# In[6]:


from datetime import datetime

for date in test_months:
    sample[date] = nn_pred

sample["ParcelId"] = sample["parcelid"]
sample.drop(["parcelid"], axis=1, inplace=True)

print( "\nWriting results to disk ..." )
sample.to_csv('NN_sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")

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

