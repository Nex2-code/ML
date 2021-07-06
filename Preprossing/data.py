import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data_set=pd.read_csv('Data.csv')
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,3].values
#label=LabelEncoder()
#label.fit(data_set['Country'])
#label.classes_
#label.transform(data_set['Country'])
#label.fit_transform(data_set['Country'])
lable=LabelEncoder()
x[:, 0]= lable.fit_transform(x[:, 0])
print(x)    

#%%
import numpy as np
from sklearn.preprocessing import StandardScaler
x1=np.array([[1,2,3],[3,4,5],[5,6,8]])
stand=StandardScaler()
scale=stand.fit_transform(x1)
print(scale)
stand.mean_
scale=stand.fit(x1)
scale.mean_
scale.scale_
scale.get_params
scale=stand.transform(x1)
print(scale)

#%%
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.preprocessing import Imputer
data_set=pd.read_csv('Datag.csv')
x= data_set.iloc[:,:-1].values
imp= Imputer(missing_values ='NaN', strategy='mean', axis = 0)
imp.fit(x[:, 1:3])
x[:, 1:3]= imp.transform(x[:, 1:3])
print(x)

#%%
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
data_set=pd.read_csv('Data.csv')
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,3].values
label=LabelEncoder()
onehot=OneHotEncoder(categorical_features=[0])
#x[:, 0]= label.fit_transform(x[:, 0])
#y=label.fit_transform(y)
x=onehot.fit_transform(x).toarray()
ct=ColumnTransformer([('onehot',OneHotEncoder(),[0])],remainder='passthrough')
#onehot=OneHotEncoder(categorical_featurers=[0])
#x=onehot.fit_transform(x)
x=nm.array(ct.fit_transform(x),dtype=nm.int)
x_tr,x_tst,y_tr,y_tst=train_test_split(x,y,test_size=0.2,random_state=0)
sc=StandardScaler();
x_tr=sc.fit_transform(x_tr)
x_tst=sc.transform(x_tst)
