import pandas as pd
import numpy as np 
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from Date_Prediction import Predict_Date

#regressors required for fitting the voting regressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import catboost as cb

#finally the voting regressor
from sklearn.ensemble import VotingRegressor 

#read the train and test csv files provided
df1 = pd.read_csv("final_dataset/Record_1Dec2021_train .csv")
df2 = pd.read_csv("final_dataset/Record_1Dec2021_test.csv")


#dropping irrelevant features
df1=df1.drop(['quantity','std_weight','Updated_accepatnce_date','category_id', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate','record_number','weight_units', 'item_price','shipping_fee','seller_id','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)
df2=df2.drop(['quantity','std_weight','Updated_accepatnce_date','category_id', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate','record_number','weight_units', 'item_price','shipping_fee','seller_id','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)

#to check if there are any Nans in the dataset
print(df1.isnull().sum()) 
print(df2.isnull().sum()) 

#remove null values if any
df1=df1.dropna(axis=0,how='any')
df2=df2.dropna(axis=0,how='any')

#get the gmt based payment datetime and final delivery date for final delivery date calculation
payment_dateTime=np.array(df2['Updated_payment_date'])
delivery_date=np.array(df2['Updated_delivery_date'])
df1=df1.drop(['Updated_payment_date','Updated_delivery_date'],axis=1)
df2=df2.drop(['Updated_payment_date','Updated_delivery_date'],axis=1)


cols1 = df1.columns.tolist()
cols2 = df2.columns.tolist()

print(cols1)

train_list = cols1[:-1] #seperate the target and the features for training
train_target = [cols1[-1]]


test_list = cols2[:-1] #seperate the target and the features for training
test_target = [cols2[-1]]

train_X=df1[train_list]
train_y= df1[train_target]
test_X= df2[test_list]
test_y= df2[test_target]

print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)

#mantain a record for to save the statistics of the model
f=open("record.txt","a")

#initialize the regressors
r1=LinearRegression()
r2=RandomForestRegressor(n_estimators=10,max_depth=100,random_state=10)
r3=AdaBoostRegressor(base_estimator=LinearRegression(),random_state=10)
r4=XGBRegressor(n_estimators=100,random_state=10)
r5=GradientBoostingRegressor(random_state=10)
r6=HistGradientBoostingRegressor(random_state=10)
r7=cb.CatBoostRegressor(loss_function='RMSE')

weights=np.array([0.1488,0.1492]) #,0.1404 ,0.1391,0.1411,0.1418,0.1394
#run the training without PCA
un_pipeline = make_pipeline(StandardScaler(), VotingRegressor(estimators=[('r1', r1),('r3',r3)],weights=weights)) #pipeline to make line regression model 
print("started training.....")
un_pipeline.fit(train_X, train_y.values.ravel())
print("going towards prediction...")
pred_test = un_pipeline.predict(test_X) #make a prediction of the delivery duration
error=mean_squared_error(test_y, pred_test,squared=False) #root mean square error
f.write("Linear Regression without PCA - RMSE obtained ="+str(error))

pred_delivery_date= Predict_Date(payment_dateTime,pred_test)

correct=0
wrong=0
#creating a dataframe for actual and predicted delivery date
for i in range(len(pred_delivery_date)):
	d_date=delivery_date[i][:-15] #removal of the timezone data from the datetime and calculation of accuracy for delivery date predicted
	d_date=d_date.strip() #remove white spaces before comparison
	d_date=datetime.datetime.strptime(d_date,'%Y-%m-%d')
	pred_delivery_date[i]=datetime.datetime.strptime(pred_delivery_date[i],'%Y-%m-%d')
	if abs((d_date - pred_delivery_date[i]).days)>0:   #considering absolute 0 difference ,the accuarcy is 70.22%
																#considering absolute 0 difference ,the accuarcy is 99.96%
		wrong+=1
	else:
		correct+=1

accuracy=correct/(correct+wrong)
print(accuracy)
f.write("\t"+"Accuracy="+str(accuracy))
f.write("\n")

		
data={"Actual_date":delivery_date,"predicted_date":pred_delivery_date}
compare=pd.DataFrame(data)

compare.to_csv("Linear_regression.csv",sep="\t") #saving the csv







