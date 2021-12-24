import pandas as pd
import numpy as np 
import datetime
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from Date_Prediction import Predict_Date

df1 = pd.read_csv("final_dataset/Record_1Dec2021_train .csv")
df2 = pd.read_csv("final_dataset/Record_1Dec2021_test.csv")

#dropping irrelevant features
# print(df1.columns.tolist())
df1=df1.drop(['quantity','std_weight','Updated_accepatnce_date', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate','record_number','weight_units', 'item_price','shipping_fee','seller_id','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)
df2=df2.drop(['quantity','std_weight','Updated_accepatnce_date', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate','record_number','weight_units', 'item_price','shipping_fee','seller_id','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)

#to check if there are any Nans in the dataset
# print(df1.isnull().sum()) #84
# print(df2.isnull().sum()) #32

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


train_list = cols1[:-1] #seperate the target and the features for training
train_target = [cols1[-1]]


test_list = cols2[:-1] #seperate the target and the features for training
test_target = [cols2[-1]]

train_X=df1[train_list]
train_y= df1[train_target]
test_X= df2[test_list]
test_y= df2[test_target]


#mantain a record of the errors obtained
f=open("record.txt","a")

#run the training with different values of components of PCA
data={}

model = HistGradientBoostingRegressor(random_state=10) #66.81
model.fit(train_X,train_y.values.ravel())
pred_test = model.predict(test_X) #make a prediction of the delivery duration
error=mean_squared_error(test_y, pred_test,squared=False) #root mean square error
f.write("HistBoost - RMSE obtained="+str(error))
print("RMSE :",error)

# calculating the final delivery date
pred_delivery_date= Predict_Date(payment_dateTime,pred_test)

#top1  and top2 accuracy
correct1=0
correct2=0
total=0

#creating a dataframe for actual and predicted delivery date
for i in range(len(pred_delivery_date)):
	total+=1
	D_date=delivery_date[i][:-15] #removal of the timezone data from the datetime and calculation of accuracy for delivery date predicted
	D_date=D_date.strip() #remove white spaces before comparison
	D_date=datetime.datetime.strptime(D_date,'%Y-%m-%d')
	pred_delivery_date[i]=datetime.datetime.strptime(pred_delivery_date[i],'%Y-%m-%d')
	if abs((D_date - pred_delivery_date[i]).days)==0:   
		correct1+=1
	if abs((D_date - pred_delivery_date[i]).days)<=1:
		correct2+=1
	
print(correct1,correct2,total)
accuracy1=correct1/total
accuracy2=correct2/total
print(accuracy1,accuracy2)
f.write("\t"+"Top 1 Accuracy="+str(accuracy1))
f.write("\t"+"Top 2 Accuracy="+str(accuracy2))
f.write("\n")

accuracy={"correct1":0,"correct2":0}
total=0

data["HGBoost"]=pred_delivery_date

data["Actual_delivery_date"]=delivery_date


compare=pd.DataFrame(data)

compare.to_csv("HGBoost.csv",sep="\t") #saving the csv













