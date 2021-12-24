import pandas as pd
import numpy as np 
import datetime

df = pd.read_csv("final_dataset/Record_1Dec2021_train .csv")
# df=df.drop(['Updated_accepatnce_date', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate', 'category_id','record_number','weight_units', 'item_price','shipping_fee','seller_id','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)

#calculating the correlation( covariance / variance of each variable) to understand both magnitude and direction of linearity
# here we are calculating the pearson's correlation coefficient ,here the value lies between -1 ans 1 ,and the the sign indicates the covariance dirrection
pearson_corr=[]
spearman_corr=[]
for col in df.columns:
	if col!= 'Calc_deliveryDays':
		y=df[['Calc_deliveryDays',col]].corr()
		x=np.array(y)
		if x.shape[1]==1:
			continue
		pearson_corr.append((col,np.array(abs(df[['Calc_deliveryDays',col]].corr()))[0][1]))
pearson_corr.sort(key = lambda x: x[1]) #sort the columns as per the correlation value


p=[pearson_corr[i][1]  for i in range(len(pearson_corr))]
attr=[pearson_corr[i][0]  for i in range(len(pearson_corr))]
data = {'attr': attr,
	'pearson':p}
        
d=pd.DataFrame(data)

d.to_csv("linearity.csv",sep="\t") #saving the csv


