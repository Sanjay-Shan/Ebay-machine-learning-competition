#import Python modules
import numpy as np
import pandas as pd
import pgeocode
from pytz import timezone
from timezonefinder import TimezoneFinder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class Date_precition:
    def Predict_Delivery_Date(self,obj_dataset,obj_predicted_dataset):
        try:
            self.obj_dataset = obj_dataset
            self.obj_predicted_dataset = obj_predicted_dataset
            
            arr_PredictedDate = []

            for i in range(obj_predicted_dataset.shape[0]):
                try:
                    str_payment_date = obj_dataset["Updated_payment_date"][i]
                    str_predicted_days = obj_predicted_dataset["output"][i]
                except Exception as e:
                    print("error " + str(e) + " index is --> " + str(i))
                    continue
                i_p_year, i_p_month, i_p_date = int(str_payment_date[0:4]),int(str_payment_date[5:7]),int(str_payment_date[8:10])
                i_p_hour,i_p_min,i_p_sec = int(str_payment_date[11:13]),int(str_payment_date[14:16]),int(str_payment_date[17:19])
                calc_payment_datetime = datetime.datetime(year=i_p_year, month=i_p_month, day=i_p_date, hour=i_p_hour, minute=i_p_min, second=i_p_sec)

                date_1 = datetime.datetime.strptime(str(calc_payment_datetime),'%Y-%m-%d %H:%M:%S')
                calc_date = date_1 + datetime.timedelta(days=int(str_predicted_days))
                calc_date = calc_date.strftime('%Y-%m-%d %H:%M:%S')

                arr_PredictedDate.append(calc_date)

            return arr_PredictedDate


        except Exception as e:
            print("error in Predict_Delivery_Date--> " + str(e))
        

    def Predict_Accuracy_Score(self,obj_dataset):
        try:
            counter_correct_val = 0
            arr_Actual_Date = obj_dataset["Actual_Date"]
            arr_Calculate_Date = obj_dataset["Calculated_Date"]

            for i in range(arr_Actual_Date.shape[0]):
                str_actualDate = arr_Actual_Date[i]
                str_calculatedDate = arr_Calculate_Date[i]
                
                i_a_year, i_a_month, i_a_date = int(str_actualDate[0:4]),int(str_actualDate[5:7]),int(str_actualDate[8:10])
                i_p_year, i_p_month, i_p_date = int(str_calculatedDate[0:4]),int(str_calculatedDate[5:7]),int(str_calculatedDate[8:10])
                
                formatdate = datetime.datetime(year=i_a_year, month=i_a_month, day=i_a_date)
                format_calculatedate = datetime.datetime(year=i_p_year, month=i_p_month, day=i_p_date)

                date_difference = abs((formatdate - format_calculatedate).days)

                if date_difference in [0]:
                    counter_correct_val = counter_correct_val + 1
                # else:
                #     print("The difference between the dates is more than the given set range.")

            return(counter_correct_val/arr_Calculate_Date.shape[0])


        except Exception as e:
            print("error in Predict_Accuracy_Score--> " + str(e))

    def convert_data_type(self,val):
        
        try:
            if not val:
                return ''
            try:
                return str(val)
            except:
                return ''
        except Exception as e:
            print("error" + e)


if __name__ == "__main__":

    try:
        obj_class = Date_precition()
        
        #mantain a record of the errors obtained
        f=open("record.txt","a")

        obj_dataset_train = pd.read_csv("final_dataset/Record_1Dec2021_train .csv")
        obj_dataset_test = pd.read_csv("final_dataset/Record_1Dec2021_test.csv")

        obj_dataset_train=obj_dataset_train.drop(['quantity','std_weight','Updated_accepatnce_date', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp','seller_id', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate','record_number','weight_units', 'item_price','shipping_fee','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)
        obj_dataset_test=obj_dataset_test.drop(['quantity','std_weight','Updated_accepatnce_date', 'Timezone_PaymentDatetime', 'Timezone_AcceptanceTimestamp','seller_id', 'Timezone_DeliveryDate', 'split','UpdatedZipCode_User','carrier_max_estimate','carrier_min_estimate','record_number','weight_units', 'item_price','shipping_fee','declared_handling_days','acceptance_scan_timestamp','weight','payment_datetime','Item_latitude', 'Item_longitued', 'User_latitude', 'User_longitued', 'item_zip', 'buyer_zip','UpdatedZipCode_Item', 'UpdatedZipCode_User','delivery_date','final_weight','split'], axis=1)

        nan_val = float("NaN")

        if(pd.isna(obj_dataset_train).all):
            obj_dataset_train.replace("",nan_val,inplace=True)
            obj_dataset_train.dropna( inplace=True,subset=["b2c_c2c","category_id","shipment_method_id","package_size","Distance","Calc_HandlingDays","Calc_CarrierDays"])
            obj_dataset_train.reset_index(drop=True,inplace=True)
            print ("Nan value is present in train dataset")

        if(pd.isna(obj_dataset_test).all):
            obj_dataset_test.replace("",nan_val,inplace=True)
            obj_dataset_test.dropna(axis=0,inplace=True,subset=["b2c_c2c","category_id","shipment_method_id","package_size","Distance","Calc_HandlingDays","Calc_CarrierDays"])
            obj_dataset_test.reset_index(drop=True,inplace=True)
            print ("Nan value is present in test dataset")            

        Features = ["b2c_c2c","shipment_method_id","package_size","Distance","Calc_HandlingDays","Calc_CarrierDays","category_id"]

        Y_dataset_train = obj_dataset_train["Calc_deliveryDays"]
        X_dataset_train = obj_dataset_train[Features]

        Y_dataset_test = obj_dataset_test["Calc_deliveryDays"]
        X_dataset_test = obj_dataset_test[Features]

        #implementing grid search for various parameters of random forest
        parameters={}
        model = make_pipeline(StandardScaler(),RandomForestRegressor(n_estimators=10,max_depth=100,random_state=10)) #pipeline to make line regression model 
        print("started training.....")
        model.fit(X_dataset_train,Y_dataset_train)
        y_predicted = model.predict(X_dataset_test)


        dataframe_value = pd.DataFrame(y_predicted,columns=['output'])

        dataframe_value["Actual_Date"] = obj_dataset_test["Updated_delivery_date"]

        dataframe_value["Calculated_Date"] = obj_class.Predict_Delivery_Date(obj_dataset_test,dataframe_value)


        dataframe_value["Calculated_Date"] = pd.to_datetime(dataframe_value.Calculated_Date, format = '%Y-%m-%d %H:%M:%S')
        dataframe_value["Calculated_Date"] = dataframe_value["Calculated_Date"].dt.strftime('%Y-%m-%d %H:%M:%S')
        dataframe_value.to_csv('Random_forest.csv',index=False)

        error=mean_squared_error(Y_dataset_test, y_predicted, squared=False)

        Accuracy_Score = obj_class.Predict_Accuracy_Score(dataframe_value)

        print("Accuracy Score: " + str(Accuracy_Score))

        f.write("Random_forest - RMSE obtained ="+str(error))
        f.write("\t"+"Top 1 Accuracy="+str(Accuracy_Score))
        f.write("\n")

    except Exception as e:
        print("error --> " + str(e))

