import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import pgeocode
import datetime as obj_datetime 
import pytz
from datetime import datetime, timedelta
from pytz import timezone
from timezonefinder import TimezoneFinder
import math
from sklearn import preprocessing  
import random
import time



class Preprocess_dataset: 

    def convert_zipcode(self,arr_dataset):
        self.arr_dataset = arr_dataset
        for i in range(arr_dataset.shape[0]):
            str_val = str(arr_dataset[i])
            if len(str_val) > 5:
                arr_dataset[i] = str_val[0:5]
            elif len(str(arr_dataset[i])) < 5:
                print("The length of the zipcode is less than 5 therefore not truncating it. The index of the zip code is --> " + str(i))
        return arr_dataset

    def add_handling_days_cutoff(self,datetime):
        total_days = datetime.days
        hours = datetime.seconds//3600
        if hours > 14:
            return total_days + 1
        else:
            return total_days

    def std_weight_unit(self,weight,weight_unit):
        num_index = 0
        try:
            for i in range(len(weight_unit)):
                if weight_unit[i]==2:
                    weight[i]=weight[i]*2.20462 #convert weight to lbs  1kg=2.20462 lbs
                    weight_unit[i]=1            #make weight_unit=2 for lbs
                num_index = i
            return weight,weight_unit
        except Exception as e:
            print("Error in std_weight_unit. The index is --> " + str(num_index))
            print("Exception is: " + str(e))

    
    def calc_distance(self,arr_itemzip,arr_userzip):
        
        try:
            self.arr_itemzip = arr_itemzip
            self.arr_userzip = arr_userzip
            nomi = pgeocode.Nominatim('us')
            dist = pgeocode.GeoDistance('us')

            distance = dist.query_postal_code(arr_itemzip, arr_userzip)
            obj_item = nomi.query_postal_code(arr_itemzip)
            obj_user = nomi.query_postal_code(arr_userzip)

            return distance,obj_item,obj_user
        except Exception as e:
            print("Error in calc_distance.")
            print("Exception is: " + str(e))


    def calc_weight_average(self,arr_weight,arr_packagesize):
        self.arr_weight = arr_weight
        self.arr_packagesize = arr_packagesize
        final_weight = []
        try:
            packagesize_weights = {}
            for i,w in enumerate(arr_weight):
                package = arr_packagesize[i]
                if package not in packagesize_weights:
                    packagesize_weights[package] = [w]
                else:
                    packagesize_weights[package].append(w)
            
            packagesize_weights_mean = {}
            for package in packagesize_weights:
                item_weights = packagesize_weights[package]
                average_item_weights = np.mean(item_weights)
                packagesize_weights_mean[package] = average_item_weights

            average_weights = packagesize_weights_mean
            overall_average = np.mean(arr_weight)
            for i,w  in enumerate(arr_weight):
                if w==0:
                    package = arr_packagesize[i]
                    if package in average_weights:
                        arr_weight[i] = average_weights[package]
                    else:
                        arr_weight[i] = overall_average
            
                final_weight.append(arr_weight[i])
            
            return final_weight

        except Exception as e:
            print("Error in weight_average.")
            print("Exception is: " + str(e))   

    def handle_timezones_timestamps(self,arr_dataset_copy):
        
        try:
            self.arr_dataset_copy = arr_dataset_copy
            arr_dataset_copy["Calc_HandlingDays"] = '-99999'
            arr_dataset_copy["Calc_CarrierDays"] = '-99999'
            arr_dataset_copy["Calc_deliveryDays"] = '-99999'
            

            arr_dataset_copy["Updated_payment_date"] = '-99999'
            arr_dataset_copy["Updated_accepatnce_date"] = '-99999'
            arr_dataset_copy["Updated_delivery_date"] = '-99999'

            arr_dataset_copy["Timezone_PaymentDatetime"] = '-99999'
            arr_dataset_copy["Timezone_AcceptanceTimestamp"] = '-99999'
            arr_dataset_copy["Timezone_DeliveryDate"] = '-99999'

            arr_item_payment_datetime = np.array(arr_dataset_copy["payment_datetime"])   
            arr_item_accept_timestamp = np.array(arr_dataset_copy["acceptance_scan_timestamp"])
            arr_DeliveryDate = np.array(arr_dataset_copy["delivery_date"])
            arr_item_buyer_zip = np.array(arr_dataset_copy["UpdatedZipCode_User"])
            arr_item_zip = np.array(arr_dataset_copy["UpdatedZipCode_Item"])
            arr_item_carrier_min = np.array(arr_dataset_copy["carrier_min_estimate"])
            arr_item_carrier_max = np.array(arr_dataset_copy["carrier_max_estimate"])

            arr_HandlingDays_Item = []
            arr_TransferDays_Carrier = []
            arr_TotalDelivery_Days = []


            print("Started with loop") 
            for i in range(arr_dataset_copy.shape[0]):
                if i==1000:
                    break
                item_payment_datetime = arr_item_payment_datetime[i]    # the payment datetime for item.
                item_accept_datetime = arr_item_accept_timestamp[i]     # the payment datetime for item acceptance.
                delivery_date = arr_DeliveryDate[i]                     # The delivery date.

                item_delivery_time_zip = arr_item_buyer_zip[i]          #Buyer Zip value
                item_acceptace_zip = arr_item_zip[i]                    #Item Zip value
                item_carrier_min = arr_item_carrier_min[i]
                item_carrier_max = arr_item_carrier_max[i]

                #Check whether the zip is in correct format
                if not (str(item_delivery_time_zip).isnumeric()):
                    print("The given Buyer zip value is not in correct format for index --> " + str(i))
                    arr_dataset_copy = arr_dataset_copy.drop(i)
                    continue

                if not (str(item_acceptace_zip).isnumeric()):
                    print("The given Item zip value is not in correct format for index --> " + str(i))
                    arr_dataset_copy = arr_dataset_copy.drop(i)
                    continue
                                    

                #next three lines we are splitting the timestamp into year,month,day,hours, min, secs, timezone. We are doing it for payment_datetime column.
                i_p_year, i_p_month, i_p_date = int(item_payment_datetime[0:4]),int(item_payment_datetime[5:7]),int(item_payment_datetime[8:10])
                i_p_hour,i_p_min,i_p_sec = int(item_payment_datetime[11:13]),int(item_payment_datetime[14:16]),int(item_payment_datetime[17:19])
                i_p_timezone = item_payment_datetime[23:24] + item_payment_datetime[25:26]
                payment_datetime = obj_datetime.datetime(year=i_p_year, month=i_p_month, day=i_p_date, hour=i_p_hour, minute=i_p_min, second=i_p_sec)

                #next three lines we are splitting the timestamp into year,month,day,hours, min, secs, timezone. We are doing it for acceptance_scan_timestamp column.
                i_a_year, i_a_month, i_a_date = int(item_accept_datetime[0:4]),int(item_accept_datetime[5:7]),int(item_accept_datetime[8:10])
                i_a_hour,i_a_min,i_a_sec = int(item_accept_datetime[11:13]),int(item_accept_datetime[14:16]),int(item_accept_datetime[17:19])
                i_a_timezone = item_accept_datetime[23:24] + item_accept_datetime[25:26]
                acceptance_datetime = obj_datetime.datetime(year=i_a_year, month=i_a_month, day=i_a_date, hour=i_a_hour, minute=i_a_min, second=i_a_sec)

                i_d_year,i_d_month,i_d_date = int(delivery_date[0:4]), int(delivery_date[5:7]), int(delivery_date[8:10])
                delivery_datetime = obj_datetime.datetime(year=i_d_year, month=i_d_month, day=i_d_date,hour=0)



                if i_p_timezone == '-4':
                    str_timezone_payment = 'US/Eastern'
                elif i_p_timezone == '-5':
                    str_timezone_payment = 'US/Central'
                elif i_p_timezone == '-6':
                    str_timezone_payment = 'US/Mountain'
                elif i_p_timezone == '-7':
                    str_timezone_payment = 'US/Pacific'
                elif i_p_timezone == '-8':
                    str_timezone_payment = 'US/Alaska'
                elif i_p_timezone == '-9':
                    str_timezone_payment = 'US/Hawaii'

                if i_a_timezone == '-4':
                    str_timezone_acceptance = 'US/Eastern'
                elif i_a_timezone == '-5':
                    str_timezone_acceptance = 'US/Central'
                elif i_a_timezone == '-6':
                    str_timezone_acceptance = 'US/Mountain'
                elif i_a_timezone == '-7':
                    str_timezone_acceptance = 'US/Pacific'
                elif i_a_timezone == '-8':
                    str_timezone_acceptance = 'US/Alaska'
                elif i_a_timezone == '-9':
                    str_timezone_acceptance = 'US/Hawaii'

                
                nomi = pgeocode.Nominatim('us')
                obj_item = nomi.query_postal_code(str(item_delivery_time_zip))  # buyer zip to lat , log

                if(math.isnan(float(obj_item[-2])) or math.isnan(float(obj_item[-3]))):
                    print("The datarow is dropped at index --> " + str(i) + ". The buyer zip code is error.")
                    arr_dataset_copy = arr_dataset_copy.drop(i)
                    continue

                tf = TimezoneFinder()
                try:
                    str_timezone_delivery = tf.timezone_at(lng=obj_item[-2], lat=obj_item[-3])
                except:
                    print("The value of item_delivery_time_zip is --> " + str(item_delivery_time_zip))
                    print("The value of obj_item  is --> " + obj_item)

               


                current_Timezone_payment = timezone(str_timezone_payment)
                current_Timezone_acceptance = timezone(str_timezone_acceptance)
                current_Timezone_delivery = timezone(str_timezone_delivery)


                GMT_Timezone = timezone('GMT')

                local_datetime_payment = current_Timezone_payment.localize(datetime(i_p_year,i_p_month,i_p_date,i_p_hour,i_p_min,i_p_sec))
                local_datetime_acceptance = current_Timezone_acceptance.localize(datetime(i_a_year,i_a_month,i_a_date,i_a_hour,i_a_min,i_a_sec))

                updated_datetime_payment = local_datetime_payment.astimezone(GMT_Timezone)
                updated_datetime_acceptance = local_datetime_acceptance.astimezone(GMT_Timezone)

                nomi = pgeocode.Nominatim('us')   #Buyer Timezone calculation
                obj_item = nomi.query_postal_code(str(item_delivery_time_zip))

                if( math.isnan(float(obj_item[-2])) or math.isnan(float(obj_item[-3]))):
                    print("The datarow is dropped at index --> " + str(i) + ". Zip code is not given in correct format.")
                    arr_dataset_copy = arr_dataset_copy.drop(i)
                    continue


                tf = TimezoneFinder()
                zone = tf.timezone_at(lng=obj_item[-2], lat=obj_item[-3])
                local_buyer_timezone = timezone(zone)
                GMT = timezone('GMT')
                local_datetime = local_buyer_timezone.localize(datetime(i_d_year, i_d_month, i_d_date, 0, 0, 0))
                updated_datetime_delivery = local_datetime.astimezone(GMT)
                fmt = '%Y-%m-%d %H:%M:%S %Z%z'

                # to convert buyer time zone to  gmt for calculation purpose  -- End

                delivery_time = obj_datetime.datetime(i_d_year, i_d_month, i_d_date)
                payment_time = obj_datetime.datetime(i_p_year, i_p_month, i_p_date, i_p_hour, i_p_min, i_p_sec)
                acceptance_time = obj_datetime.datetime(i_a_year, i_a_month, i_a_date, i_a_hour, i_a_min, i_a_sec)

                delivery_acceptance_difference = delivery_time - acceptance_time
                delivery_payment_difference = delivery_time - payment_time
                acceptance_payment_difference = acceptance_time - payment_time
                


                delivery_acceptance_difference = updated_datetime_delivery - updated_datetime_acceptance
                delivery_payment_difference = updated_datetime_delivery - updated_datetime_payment
                acceptance_payment_difference = updated_datetime_acceptance - updated_datetime_payment

                # 0th index holds the value '-' if time A < time B

                delivery_acceptance_check = str(delivery_acceptance_difference)
                delivery_payment_check = str(delivery_payment_difference)
                acceptance_payment_check = str(acceptance_payment_difference)

                if delivery_acceptance_check[0] == '-' or delivery_payment_check[0] == '-' or acceptance_payment_check[0] == '-':
                    print("Dropping because of irregularity in payment datetime/delivery date/acceptance datetime --> "+str(i))
                    arr_dataset_copy = arr_dataset_copy.drop(i)

                if item_carrier_min < 0 or item_carrier_max < 0:
                    print(" Carrier days missing --> "+str(i))
                    arr_dataset_copy = arr_dataset_copy.drop(i)

                handling_days_item = updated_datetime_acceptance - updated_datetime_payment
                Total_handling_days = self.add_handling_days_cutoff(handling_days_item)
                arr_dataset_copy.at[i,'Calc_HandlingDays'] = str(Total_handling_days)
                arr_dataset_copy.at[i,'Calc_CarrierDays'] = str(delivery_acceptance_difference.days + 1) ##Adding a day becuase we need to count the day on which the parcel is getting delivered.
                arr_dataset_copy.at[i,'Calc_deliveryDays'] = str(delivery_payment_difference.days + 1) ##Adding a day becuase we need to count the day on which the parcel is getting delivered.

                arr_dataset_copy.at[i,"Updated_payment_date"] = updated_datetime_payment
                arr_dataset_copy.at[i,"Updated_accepatnce_date"] = updated_datetime_acceptance
                arr_dataset_copy.at[i,"Updated_delivery_date"] = updated_datetime_delivery

                arr_dataset_copy.at[i,"Timezone_PaymentDatetime"] = str_timezone_payment
                arr_dataset_copy.at[i,"Timezone_AcceptanceTimestamp"] = str_timezone_acceptance
                arr_dataset_copy.at[i,"Timezone_DeliveryDate"] = local_buyer_timezone

            return arr_dataset_copy

        except Exception as e:
            print("Error in handle_timezones_timestamps.")
            print("Exception is: " + str(e))
        
    

if __name__ == "__main__":


    try:
        numberofrecords = 15000000
       
        skip = 1500000
        nrow_val = 150000
        obj_dataset = pd.read_csv("train.tsv", sep='\t', skiprows=range(1,skip),nrows=nrow_val)
        
        #obj_dataset = pd.read_csv("train.tsv", sep='\t',nrows=50)
        print(obj_dataset.shape)

        obj_class = Preprocess_dataset()
        label_encoder = preprocessing.LabelEncoder()

        obj_dataset['b2c_c2c']= label_encoder.fit_transform(obj_dataset['b2c_c2c']) # Encode labels in column 'b2c_c2c'.
        obj_dataset["package_size"] = obj_dataset.package_size.map({'LETTER':1,'PACKAGE_THICK_ENVELOPE':2,'LARGE_PACKAGE':4,'LARGE_ENVELOPE':3,'EXTRA_LARGE_PACKAGE':5,'VERY_LARGE_PACKAGE':6,'NONE':0})

        weight= np.array(obj_dataset["weight"])
        arr_package_size = np.array(obj_dataset["package_size"])
        
        obj_dataset["final_weight"] = obj_class.calc_weight_average(weight,arr_package_size)

        weight_unit= np.array(obj_dataset["weight_units"])
        obj_dataset["std_weight"],obj_dataset["weight_units"]= obj_class.std_weight_unit(np.array(obj_dataset["final_weight"]),weight_unit) #convert all weights to one format i.e. lbs

        item_zip = np.array(obj_dataset["item_zip"])
        user_zip = np.array(obj_dataset["buyer_zip"])
        converted_zipcode_user = []
        converted_zipcode_item = []
        converted_zipcode_item = obj_class.convert_zipcode(item_zip)
        converted_zipcode_user = obj_class.convert_zipcode(user_zip)

        #the five digit Zipcode is present in the following columns.    
        obj_dataset["UpdatedZipCode_Item"] = converted_zipcode_item
        obj_dataset["UpdatedZipCode_User"] = converted_zipcode_user

        #The following line converts the updated zip code into string for user. The reason being it is giving an error in further processing.
        obj_dataset["UpdatedZipCode_User"] = obj_dataset["UpdatedZipCode_User"].astype(str)

        start=time.time()
        arr_distance,obj_item,obj_user = obj_class.calc_distance(np.array(obj_dataset["UpdatedZipCode_Item"]),np.array(obj_dataset["UpdatedZipCode_User"]))
        obj_dataset["Distance"] = arr_distance
        obj_dataset["Item_latitude"] = obj_item.latitude
        obj_dataset["Item_longitued"] = obj_item.longitude
        obj_dataset["User_latitude"] = obj_user.latitude
        obj_dataset["User_longitued"] = obj_user.longitude
        print("time taken to calculate distance :",time.time()-start)

        obj_dataset_preprocessed = obj_class.handle_timezones_timestamps(obj_dataset)
        
    # if (np.isnan(converted_zipcode_user).all()):
    #     print("their exists null value")
    # else:

        print(obj_dataset_preprocessed.shape)
        obj_dataset_preprocessed.to_csv("Record_22Nov2021_3AM.csv",index=False)

    except Exception as e:
        print("Error in __main__.")
        print("Exception is: " + str(e))