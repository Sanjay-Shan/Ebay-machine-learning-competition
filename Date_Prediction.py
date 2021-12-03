#import python modules
import pandas as pd
import numpy as np 
import datetime

#function to predict the date by adding it to the payment_date(which is the origin of the system)
def Predict_Date(arr_paymentdate,arr_predictedday):
	try:
		arr_paymentdate = arr_paymentdate
		arr_predictdate = arr_predictedday

		arr_PredictedDate = []

		for i in range(arr_predictedday.shape[0]):
			try:
				str_date = arr_paymentdate[i]
				str_day = arr_predictedday[i]
			except Exception as e:
				print("error " + str(e))
				continue
			i_p_year, i_p_month, i_p_date = int(str_date[0:4]),int(str_date[5:7]),int(str_date[8:10])
			i_p_hour,i_p_min,i_p_sec = int(str_date[11:13]),int(str_date[14:16]),int(str_date[17:19])
			payment_datetime = datetime.datetime(year=i_p_year, month=i_p_month, day=i_p_date , hour=i_p_hour, minute=i_p_min, second=i_p_sec)
			
			date_1 = datetime.datetime.strptime(str(payment_datetime),'%Y-%m-%d %H:%M:%S')
			
			calc_date = date_1 + datetime.timedelta(days=int(str_day))
			calc_date = calc_date.strftime('%Y-%m-%d') #%H:%M:%S')


			arr_PredictedDate.append(calc_date)

		return arr_PredictedDate
	except Exception as e:
		print("error --> " + str(e))

