import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yf
from keras.models import load_model
import streamlit as st
import datetime

st.set_page_config(page_title='Stock Trend Prediction')
st.title('Stock Trend Prediction')



user_input=st.text_input('Enter Stock Ticker','TSLA')
start = st.date_input('Enter start date (YYYY-MM-DD)', value = datetime.date(2022,12,23))
end = st.date_input('Enter end date (YYYY-MM-DD)')


yf.pdr_override()
df = web.get_data_yahoo(user_input, start, end)
df = df.reset_index()


save = st.checkbox('Save data into MySQL DB')
if save:
    
    dfd = df.to_dict(orient = 'list')  ##to convert dataframe into dictionary
    del dfd['Adj Close']
    del dfd['Volume']
   
    k = list(dfd.values())             ##converting valuses of each key into list-- to enter into separate columns w/ key as name
    def tolist(indx, k = list(dfd.values())):
        l = list(k[indx])
        lstnm = []
        for i in l:
            i = str(i)
            x = i[0:10]
            lstnm.append(x)
        return lstnm

    dfdt = tolist(0)
    opn = tolist(1)
    hgh = tolist(2)
    lw = tolist(3)
    cls = tolist(4)

    def rnd(lst):           
        nlst = []
        for i in lst:
            i = float(i)
            x = round(i,2)
            nlst.append(x)
        return nlst

    opn = rnd(opn)
    high = rnd(hgh)
    low = rnd(lw)
    close = rnd(cls)

    import mysql.connector                      ##establishing database connectivity and creating DB
    mydb=mysql.connector.connect(host='127.0.0.1', port = 3306, user='root',password='sqlx7260', auth_plugin='mysql_native_password')
    cursor= mydb.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS STOCK_INFO;")
    cursor.execute("USE STOCK_INFO;")

    ##creating table and entering values from dict
    table_nm=st.text_input('Enter table name')
    try:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_nm}(Date DATE , Open DECIMAL(6,2), High DECIMAL(6,2), Low DECIMAL(6,2), Close DECIMAL(6,2))")
        c=len(dfdt)
        i=0
        while i<c:
            for l in dfdt:
                l = l.replace('-','')
                k = int(l[:4]) 
                h = int(l[4:6])
                g = int(l[6:])
                n = datetime.date(k,h,g)
            
                cursor.execute(f"INSERT INTO {table_nm} (Date, Open, High, Low, Close) VALUES ('{n}', {opn[i]}, {high[i]}, {low[i]}, {close[i]})")
                i+=1
        mydb.commit()
    except:
        st.write('Enter valid table name: Table name cannot contain only numbers or have special characters')
    

    ##summary statistics
    try:
        disp = st.checkbox('Display summary statistics')
        if disp:
            cursor.execute(f'SELECT AVG(high) FROM {table_nm}')
            avh=cursor.fetchone()
            for i in avh:
                st.write(f'Average high value: {i}')

            cursor.execute(f'SELECT AVG(low) FROM {table_nm}')
            avl=cursor.fetchone()
            for i in avl:
                st.write(f'Average low value: {i}')

            cursor.execute(f'SELECT MAX(high) FROM {table_nm}')
            ath=cursor.fetchone()
            for i in ath:
                st.write(f'All time high value: {i}')

            cursor.execute(f'SELECT MIN(low) FROM {table_nm}')
            atl=cursor.fetchone()
            for i in atl:
                st.write(f'All time low value: {i}')
    except:
        st.write('Enter valid table name: Table name cannot contain only numbers or have special characters')
        


st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

st.subheader('Closing Prices vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r')
st.pyplot(fig)

st.subheader('Closing Prices vs Time with 100 days moving avg.')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g', label = '100 days MA')
plt.plot(df.Close, 'r', label = 'Closing prices')
st.pyplot(fig)

st.subheader('Closing Prices vs Time with 100 days and 200 days moving avg.')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g', label = '100 days MA')
plt.plot(ma200, 'b', label = '200 days MA')
plt.plot(df.Close, 'r', label = 'Closing prices')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
try:
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)
except:
    st.subheader('Error prompted:')
    st.write('enter at least 1 day for predictions')
    st.stop()

model = load_model('model_1.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test = np.array(x_test)
y_test = np.array(y_test)
y_predicted = []
try:
    y_predicted = model.predict(x_test)
except:
    st.subheader('Error prompted')
    st.write('Enter at least 100 day frame for value predictions')
    st.stop()
scaler=scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
    
st.subheader('Prediction vs Original Prices')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Stock trend')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
