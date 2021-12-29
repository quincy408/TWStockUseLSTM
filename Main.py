# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:49:10 2021

@author: quincy408
"""
from __future__ import unicode_literals
import requests
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
import configparser
import ssl
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
os.environ['PATH'] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

import shioaji as sj
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
app = Flask(__name__)

config = configparser.ConfigParser()
config.read('config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))
def LineNotifyPush(msg, image_path, Token):
    headers = {
        "Authorization":"Bearer " + Token
        }
    payload = {'message':msg}
    files = {'imageFile':open(image_path,'rb')}
    r = requests.post('https://notify-api.line.me/api/notify',
                headers = headers, params = payload, files = files)
    if r.status_code == 200:
        print('推播完成')
    else:
        print(r.status_code)
        
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        print(body, signature)
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def ProcessMessage(event):
    Message = event.message.text
    UID = event.source.user_id
    profile_data = {'Authorization': 'Bearer ' + config.get('line-bot', 'channel_access_token')}
    profile = requests.get('https://api.line.me/v2/bot/profile/'+ UID, headers=profile_data)
    profile_json = profile.json()
    NAME = profile_json['displayName']
    api = sj.Shioaji()

    UID = "" # 永豐帳號
    PWD = "" # 永豐密碼
    api.login(UID, PWD)
    
    StockID = Message
    
    EndDate = (datetime.datetime.now() + datetime.timedelta(days=0)).strftime("%Y-%m-%d")
    StartDate = (datetime.datetime.now() + datetime.timedelta(days=-1000)).strftime("%Y-%m-%d")
    StockContract = api.Contracts.Stocks[StockID]
                        
    kbars = api.kbars(StockContract, start = StartDate, end = EndDate)
    
    kbarsDF = pd.DataFrame({**kbars})
    kbarsDF.ts = pd.to_datetime(kbarsDF.ts)
    kbarsDF['ts'] = kbarsDF['ts'].dt.date
    kbarsDF.rename(columns={'ts': 'Date'}, inplace=True)
    kbarsDF = kbarsDF.groupby('Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last','Volume':'sum'})
    kbarsDF.reset_index(inplace=True, drop=False)
    kbarsDF.index = kbarsDF['Date']
    
    data = kbarsDF.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
    
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    
    dataset = new_data.values
    
    train = dataset[0:400,:]
    valid = dataset[238:,:]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    
    
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    train = new_data[:400]
    valid = new_data[238:]
    valid['Predictions'] = closing_price
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.title(str(kbarsDF['Date'][-1] + datetime.timedelta(days=+1)) +\
              '收盤價:' + str(closing_price[-1]))
    plt.savefig(Message + ".png")
    if closing_price[-1] > kbarsDF['Close'][-1]:
        msg = '明天可能會漲喔'
    elif closing_price[-1] < kbarsDF['Close'][-1]:
        msg = '明天可能會跌喔'
    else:
        msg = '明天可能不太會有變動'
    LineNotifyPush(msg, Message + '.png',\
                   "") # line notify 的 token
    
if __name__ == "__main__":
    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    context.load_verify_locations("./ca_bundle.crt")
    context.load_cert_chain("./certificate.crt", "./private.key")
    app.run(ssl_context=context, port=25555,host='0.0.0.0')
