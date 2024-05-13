from flask import Flask, request, jsonify
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import os
from flask_caching import Cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

app = Flask(__name__)

cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

@app.route('/predict', methods=['GET'])
def predict_stock():
    quote = request.args.get('quote')
    if not quote:
        return jsonify({'error': 'Stock quote symbol is required'}), 400

    # Fetch historical data
    get_historical(quote)

    # Read the CSV file
    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_path = f'{quote}_{current_date}.csv'
    df = pd.read_csv(csv_path)

    # Perform ARIMA analysis
    arima_pred, error_arima, image = ARIMA_ALGO(df, quote)
    accuracy = 100 - error_arima


    # Return prediction, error, and image
    return jsonify({
        'quote': quote,
        'arima_prediction': arima_pred,
        'arima_accuracy': accuracy,
        'arima_rmse': error_arima,
        'image': image
    })

# Function to fetch historical data
@cache.memoize(timeout=43200)
def get_historical(quote):
    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_path = f'{quote}_{current_date}.csv'
    print("file path", os.path.isfile(csv_path))
    if os.path.isfile(csv_path):
        # If CSV file exists, read it
        df = pd.read_csv(csv_path)
    else:
        print("wrong approach")
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        print("dataaaaaa: ",data)
        df = pd.DataFrame(data=data)
        df.to_csv(csv_path)
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(csv_path, index=False)
        return

# ARIMA analysis function
@cache.memoize(timeout=43200)
def ARIMA_ALGO(df, quote):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    #for daily basis
    def parser(x):
        return x

    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions

    # Prepare data for analysis
    Quantity_date = df[['Close', 'Date']]
    Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
    Quantity_date['Close'] = Quantity_date['Close'].map(lambda x: float(x))
    Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
    Quantity_date = Quantity_date.drop(['Date'], axis=1)

    # Plot trends and save the plot
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(Quantity_date)
    plt.savefig('static/Trends.png')
    plt.close(fig)

    # Prepare training and testing data for the ARIMA model
    quantity = Quantity_date.values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:len(quantity)]

    # Perform ARIMA modeling
    predictions = arima_model(train, test)

    # Plot actual vs. predicted prices and save the plot
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.legend(loc=4)
    image_path = f'static/{quote}-ARIMA.png'
    plt.savefig(image_path)
    plt.close(fig)

    print()
    print("##############################################################################")
    arima_pred = predictions[-2]
    print("Tomorrow's Closing Price Prediction by ARIMA:", arima_pred)
    # RMSE calculation
    error_arima = math.sqrt(mean_squared_error(test, predictions))
    print("ARIMA RMSE:", error_arima)
    print("##############################################################################")
    return arima_pred, error_arima, image_path

if __name__ == '__main__':
    app.run(debug=True)
