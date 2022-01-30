import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from keras.models import load_model
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests
from PIL import Image

st.title('Stock Prediction')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets1.lottiefiles.com/packages/lf20_OOvaPt.json"
lottie_json = load_lottieurl(lottie_url)
# quality can be low medium high renderer = "svg"
st_lottie(lottie_json, speed = 0.5, reverse = False, loop = True, quality = "high", height = 200, width = 200, key = None)

""" For IBM """

options = st.multiselect(
    'Select the parameters for prediction',
    ('open', 'high', 'low', 'volume'))

st.write('You selected:', options)


if st.button('Train model'):
    with st.spinner("Training ongoing"):
        options.append('close')
        options.append('timestamp')
        main_df = pd.read_csv("daily_IBM.csv")
        main_df.head()

        df = pd.read_csv("daily_IBM.csv", parse_dates=['timestamp'])

        j = 0
        for i in df.columns:
            if(i in options):
                j+=1
            else:
                df.pop(i)

        st.write("Training snippet of data frame")
        st.write(df.describe())

        df['timestamp'] = pd.to_datetime(df.timestamp , format = '%Y/%m/%d')
        data = df.drop(['timestamp'], axis=1)
        data.index = df.timestamp

        from statsmodels.tsa.vector_ar.var_model import VAR
        #make final predictions
        model = VAR(endog=data)
        model_fit = model.fit()
        yhat_IBM = model_fit.forecast(model_fit.y, steps=1)
        from datetime import date
        today = date.today()
        # dd/mm/YY
        d1 = today.strftime("%Y/%m/%d")
        st.write(' Predicted Stock Value for IBM')
        st.write(yhat_IBM)
        st.write('For '+d1)


        st.subheader('Close vs Time Chart')
        fig = plt.figure(figsize = (12, 6))

        df_plot = main_df['close']
        indi = df.columns.get_loc("close")
        df_plot.loc[len(df.index)] = yhat_IBM[0][indi-1]
        closing_ibm = yhat_IBM[0][indi-1]


        # Reading it again
        df = pd.read_csv("daily_IBM.csv", parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df.timestamp , format = '%Y/%m/%d')
        data = df.drop(['timestamp'], axis=1)
        data.index = df.timestamp

        from statsmodels.tsa.vector_ar.var_model import VAR
        #make final predictions
        model = VAR(endog=data)
        model_fit = model.fit()
        yhat_IBM = model_fit.forecast(model_fit.y, steps=1)

        test_size = 1

        df_train = df_plot[:-test_size]
        df_test = df_plot[-test_size:]

        plt.title('Stock data train and test sets', size=20)

        plt.plot(df_train, label='Training set')
        #main_df.plot(x = 'timestamp', y = 'adjusted_close', color = 'orange')
        plt.plot(df_test, label='Predicted point', marker="o", markersize=10, markerfacecolor="yellow")
        plt.legend();
        plt.xlabel(" Days count ")
        plt.ylabel("Closing value")
        st.pyplot(fig)
        st.subheader('Stock for IBM vs Time Chart')
        fig2 = plt.figure(figsize = (12, 6))

        df_close = df['close']
        df_close.loc[len(main_df.index)] = yhat_IBM[0][3]
        df_open = df['open']
        df_open.loc[len(main_df.index)] = yhat_IBM[0][0]
        df_high = df['high']
        df_high.loc[len(main_df.index)] = yhat_IBM[0][1]
        df_low = df['low']
        df_low.loc[len(main_df.index)] = yhat_IBM[0][2]
        plt.title('Stocks for IBM ', size=20)

        test_size = 1
        plt.plot(df_close[:-test_size], color = 'red' )
        plt.plot(df_close[-test_size:], label='Predicted close point', marker="o", markersize=10, markerfacecolor="r")
        plt.plot(df_open[:-test_size], 'b')
        plt.plot(df_open[-test_size:], label='Predicted open point', marker="o", markersize=10, markerfacecolor="b")
        plt.plot(df_high[:-test_size], 'g')
        plt.plot(df_high[-test_size:], label='Predicted high point', marker="o", markersize=10, markerfacecolor="g")
        plt.plot(df_low[:-test_size], 'y')
        plt.plot(df_low[-test_size:], label='Predicted low point', marker="o", markersize=10, markerfacecolor="y")
        plt.legend();
        plt.xlabel(" Days count ")
        st.pyplot(fig2)

        import pandas_datareader as data
        end = '2022-01-28'
        start = '2021-09-08'
        #ADD more companies here.
        companies = ['AAPL', 'TSLA', 'GOOGL', 'QCOM', 'FB', 'MSFT', 'AMD', 'MKC', 'WMT']
        images = ["apple.jpg", "tesla.jpg", "google.jpg", "qualcomm.jpg", "fb.jpg", "mf.jpg", "amd.jpg", "mkc.jpg", "wmt.jpg"]
        Adj_closing = []
        companies_betterthan_IBM = []
        companies_worsethan_IBM = []

        for i in companies:
            df_new = data.DataReader(i, 'yahoo', start, end)
            x = np.array(df_new)
            #fit the model
            from statsmodels.tsa.vector_ar.var_model import VAR
            model = VAR(endog=x)
            model_fit = model.fit()

            # make prediction on validation
            yhat_new = model_fit.forecast(model_fit.y, steps=1)
            index = df_new.columns.get_loc("Adj Close")
            Adj_closing.append(yhat_new[0][index])
        better = []
        not_better = []
        for i in range(len(Adj_closing)):
            if(Adj_closing[i]>yhat_IBM[0][3]):
                companies_betterthan_IBM.append(companies[i] + " : " + str(Adj_closing[i]))
                better.append(i)
            else:
                companies_worsethan_IBM.append(companies[i] + " : " + str(Adj_closing[i]))
                not_better.append(i)

        Adj_closing.append(closing_ibm)
        companies.append('IBM')

        from matplotlib import pyplot as plt

        st.subheader('Comparing other stocks with IBM')
        # Figure Size
        fig3 = plt.figure(figsize =(10, 7))

        # Horizontal Bar Plot
        plt.bar(companies, Adj_closing)
        plt.xlabel("Companies")
        plt.ylabel("Stock Adjusted Closing")

        # Show Plot
        plt.show()
        st.pyplot(fig3)


        stock_data = {'Adj_closing':Adj_closing}
        #st.dataframe(test)

        st.bar_chart(stock_data)
        #chart_data = pd.DataFrame(
             #np.random.randn(50, 1),
             #columns=["a"])

        #st.bar_chart(chart_data)


        """______ """
        """ These are the companies better than IBM """
        for i, j in zip(companies_betterthan_IBM, better):
            img = Image.open(images[j])
            st.image(img)
            st.write(i)

        """______  """
        """ These are the companies IBM is better than"""
        """ Umm! Let's think once more """
        for i, j in zip(companies_worsethan_IBM, not_better):
            img = Image.open(images[j])
            st.image(img)
            st.write(i)
