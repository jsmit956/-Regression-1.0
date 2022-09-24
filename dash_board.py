#Steamlit_Stock


#Republican":0,"Democrat":1

import datetime
import streamlit as st
import pandas as pd
import altair as alt
from util import train_test_time


from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from sklearn.dummy import DummyRegressor
import numpy as np
import re


PARTY_ENCODING={"Republican":0,"Democratic":1}
# Drop down display for President and Senate and House and add the side bar
with st.sidebar:
    prez= st.radio("Please pick President's party",("Democratic","Republican"))
    b_prez=PARTY_ENCODING[prez]
    house = st.radio("Please pick majority party for the House of Representatives",("Democratic","Republican"))
    b_house=PARTY_ENCODING[house]
    senate=st.radio("Please majority party for the Senate",("Democratic","Republican"))
    b_senate=PARTY_ENCODING[senate]

#mortgage rate GDP and Fed funds rate and move to a side bar
with st.sidebar:
    mortgage=st.number_input("Please insert a 30 year mortgage")
    st.write("the mortgage rate you used is",mortgage)

    gdp=st.number_input("Please insert a GDP")
    st.write("The GDP you are using is",gdp)

    fed=st.number_input("Please insert a Fed Funds Rate")
    st.write("The Fed Funds you are using is",fed)
    num=st.number_input("Please input the number of days to predict on")
    st.write(num)

 #trying to figure this part out won't work
with st.sidebar:
    d = st.date_input(
    "When do you want to start",
    datetime(2019, 7, 6,))
    


#get data
def get_data():
    source=pd.read_csv("altformart.csv",infer_datetime_format=True)
    source["Date"]=pd.to_datetime(source["Date"])
    return source


source=get_data()



start=pd.Timestamp(d)
end=start+pd.DateOffset(months=1)
new_source=source[(source.Date > start)&(source.Date <end)]
x= alt.Chart(new_source, title="Evolution of stock prices").mark_line().encode(
    x= "Date:T",
    y="Price:Q",
    color="Symbol:N", 
    ).properties(
        width=750,
        height=550
    ).interactive()
st.altair_chart(x)

input_array=[fed,gdp,b_house,b_senate,b_prez,mortgage]
input_array=np.array(input_array).reshape(1,-1)


def machine_learning():
    regress=pd.read_csv("ready_for_machinelearning.csv",index_col="Date")
    return regress

regress=machine_learning()

all_tickers=['VBR',"SPY","QQQ","VTV"]
for x in all_tickers:
    X_train, X_test, y_train, y_test = train_test_time(regress,x)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    input_array_scaled=scaler.transform(input_array)
    X_test_scaled = scaler.transform(X_test)

    dum = DummyRegressor(strategy="median")
    reg = Ridge()
    reg2 = Ridge(alpha=2.0)

    dum.fit(X_train_scaled,y_train)
    reg.fit(X_train_scaled,y_train)
    reg2.fit(X_train_scaled,y_train)

    dum_predict= dum.predict(X_test_scaled)
    reg_predict = reg.predict(X_test_scaled)
    input_predict=reg.predict(input_array_scaled)
    input_predict=np.around(input_predict,2)
    input_predict=re.sub('( \[|\[|\])', '', str(input_predict))
    st.write(f"your prediction for {x}  is {input_predict}") 

