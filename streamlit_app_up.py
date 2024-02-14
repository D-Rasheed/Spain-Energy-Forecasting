import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, mean_absolute_error, r2_score

st.set_option('deprecation.showPyplotGlobalUse', False)
#ADD SOME SOMMENT FOR UNDERSTANDING
# Importing data
df_viz=pd.read_csv("D:\programming project feb 2024\Spain Energy Dataset\df_electricity_cleaned.csv")
#Reading our Dataset
df_ml=pd.read_csv("D:\programming project feb 2024\Spain Energy Dataset\df_viz.csv",index_col='time',)

st.sidebar.image("https://i0.wp.com/thetechtian.com/wp-content/uploads/2022/07/Different-Sources-of-Energy.jpg?fit=1600%2C1067&ssl=1",width=300)
st.sidebar.header('SPAIN ENERGY PRODUCTION DATASET')
menu = st.sidebar.radio(
    "Menu:",
    ("Intro", "Data", "Analysis", "Models"),
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Project Submitted By: Danish Rasheed')
st.sidebar.write('Matricola No.: VR497604')
st.sidebar.write('Github Repositories:')
st.sidebar.write('https://github.com/D-Rasheed/Spain-Energy-Forecasting')



if menu == 'Intro':
   st.image("https://www.shutterstock.com/shutterstock/photos/2187642671/display_1500/stock-vector-how-much-energy-we-consume-fossil-fuel-renewable-energy-nuclear-petroleum-oil-natural-gas-2187642671.jpg",width=700)
   st.title('SPAIN ENERGY PRODUCTION')
   st.header('Context')
   st.write('In a paper released early 2019, forecasting in energy markets is identified as one of the highest leverage contribution areas of Machine/Deep Learning toward transitioning to a renewable based electrical infrastructure.')
   st.header('Content')
   st.write('This dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain. Consumption and generation data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España. Weather data was purchased as part of a personal project from the Open Weather API for the 5 largest cities in Spain and made public here.')
   st.write('''## Sources and References

kaggle: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather?select=energy_dataset.csv

Streamlit: https://streamlit.io/

Streamlit Doc: https://docs.streamlit.io/


## Used tools
| Data mining		| Visualization 	|
|---				|---				|
| - Jupyter Notebook| - Streamlit		|
| - Sklearn 		| - Python			|
| - Python			| - Numpy			|
| - Pandas			| - Matplotlib		|
| - Numpy			| - Seaborn		    |

''')

elif menu == 'Data':
   st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTidu96E25gtVXJ7bhUWgNE8_6jnIbeReG5UsoDpRBAjrBxlpnUPLl164mbO6M5VzVoLRs&usqp=CAU",width=700)
   st.title("DataFrame:")
   st.write(">***35064 entries | 29 columns***")
   st.dataframe(df_viz)
   
elif menu == 'Analysis':
# Convert 'time' column to datetime with UTC
   df_viz["time"] = pd.to_datetime(df_viz["time"], utc=True)
   df_viz = df_viz.set_index('time')

# Set timeframe data for easier accessing
   df_viz["year"] = df_viz.index.year
   df_viz["month"] = df_viz.index.month
   df_viz["month_name"] = df_viz.index.month_name()
   df_viz["weekdays"] = df_viz.index.day_name()

# Streamlit app header
   st.title("Energy Generation and Price Analysis")

# Line plot for generation data
   st.subheader("Energy Generation Overview")
   cols_plot = ["generation_wind_onshore", "generation_solar", "generation_fossil_hard_coal", "generation_fossil_gas"]
   st.line_chart(df_viz[cols_plot].loc["2018-01":"2018-12"])

# Box plots for monthly generation
   st.subheader("Monthly Generation Box Plots")
   for name in ["generation_wind_onshore", "generation_solar", "generation_fossil_hard_coal", "generation_fossil_gas"]:
    fig, ax = plt.subplots()
    sns.boxplot(data=df_viz, x='month_name', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_xlabel("Months")
    ax.set_title(name)
    st.pyplot(fig)

# Line plot for price data
   st.subheader("Electricity Price Analysis")
   monthly_mean=df_viz["price_actual"].resample("m").mean()
   weekly_mean=df_viz["price_actual"].resample("w").mean()
   daily_mean=df_viz["price_actual"].resample("d").mean()
   
   Price_plot = ["price_actual", "price_day_ahead"]
   fig, ax = plt.subplots(figsize=[10, 8])
   ax.plot(daily_mean.loc["2018-01":"2018-12"], marker="*", linewidth=0.5, label="Daily mean")
   ax.plot(weekly_mean.loc["2018-01":"2018-12"], marker="o", linewidth=0.8, label="Weekly average resample")
   ax.plot(monthly_mean.loc["2018-01":"2018-12"], marker="+", linewidth=1.8, label="Monthly mean")
   ax.set_ylabel("Price")
   ax.legend()
   st.pyplot(fig)

# Line plot for weekly mean price
   st.subheader("Weekly Mean Price")
   Price_plot = ["price_actual", "price_day_ahead"]
   st.line_chart(df_viz[Price_plot].resample("w").mean())

# Histogram for price distribution
   st.subheader("Price Distribution Histogram")
   plt.figure(figsize=(8, 5))
   sns.histplot(df_viz, x='price_actual')
   st.pyplot()


elif menu =='Models':
   st.image("https://smartindustry.vn/wp-content/uploads/2020/02/what-is-deep-learning-large.jpg",width=700)
   # Streamlit app header
   st.title("MODEL SELECTION")
   #Adding a dropdown to select anyone model
   selected_model = st.selectbox("## Select Model", ["Linear Regression", "Ridge", "Random Forest"])
   
   # creating train test data sets
   x = df_ml.drop(['price_actual','year','month','month_name','weekdays'], axis=1)
   y = df_ml['price_actual']
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

   # Model training and evaluation

   #LinearRegression
   if selected_model == "Linear Regression":
        lr_model = LinearRegression()
        lr_model.fit(x_train, y_train)
        lr_predictions = lr_model.predict(x_test)
        st.write("------------------------------------")
        st.write("Linear Regression")
        lr_mae = mean_absolute_error(y_test, lr_predictions)
        st.write("MAE:", lr_mae)
        lr_r2 = r2_score(y_test, lr_predictions)
        st.write("R-squared:", lr_r2)
   
   #Ridge
   elif selected_model == "Ridge":
        r_model = Ridge()
        r_model.fit(x_train, y_train)
        r_predictions = r_model.predict(x_test)
        st.write("------------------------------------")
        st.write("Ridge")
        r_mae = mean_absolute_error(y_test, r_predictions)
        st.write("MAE:", r_mae)
        r_r2 = r2_score(y_test, r_predictions)
        st.write("R-squared:", r_r2)

   #Random Forest Regressor
   elif selected_model == "Random Forest":
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(x_train, y_train)
        rf_predictions = rf_model.predict(x_test)
        st.write("------------------------------------")
        st.write("Random Forest ")
        rf_mae = mean_absolute_error(y_test, rf_predictions)
        st.write("MAE:", rf_mae)
        rf_r2 = r2_score(y_test, rf_predictions)
        st.write("R-squared:", rf_r2)
        st.write("------------------------------------")
   else:
        st.error("Invalid model selection")