import streamlit as st
import pandas as pd
import numpy as np

pipe = pd.read_pickle(r'C:\Users\HP\Downloads\MlModel.pkl')
df = pd.read_pickle(r'C:\Users\HP\Downloads\LaptopSpecs.pkl')

Touchscreen = list(set(df['Touchscreen']))
ips = list(set(df['IPS']))

st.title("Laptop Predictor")
Company = st.selectbox('Brand', df['Company'].unique())
TypeName = st.selectbox('Type', df['TypeName'].unique())
Ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
Weight = st.number_input('Weight')
Touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
IPS = st.selectbox('IPS', ['Yes', 'No'])
Inches = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920 x 1080', '1366 x 768', '1600 x 900', '3840 x 1000', '2880 x 1800', '2560 x 1600', '2560 x 1440', '2304 x 1440'])
Cpu_Brand = st.selectbox('Brand', df['Cpu_Brand'].unique())

HDD = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
SSD = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])
Gpu_Brand = st.selectbox('GPU', df['Gpu_Brand'].unique())
operating_system = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    PPI = None

    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    PPI = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Inches

    # Create a dictionary with input features
    input_data = {
        'Company': Company,
        'TypeName': TypeName,
        'Ram': Ram,
        'Weight': Weight,
        'Touchscreen': Touchscreen,
        'IPS': ips,
        'Inches': Inches,
        'ScreenResolution': resolution,
        'Cpu_Brand': Cpu_Brand,
        'HDD': HDD,
        'SSD': SSD,
        'Gpu_Brand': Gpu_Brand,
        'os': operating_system,
        'PPI': PPI
    }

    # Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([input_data])

    # Use the trained model to predict the price
    predicted_price = round(int(np.exp(pipe.predict(input_df)[0])))

    # Display the predicted price
    st.success(f'The predicted price of the laptop is: Rs {predicted_price:.2f}')



