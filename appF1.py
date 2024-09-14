import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox

import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly.subplots as sp

# Page configuration
st.set_page_config(page_title="RaceIntel🏎️", layout="wide", page_icon="🏁")

# Title
st.title("RaceIntel🏎️")

# Load the model
model = joblib.load("model/modelF1.pkl")

#cargar datos
df = pd.read_csv('data/processed/features.csv')

 

st.header('Model')
st.markdown('Introduce los datos del conductor', unsafe_allow_html=True)

@st.cache_resource
def prediction(driverId , year,round , grid , lap):
     # New dataset with the input data
    constructorId= buscarDriver(driverId, 'constructorId')
    constructortotalpoints= buscarDriver(driverId, 'constructorPoints')
    constructorposition= buscarDriver(driverId, 'constructorPosition')
    drivertotalpoints=buscarDriver(driverId, 'driverPoints')
    driverposition=buscarDriver(driverId, 'position')
    currentracepoints = buscarDriver(driverId, 'currentRacePoints')
    statusId= buscarDriver(driverId, 'status')
    age= buscarDriver(driverId, 'age')
    input_data = pd.DataFrame([[driverId ,constructorId, year,round , grid , lap, constructortotalpoints, constructorposition, drivertotalpoints, driverposition, currentracepoints,statusId, age]], 
                                    columns=['driverId', 'constructorId', 'year', 'round', 'grid', 'laps',
                                            'constructortotalpoints', 'constructorposition', 'drivertotalpoints',
                                            'driverposition', 'currentracepoints','statusId', 'age'])
    
     # Make the prediction
    prediction = model.predict(input_data)

    return prediction[0]
    
# funciones para obtener datos
def buscarDriver(driverId, param):
    consulta = df.loc[(df['driverId'] == driverId) & (df['raceId'] == df['raceId'].max())]
    if param == 'age':
        return consulta['age']
    elif param == 'driverPoints':
        return consulta['drivertotalpoints']
    elif param == 'position':
        return consulta['driverposition']
    elif param == 'status':
        return consulta['statusId']
    elif param == 'currentRacePoints':
        return consulta['currentracepoints']
    elif param =='constructorId':
        return consulta['constructorId']
    elif param == 'constructorPoints':
        return consulta['contructortotalpoints']
    elif param == 'constructorPosition':
        return consulta['constructorposition']

# #mapeo de categorias:

# constructor = {
#     'McLaren': 1,
#     'Williams':3,
#     'Ferrari': 6,
#     'Red Bull': 9,
#     'Kick Sauber': 15,
#     'Aston Martin': 117,
#     'Mercedes': 131,
#     'Haas F1 Team': 210,
#     'Alpine': 214,
#     'RB': 215,

# }
driver = {
    'Lewis Hamilton': 1,
    'Fernando Alonso': 4,
    'Nico Hülkenberg':807,
    'Sergio Pérez': 815,
    'Daniel Ricciardo': 817,
    'Valtteri Bottas': 822,
    'Kevin Magnussen': 825,
    'Max Verstappen':830,
    'Carlos Sainz':832,
    'Esteban Ocon': 839,
    'Lance Stroll': 840,
    'Pierre Gasly': 842,
    'Charles Leclerc': 844,
    'Lando Norris': 846,
    'George Russell': 847,
    'Alexander Albon': 848,
    'Yuki Tsunoda': 852,
    'Guanyu Zhou': 855,
    'Oscar Piastri': 857,
    'Logan Sargeant': 858
}

driverId= st.selectbox('Conductor:',('Lewis Hamilton','Fernando Alonso','Nico Hülkenberg','Sergio Pérez','Daniel Ricciardo','Valtteri Bottas','Kevin Magnussen','Max Verstappen','Carlos Sainz','Esteban Ocon','Lance Stroll','Pierre Gasly',
    'Charles Leclerc','Lando Norris','George Russell','Alexander Albon','Yuki Tsunoda','Guanyu Zhou','Oscar Piastri','Logan Sargeant'))
# constructorId=st.radio('Escuderia:',('McLaren','Williams','Ferrari','Red Bull','Kick Sauber','Aston Martin','Mercedes','Haas F1 Team','Alpine','RB'))
year= 2024
round= st.slider('Ronda:', 1, 22, disabled=False)
grid= st.slider('Pocision de salida:', 1, 21, disabled=False)
laps= st.slider('Vueltas por carrera', 40, 80, disabled=False)




# Button to make the prediction
if st.button("Predict"):
    name= driverId
    driverId = driver[driverId]
    result = prediction(driverId , year,round , grid , laps)
    if result >21: 
        st.write(f"El conductor {name} no terminará la carrera")
    else:
        st.write(f"El conductor {name} terminará en la posición {result}")