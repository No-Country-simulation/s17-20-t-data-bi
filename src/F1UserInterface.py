import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os
import joblib
from io import BytesIO
import requests
import sklearn


st.title("F1 Race Intel")
st.info(
    "Machine Learning Model for Prediction of Formula 1 Grand Prix Results."
)

#ID mappings for the diabetes dataset
#Admission type: Nominal Integer identifier corresponding to 8 distinct values

admission_type_id_mapping= {
    1:"Emergency",
    2:"Urgent",
    3:"Elective",
    4:"Newborn",
    5:"Not Available",
    6:"NULL",
    7:"Trauma Center",
    8:"Not Mapped",
}

#Discharge disposition: Nominal Integer identifier corresponding to 29 distinct values

discharge_disposition_id_mapping={
    1:"Discharged to home",
    2:"Discharged/transferred to another short term hospital",
    3:"Discharged/transferred to SNF",
    4:"Discharged/transferred to ICF",
    5:"Discharged/transferred to another type of inpatient care institution",
    6:"Discharged/transferred to home with home health service",
    7:"Left AMA",
    8:"Discharged/transferred to home under care of Home IV provider",
    9:"Admitted as an inpatient to this hospital",
    10:"Neonate discharged to another hospital for neonatal aftercare",
    11:"Expired",
    12:"Still patient or expected to return for outpatient services",
    13:"Hospice / home",
    14:"Hospice / medical facility",
    15:"Discharged/transferred within this institution to Medicare approved swing bed",
    16:"Discharged/transferred/referred another institution for outpatient services",
    17:"Discharged/transferred/referred to this institution for outpatient services",
    18:"NULL",
    19:"Expired at home. Medicaid only, hospice.",
    20:"Expired in a medical facility. Medicaid only, hospice.",
    21:"Expired, place unknown. Medicaid only, hospice.",
    22:"Discharged/transferred to another rehab fac including rehab units of a hospital .",
    23:"Discharged/transferred to a long term care hospital.",
    24:"Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
    25:"Not Mapped",
    26:"Unknown/Invalid",
    30:"Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
    27:"Discharged/transferred to a federal health care facility.",
    28:"Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
    29:"Discharged/transferred to a Critical Access Hospital (CAH)",
}

#Admission source: Nominal Integer identifier corresponding to 21 distinct values

admission_source_id_mapping = {
    1:"	Physician Referral",
    2:"	Clinic Referral",
    3:"HMO Referral",
    4:"Transfer from a hospital",
    5:"Transfer from a Skilled Nursing Facility (SNF)",
    6:"Transfer from another health care facility",
    7:"Emergency Room",
    8:"Court/Law Enforcement",
    9:"Not Available",
    10:"Transfer from critial access hospital",
    11:"Normal Delivery",
    12:"Premature Delivery",
    13:"Sick Baby",
    14:"Extramural Birth",
    15:"Not Available",
    17:"NULL",
    18:"Transfer From Another Home Health Agency",
    19:"Readmission to Same Home Health Agency",
    20:"Not Mapped",
    21:"Unknown/Invalid",
    22:"Transfer from hospital inpt/same fac reslt in a sep claim",
    23:"Born inside this hospital",
    24:"Born outside this hospital",
    25:"Transfer from Ambulatory Surgery Center",
    26:"Transfer from Hospice",
}


# diccionario de valores limite

diccionario = {'upper_limit_admission_type_id': 3.0,
 'lower_limit_admission_type_id': 1.0,
 'upper_limit_discharge_disposition_id': 4,
 'lower_limit_discharge_disposition_id': 1,
 'upper_limit_admission_source_id': 7.0,
 'lower_limit_admission_source_id': 1.0,
 'upper_limit_time_in_hospital': 6.0,
 'lower_limit_time_in_hospital': 2.0,
 'upper_limit_num_lab_procedures': 57.0,
 'lower_limit_num_lab_procedures': 31.0,
 'upper_limit_num_procedures': 2.0,
 'lower_limit_num_procedures': 0.0,
 'upper_limit_num_medications': 20.0,
 'lower_limit_num_medications': 11.0,
 }

diccionario1 = {'upper_limit_admission_type_id': 6.0,
 'lower_limit_admission_type_id': -2.0,
 'upper_limit_discharge_disposition_id': 8.5,
 'lower_limit_discharge_disposition_id': -3.5,
 'upper_limit_admission_source_id': 16.0,
 'lower_limit_admission_source_id': -8.0,
 'upper_limit_time_in_hospital': 12.0,
 'lower_limit_time_in_hospital': -4.0,
 'upper_limit_num_lab_procedures': 96.0,
 'lower_limit_num_lab_procedures': -8.0,
 'upper_limit_num_procedures': 5.0,
 'lower_limit_num_procedures': -3.0,
 'upper_limit_num_medications': 33.5,
 'lower_limit_num_medications': -2.5,
 'upper_limit_number_outpatient': 0.0,
 'lower_limit_number_outpatient': 0.0,
 'upper_limit_number_emergency': 0.0,
 'lower_limit_number_emergency': 0.0,
 'upper_limit_number_inpatient': 2.5,
 'lower_limit_number_inpatient': -1.5,
 'upper_limit_number_diagnoses': 13.5,
 'lower_limit_number_diagnoses': 1.5}

def user_input_features(  driverId,
                          constructorId,
                          year,
                          round,
                          grid,
                          laps,
                          constructortotalpoints,
                          constructorposition,
                          drivertotalpoints,
                          driverposition,
                          currentracepoints,
                          statusId,
                          age ):


  
  data=[pd.to_numeric(driverId),
        pd.to_numeric(constructorId),
        pd.to_numeric(year),
        pd.to_numeric(round),
        pd.to_numeric(grid),
        pd.to_numeric(laps),
        pd.to_numeric(constructortotalpoints),
        pd.to_numeric(constructorposition),
        pd.to_numeric(drivertotalpoints),
        pd.to_numeric(driverposition),
        pd.to_numeric(currentracepoints),
        pd.to_numeric(statusId),
        pd.to_numeric(age)]

  df = pd.DataFrame(data).T
  df.columns = ['driverId','constructorId','year','round','grid','laps','constructortotalpoints','constructorposition','drivertotalpoints','driverposition','currentracepoints','statusId','age']
  

  current_prediction = loaded_model.predict(df)
  


  text_to_display="The model's prediction is that this driver will end in position "+str(current_prediction)
  #st.popover("Model executed. Switch to next tab to see the model's Prediction")
  #with tab4:
  st.popover(text_to_display)
  return text_to_display

tab1,tab2,tab3,tab4 = st.tabs(['Main', 'About F1RaceIntel', 'Data Input', 'Model Status' ])
with tab1:
   

  image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/F1_%28registered_trademark%29.svg/250px-F1_%28registered_trademark%29.svg.png'
  response = requests.get(image_url)

  # Check if the request was successful
  if response.status_code == 200:
    
    image = Image.open(BytesIO(response.content))
    st.image(image)
  else:
    st.write("Failed to download image. Status code:", response.status_code)
    
with tab2:
  st.write("Daniel Carrillo ")
  st.write("Sergio González")
  st.write("Miguel Ismerio")
  st.write("Gabriela Pardo")
  st.write("Micaela Pequeño")  
  st.write("Antonia Soler")
  
with tab4:

  path='https://github.com/asoler2004/nocountrys17/raw/main/model.pkl'
  response = requests.get(path)

  # Check if the request was successful
  if response.status_code == 200:
    # Load the model using pickle
    loaded_model = pickle.loads(response.content)
    #st.write("Model loaded successfully!")
  else:
    st.write("Failed to download the model. Status code:", response.status_code)
  
  st.write(loaded_model)


   
with tab3:
  driverId= st.text_input("driverId")
  constructorId=st.text_input("constructorId")
  year= st.text_input("year")
  round= st.text_input("round")
  grid= st.text_input("grid")
  laps= st.text_input("laps")
  constructortotalpoints=st.text_input("constructortotalpoints")
  constructorposition=st.text_input("constructorposition")
  drivertotalpoints=st.text_input("drivertotalpoints")
  driverposition=st.text_input("driverposition")
  currentracepoints=st.text_input("currentracepoints")
  statusId= st.text_input("statusId")
  age=st.text_input("age")
                    
  st.button("Submit",on_click= user_input_features,args=[ driverId,
                                                          constructorId,
                                                          year,
                                                          round,
                                                          grid,
                                                          laps,
                                                          constructortotalpoints,
                                                          constructorposition,
                                                          drivertotalpoints,
                                                          driverposition,
                                                          currentracepoints,
                                                          statusId,
                                                          age ])
                                                            



