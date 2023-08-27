import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import pickle as pkl

st.set_page_config(page_title="LOS Prediction",page_icon="⚕️",layout="wide",initial_sidebar_state="expanded")

html_temp = """ 
<div style ="padding:13px;padding-bottom:50px"> 
<h1 style ="color:white;text-align:center;">Prediction and Visualization Length of Stay in Hospital</h1> 
</div> 
"""

# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 

@st.cache_data
def data():
    data = pd.read_csv("./data/final_data.csv")
    return data

@st.cache_data
def admin():
    admin = pd.read_csv("./data/admissions.csv.gz", compression='gzip', error_bad_lines=False)
    return admin

@st.cache_data
def patient():
    patient = pd.read_csv("./data/patients.csv.gz", compression='gzip', error_bad_lines=False)
    return patient

a = data()
b = admin()
c = patient()

def prediction_los(input_data):
    load_model = pkl.load(open('./data/los_model.sav', 'rb'))

    input_data_as_numpy_array = np.asarray(input_data)
    # Convert the tuple to a 2D numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Now you can use the model to predict
    predict_los = load_model.predict(input_data_reshaped)
    predictions = predict_los[0]
    st.write("Predicted LOS:", predictions, 'days')

def prediction():
    load_model = pkl.load(open('./data/los_model.sav', 'rb'))

    # # Your input data as a tuple
    # input_data_tuple = (12, 21, 7.7, 22.666667, 115, 1, 1.13, 169, 0, 30.766667, 11, 0, 0, 1.1, 0, 1.7, 32.4, 33.9, 95.5, 1.85, 0, 13.2, 25.3, 0, 3.65, 167.5, 4.6, 4.3, 13.4, 0, 0, 3.185, 143.5, 22, 16.1, 44, 7.303333, 233, 0, 0, 61, 0, 66.935484, 90.3, 137.387097, 3.541667, 5.875000, 3.458333, 104.804878, 51, 69.96, 86.32, 142.36, 97.5, 14.875, 99.316667)

    # # Convert the tuple to a 2D numpy array
    # input_data_array = np.array(input_data_tuple).reshape(1, -1)

    # # Now you can use the model to predict
    # predictions = load_model.predict(input_data_array)
    # st.write("Predicted outcome:", predictions)

    st.write("PLEASE INPUT THE MEDICAL RECORD")
    input_data = list()
    col1,col2,col3 = st.columns(3)

    with col1:
        anion_gap = st.text_input("Anion Gap Level")
        co2 = st.text_input("Calculated Total CO2 Level")
        free_calcium = st.text_input('Free Calcium Level')
        hematocrit = st.text_input('Hematocrit Level')
        i = st.text_input('I Level')
        lactate = st.text_input('Lactate Level')
        mcv = st.text_input('Mean Corpuscular Volume Level')
        pt = st.text_input('Prothrombin Time')
        phosphate = st.text_input('Phosphate Level')
        potassium_WB = st.text_input('Potassium Whole Blood Level')
        rapamycin = st.text_input('Rapamycin Level')
        urea_nitrogen = st.text_input('Urea Nitrogen Level')
        pH = st.text_input('pH Level')
        temp_gender = st.radio('Gender',
                          ("Male",
                           "Female"
                          ))
        if temp_gender == 'Male':
            gender = '0'
        else:
            gender = '1'
        diastolic_blood_pressure = st.text_input('Arterial Blood Pressure Diastolic Level')
        eye_opening = st.text_input('GCS - Eye Opening')
        heart_rate = st.text_input('Heart Rate')
        nibp_mean = st.text_input('Non Invasive Blood Pressure Mean Level')
        resp_rate = st.text_input('Respiratory Rate')

    with col2:
        bicarbonate = st.text_input("Bicarbonate Level")
        chloride = st.text_input("Chloride Level")
        glucose = st.text_input('Glucose Level')
        hemoglobin = st.text_input('Hemoglobin Level')
        inr = st.text_input('INR(P/T) Level')
        mch = st.text_input('Mean Corpuscular Hemoglobin Level')
        magnesium = st.text_input('Magnesium Level')
        ptt = st.text_input('Partial Thromboplastin Time')
        platelet_count = st.text_input('Platelet Count Level')
        rdw = st.text_input('Red Cell Distribution Width') 
        red_blood_cell = st.text_input('Red Blood Cells Level')
        white_blood_cell = st.text_input('White Blood Cells Level')
        po2 = st.text_input('PO2 Level')
        age = st.text_input('Age')
        arterial_blood_pressure_mean = st.text_input('Arterial Blood Pressure Mean Level')
        motor_response = st.text_input('GCS - Motor Response')
        o2_fraction = st.text_input('Inspired O2 Fraction Level')
        nibp_systolic = st.text_input('Non Invasive Blood Pressure Systolic Level')
        temperature = st.text_input('Temperature (°F)')
        
    with col3:        
        calcium = st.text_input('Total Calcium Level')
        creatinine = st.text_input('Creatinine Level')
        h = st.text_input('H Level')
        heparin = st.text_input('Heparin Level')
        l = st.text_input('L Level')
        mchc = st.text_input('Mean Corpuscular Hemoglobin Concentration Level')
        oxy_saturation = st.text_input('Oxygen Saturation Level')
        phenobarbital = st.text_input('Phenobarbital Level')
        potassium = st.text_input('Potassium Level')
        rdw_sd = st.text_input('Red Cell Distribution Width - Standard Deviation Level')
        sodium = st.text_input('Sodium Level')
        pco2 = st.text_input('pCO2 Level')
        tacroFK = st.text_input('Tacrolimus FK Level')
        jh_hlm = st.text_input('Activity / Mobility (JH-HLM)')
        systolic_blood_pressure = st.text_input('Arterial Blood Pressure Systolic Level')
        verbal_response = st.text_input('GCS - Verbal Response')
        nibp_diastolic = st.text_input('Non Invasive Blood Pressure Diastolic Level')
        o2_saturation = st.text_input('O2 Saturation Pulseoxymetry Level')
        
    
    if st.button('LOS Result'):
        input_data.extend([anion_gap, bicarbonate, calcium, co2, chloride, creatinine, free_calcium, glucose, h, hematocrit, hemoglobin, heparin, i, inr, l,
                      lactate, mch, mchc, mcv, magnesium, oxy_saturation, pt, ptt, phenobarbital, phosphate, platelet_count, potassium, potassium_WB, rdw,
                      rdw_sd, rapamycin, red_blood_cell, sodium, urea_nitrogen, white_blood_cell, pco2, pH, po2, tacroFK, gender, age, jh_hlm, diastolic_blood_pressure,
                      arterial_blood_pressure_mean, systolic_blood_pressure, eye_opening, motor_response, verbal_response, heart_rate, o2_fraction, nibp_diastolic, nibp_mean,
                      nibp_systolic, o2_saturation, resp_rate, temperature])
        

        numeric_input_data = [float(value) if value and value != "0" else 0 for value in input_data]
        prediction_los(numeric_input_data)

def visualization_menu(a,b,c):
        menu = st.radio("SELECT THE VISUALIZATION TYPE",
                        ("Length of Stay",
                        "MIMIC-IV"
                        )
    )
        
        if menu == 'MIMIC-IV':
            visualization(a,b,c)
        else:
            filter(a)

def visualization(a,b,c):

    st.write('\n')
    data_load_state = st.text('Loading data...')
    data = a
    admin = b
    patient = c

    col1, col2 = st.columns(2)
    with col1:
        #ROW 1 - LOS HOSPITAL ADMINSSIONS
        st.subheader('Distribution of LOS for all Hospital Admissions')
        fig1 = px.histogram(data, x='los', nbins=200, range_x=[0, 50],
                        labels={'los': 'Length of Stay (days)', 'count': 'Count'},
                        )
        fig1.update_traces(marker=dict(line=dict(color='black', width=1)))
        st.plotly_chart(fig1)

        #ROW 2 - GENDER
        # Replace 0 with 'Male' and 1 with 'Female' in the 'gender' column
        data['gender'].replace({0: 'Male', 1: 'Female'}, inplace=True)
        gender_counts = data['gender'].value_counts()
        st.subheader('Gender')
        st.bar_chart(gender_counts)
        
    with col2:
        #ROW 1 - AGE
        st.subheader('Age')
        fig2 = px.histogram(patient, x='anchor_age', nbins=20,
                        labels={'anchor_age': 'Age', 'count': 'Count'},
                        )
        fig2.update_traces(marker=dict(line=dict(color='black', width=1)))
        st.plotly_chart(fig2)

        #ROW 2 - RACE
        admin['race'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
        admin['race'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
        admin['race'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
        admin['race'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
        admin['race'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 
                                'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)
        admin['race'].loc[~admin['race'].isin(admin['race'].value_counts().nlargest(5).index.tolist())] = 'OTHER/UNKNOWN'
        admin['race'].value_counts()

        st.subheader('Races of Patients')
        fig4 = px.histogram(admin, y='race', nbins=200,
                        labels={'race': 'Race', 'count': 'Count'},
                        )
        st.plotly_chart(fig4)
    
def filter(a):
        st.write('\n')
        data_load_state = st.text('Loading data...')
        data = a

        data['ICU'].replace({0: 'NON-ICU', 1: 'ICU'}, inplace=True)
        icu = st.sidebar.multiselect("Select ICU or NON-ICU:",
                                options=data["ICU"].unique(),
                                default=data["ICU"].unique())
        if not icu:
            st.warning("Please select at least one.")
            return

        data['gender'].replace({0: 'MALE', 1: 'FEMALE'}, inplace=True)
        gender = st.sidebar.multiselect("Select the Gender:",
                                options=data["gender"].unique(),
                                default=data["gender"].unique()
                                )
        if not gender:
            st.warning("Please select at least one gender.")
            return
            
        race = st.sidebar.radio("Select the Race:",
                                    ("ASIAN",
                                    "WHITE",
                                    "HISPANIC/LATINO",
                                    "BLACK/AFRICAN AMERICAN",
                                    "OTHER"
                                    )
                                )
        
        asian = 1 if "ASIAN" in race else 0
        white = 1 if "WHITE" in race else 0
        hispanic = 1 if "HISPANIC/LATINO" in race else 0
        black = 1 if "BLACK/AFRICAN AMERICAN" in race else 0
        other = 1 if "OTHER" in race else 0

        age = st.sidebar.radio("Select the Age:",
                                    ("0-18 - CHILD",
                                    "19-34 - YOUNG ADULT",
                                    "35-55 - MIDDLE ADULT",
                                    "56+ - SENIOR"
                                    )
        )

        child = 1 if "0-18 - CHILD" in age else 0
        youngadult = 1 if "19-34 - YOUNG ADULT" in age else 0
        middleadult = 1 if "35-55 - MIDDLE ADULT" in age else 0
        senior = 1 if "56+ - SENIOR" in age else 0

        admission = st.sidebar.radio("Select the Admission Type:",
                                           ("EW EMER",
                                            "EU OBSERVATION",
                                            "OBSERVATION ADMIT",
                                            "URGENT",
                                            "SURGICAL SAME DAY ADMISSION",
                                            "DIRECT EMER.",
                                            "DIRECT OBSERVATION",
                                            "ELECTIVE",
                                            "AMBULATORY OBSERVATION"
                                           )
                                           )
        
        ew_emer = 1 if "EW EMER" in admission else 0
        eu_obs = 1 if "EU OBSERVATION" in admission else 0
        obs_admit = 1 if "OBSERVATION ADMIT" in admission else 0
        urgent = 1 if "URGENT" in admission else 0
        surgical_day_admission = 1 if "SURGICAL SAME DAY ADMISSION" in admission else 0
        direct_emar = 1 if "DIRECT EMER" in admission else 0
        direct_obs = 1 if "DIRECT OBSERVATION" in admission else 0
        elective = 1 if "ELECTIVE" in admission else 0
        ambulatory_obs = 1 if "AMBULATORY OBSERVATION" in admission else 0

        df_selection = data.query(
            "gender == @gender & "
            "ICU == @icu & "
            "`RAC_ASIAN` == @asian & "
            "`RAC_BLACK/AFRICAN AMERICAN` == @black & "
            "`RAC_HISPANIC/LATINO` == @hispanic & "
            "`RAC_OTHER/UNKNOWN` == @other & "
            "`RAC_WHITE` == @white & "
            "AGE_middle_adult == @middleadult & "
            "AGE_senior == @senior & "
            "AGE_young_adult == @youngadult & "
            "AGE_child == @child & "
            "`ADM_AMBULATORY OBSERVATION` == @ambulatory_obs &"
            "`ADM_DIRECT EMER.` == @direct_emar &"
            "`ADM_DIRECT OBSERVATION` == @direct_obs &"
            "ADM_ELECTIVE == @elective &"
            "`ADM_EU OBSERVATION` == @eu_obs &"
            "`ADM_EW EMER.` == @ew_emer &"
            "`ADM_OBSERVATION ADMIT` == @obs_admit &"
            "`ADM_SURGICAL SAME DAY ADMISSION` == @surgical_day_admission &"
            "ADM_URGENT == @urgent "
        )

        st.title(":bar_chart: Length of Stay Dashboard")
        st.markdown("##")

        lowest_stay = int(df_selection["los"].min())
        highest_stay = int(df_selection["los"].max())
        average_stay = round(df_selection["los"].mean(),1)

        col1, col2,col3 = st.columns(3)
        with col1:
            st.subheader("Lowest Length of Stay")
            st.subheader(f"{lowest_stay}")
        with col2:
            st.subheader("Highest Length of Stay")
            st.subheader(f"{highest_stay}")
        with col3:
            st.subheader("Aevrage Length of Stay")
            st.subheader(f"{average_stay}")
        st.markdown("---")

        st.subheader('Distribution of LOS for all Hospital Admissions')
        fig2 = px.histogram(df_selection, x='los', nbins=200, range_x=[0, 50],
                        labels={'los': 'Length of Stay (days)', 'count': 'Count'},
                        )
        fig2.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig2.update_layout(width=800, height=600)
        st.plotly_chart(fig2, use_container_width=True)


page = st.sidebar.selectbox('SELECT PAGE',['Predictions','Visualization']) 
st.sidebar.write("---")
if page == 'Predictions':
    prediction()
else:
    visualization_menu(a,b,c)