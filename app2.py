import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import pickle as pkl

st.set_page_config(page_title="LOS Prediction",page_icon="⚕️",layout="wide",initial_sidebar_state="expanded")

@st.cache_data
def data():
    data = pd.read_csv("./data/los.csv.gz", compression='gzip', error_bad_lines=False)
    return data

@st.cache_data
def admin():
    admin = pd.read_csv("./data/admissions.csv.gz", compression='gzip', error_bad_lines=False)
    return admin

@st.cache_data
def patient():
    patient = pd.read_csv("./data/patients.csv.gz", compression='gzip', error_bad_lines=False)
    return patient

@st.cache_data
def disease_model():
    with gzip.open('./data/disease_model.sav.gz', 'rb') as f:
        model = pkl.load(f)
    return model

@st.cache_data
def los_model():
    model = pkl.load(open('./data/los_model.sav', 'rb'))
    return model
    
a = data()
b = admin()
c = patient()
f = los_model()

def los_prediction(f):
    html_temp = """ 
        <div style ="padding:13px;padding-bottom:50px"> 
        <h1 style ="color:white;text-align:center;">Length of Stay Prediction</h1> 
        </div> 
        """
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.write("---")
    st.header("PLEASE UPLOAD THE MEDICAL RECORD")
    uploaded_file = st.file_uploader("Upload your file here...", type=['csv'])
    trained_model = f
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        dataframe.fillna(0, inplace=True)
        result = trained_model.predict(dataframe)
    
    if st.button('Show Result') and result is not None:
        patient_id = [f'P{i:03}' for i in range(1, len(result) + 1)]
        prediction_df = pd.DataFrame({'Patient_ID': patient_id, 'Length_of_Stay': result})
        st.table(prediction_df.style.hide_index())

def los_visualization_menu(a,b,c):
        html_temp = """ 
        <div style ="padding:13px;padding-bottom:50px"> 
        <h1 style ="color:white;text-align:center;">Length of Stay Visualization</h1> 
        </div> 
        """
        st.markdown(html_temp, unsafe_allow_html = True) 
        menu = st.radio("SELECT THE VISUALIZATION TYPE",
                        ("Length of Stay",
                        "MIMIC-IV"
                        )
        )
        
        if menu == 'MIMIC-IV':
            los_visualization(a,b,c)
        else:
            los_filter(a)


def los_visualization(a,b,c):
    st.write('\n')
    st.text('Loading data...')
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
    
def los_filter(a):
        st.write('\n')
        st.text('Loading data...')
        data = a

        data['gender'].replace({0: 'MALE', 1: 'FEMALE'}, inplace=True)
        gender = st.sidebar.multiselect("Select the Gender:",
                                options=data["gender"].unique(),
                                default=data["gender"].unique()
                                )
        if not gender:
            st.warning("Please select at least one gender.")
            return
        age = st.sidebar.number_input("Age:", min_value=18, max_value=100, step=1, format="%d")
        
        df_selection = data.query(
            "gender == @gender &"
            "anchor_age == @age"
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
        fig1 = px.histogram(df_selection, x='los', nbins=200, range_x=[0, 115],
                        labels={'los': 'Length of Stay (days)', 'count': 'Count'},
                        )
        fig1.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig1.update_layout(width=800, height=600)
        st.plotly_chart(fig1, use_container_width=True)


page = st.sidebar.selectbox('SELECT PAGE',['LOS-Predictions','LOS-Visualization'])
if page == 'LOS-Predictions':
    los_prediction(f)
else:
    los_visualization_menu(a,b,c)
