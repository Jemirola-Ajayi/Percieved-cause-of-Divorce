import streamlit as st
import joblib 
import numpy as np


#load the model
model = joblib.load('Best_Bagging_Model_For_Women_Diabetes-Risk_prediction.pkl')

#Title
st.title('Women Diabetics Prediction using a Tunned Bagged model')

st.markdown('Enter the required featutres below to get a prediction from the Model')


#Input fields
feature1 = st.number_input('Pregnancies')
feature2 = st.number_input('Glucose')
feature3 = st.number_input('BloodPressure')
feature4 = st.number_input('SkinThickness')
feature5 = st.number_input('Insulin')
feature6 = st.number_input('BMI')
feature7 = st.number_input('Pedigree')
feature8 = st.number_input('Age')

#Prediction buitton
if st.button('Predict'):
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])
    Prediction = model.predict(input_data)
    st.success(f'Prediction: {Prediction[0]}')

st.markdown("if you're a woman and the result of your prediction is 1, you are likely to have Diabetes!\n"
            "You should reach out to your Doctor")



