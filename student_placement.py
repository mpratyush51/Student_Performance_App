import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://pratyushm555:Krishnaisgod@cluster0.skllidd.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['Student']
collection = db["student_performance"]

@st.cache_resource  # This tells Streamlit: "Load this once and remember it!"
def load_model():
    with open("student_placement_model.pkl", 'rb') as file:
        model, scaler , le = pickle.load(file)

    return model, scaler , le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_standardised = scaler.transform(df)
    return df_standardised

def predict_data(data):
    model, scaler, le  = load_model()
    processed_data = preprocessing_input_data(data = data , scaler = scaler, le = le)
    prediction = model.predict(processed_data)
    return prediction


def main():
    # st.title("Student Placement Prediction")
    # st.write("Enter your data to get a prediction for you performance")

    # study_hours = st.number_input("Hours Studied", min_value = 1 , max_value = 10, value = 5)
    # prev_scores = st.number_input("Previous Scores", min_value = 40 , max_value = 100, value = 75)
    # extra = st.selectbox("Extracurricular Activities", ["Yes" , "No"])
    # sleep_hours = st.number_input("Sleep Hours", min_value = 4 , max_value = 10, value = 7)
    # paper_solved = st.number_input("Sample Question Papers Practiced", min_value = 0 , max_value = 10, value = 5)
    
    # if st.button("Predict Score"):

    #     user_data = {
    #         "Hours Studied" : study_hours,
    #         "Previous Scores" : prev_scores,
    #         "Extracurricular Activities" : extra,
    #         "Sleep Hours" : sleep_hours,
    #         "Sample Question Papers Practiced" :paper_solved
    #     }

    #     prediction = predict_data(data = user_data)

    #     st.success(f"Your score is {prediction}")



    
    # --- UI Improvements ---
    st.set_page_config(page_title="Placement Predictor", page_icon="📝")
    
    st.title("🎓 Student Performance Prediction")
    st.markdown("Enter your details below to estimate your performance score.")
    
    # Organized Layout
    col1, col2 = st.columns(2)
    
    with col1:
        study_hours = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
        prev_scores = st.number_input("Previous Scores", min_value=40, max_value=100, value=75)
        extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        
    with col2:
        sleep_hours = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
        paper_solved = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)

    if st.button("Predict Score", use_container_width=True):
        user_data = {
            "Hours Studied": study_hours,
            "Previous Scores": prev_scores,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": paper_solved
        }

        try:
            prediction = predict_data(data=user_data)
            # prediction is usually an array, so we take the first element [0]
            result_value = float(prediction.item())

            # 2. Create a clean copy for MongoDB
            # MongoDB doesn't like it if 'user_data' contains NumPy types from the label encoder
            db_record = {
                "Hours Studied": int(study_hours),
                "Previous Scores": int(prev_scores),
                "Extracurricular Activities": extra,
                "Sleep Hours": int(sleep_hours),
                "Sample Question Papers Practiced": int(paper_solved),
                "Prediction": result_value
            }

            collection.insert_one(db_record)
            
            st.success(f"### Predicted Performance Index: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()

