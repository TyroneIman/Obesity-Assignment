import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Load model
model = joblib.load('Obesity_model.pkl')
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = RobustScaler()
scaler.fit(data[numerical_cols])

# Mapping untuk output prediksi yang lengkap
obesity_levels = {
    'Insufficient_Weight': 'Underweight',
    'Normal_Weight': 'Normal Weight',
    'Overweight_Level_I': 'Overweight Level I',
    'Overweight_Level_II': 'Overweight Level II',
    'Obesity_Type_I': 'Obesity Type I',
    'Obesity_Type_II': 'Obesity Type II',
    'Obesity_Type_III': 'Obesity Type III'
}

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def encoding(self, input_data):
        # Create a copy to avoid modifying the original
        df = input_data.copy()
        
        # Columns to process
        label_enc_cols = ['CAEC', 'CALC']
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        
        # Label Encoding for ordinal columns
        for col in label_enc_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Binary encoding
        for col in binary_cols:
            df[col] = df[col].map({'no': 0, 'yes': 1, 'Female': 0, 'Male': 1})
        
        # Transportation mapping
        transport_mapping = {
            'Public_Transportation': 0, 
            'Walking': 1, 
            'Automobile': 2, 
            'Motorbike': 3, 
            'Bike': 4
        }
        df['MTRANS'] = df['MTRANS'].map(transport_mapping)
        
        return df

def preprocess_input(age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, 
                    family_history, favc, caec, smoke, scc, calc, mtrans):
    # Membuat dataframe dari input user
    input_data = pd.DataFrame([[age, height, weight, fcvc, ncp, ch2o, faf, tue, gender, 
                               family_history, favc, caec, smoke, scc, calc, mtrans]],
                            columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Gender', 
                                     'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])
    
    # Initialize and use the preprocessor
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.encoding(input_data)
    
    # Scale numerical columns
    processed_data[numerical_cols] = scaler.transform(processed_data[numerical_cols])
    
    return processed_data.values

def main():
    st.title('Obesity Level Prediction App')
    st.markdown(
        "<div style='background-color:#E7F3FC; padding:20px; border-radius:5px; margin-bottom:20px;'>"
        "<p style='color:darkblue; margin:0;'>This app predicts your obesity level based on lifestyle and physical metrics!</p>"
        "</div>",
        unsafe_allow_html=True
    )

    with st.expander("📊 Dataset Overview"):
        st.write("This is the raw data used for training the model:")
        st.dataframe(data.head())
        st.write(f"Dataset shape: {data.shape}")

    with st.expander("📈 Data Visualization"):
        st.write("Height vs Weight based on Obesity Level")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x="Height", y="Weight", hue="NObeyesdad", 
                        palette="coolwarm", ax=ax, s=50)
        plt.xlabel("Height (m)", fontsize=12)
        plt.ylabel("Weight (kg)", fontsize=12)
        plt.title("Height vs Weight by Obesity Level", fontsize=14)
        plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    with st.form("user_input_form"):
        st.subheader("📝 Enter Your Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Personal Information**")
            age = st.slider("Age", min_value=1, max_value=100, value=25)
            height = st.slider("Height (m)", min_value=1.0, max_value=2.5, value=1.7, step=0.01)
            weight = st.slider("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            family_history = st.selectbox("Family history with overweight", ["yes", "no"])
            
        with col2:
            st.write("**Lifestyle Habits**")
            favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["yes", "no"])
            fcvc = st.slider("Frequency of vegetable consumption (FCVC)", 1, 3, 2)
            ncp = st.slider("Number of main meals (NCP)", 1, 4, 3)
            ch2o = st.slider("Daily water consumption (CH2O)", 1, 3, 2)
            
        st.write("**Physical Activity & Eating Habits**")
        col3, col4 = st.columns(2)
        
        with col3:
            faf = st.slider("Physical activity frequency (FAF)", 0, 3, 1)
            tue = st.slider("Time using technology devices (TUE)", 0, 2, 1)
            smoke = st.selectbox("Do you smoke?", ["yes", "no"])
            
        with col4:
            scc = st.selectbox("Do you monitor calories? (SCC)", ["yes", "no"])
            caec = st.selectbox("Consumption of food between meals (CAEC)", 
                               ["no", "Sometimes", "Frequently", "Always"])
            calc = st.selectbox("Alcohol consumption (CALC)", 
                               ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("Transportation used (MTRANS)", 
                                ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

        submitted = st.form_submit_button("Predict Obesity Level")

        if submitted:
            try:
                # Preprocess input
                processed_data = preprocess_input(
                    age, height, weight, fcvc, ncp, ch2o, faf, tue, 
                    gender, family_history, favc, caec, smoke, scc, calc, mtrans
                )
                
                # Make prediction
                prediction = model.predict(processed_data)
                predicted_class = prediction[0]
                
                # Display results
                st.success("### Prediction Results")
                
                # Show input summary
                st.write("**Your Input Summary:**")
                input_summary = {
                    "Age": age,
                    "Height (m)": height,
                    "Weight (kg)": weight,
                    "Gender": gender,
                    "Family History": family_history,
                    "High Caloric Food": favc,
                    "Vegetable Consumption": fcvc,
                    "Main Meals": ncp,
                    "Water Consumption": ch2o,
                    "Physical Activity": faf,
                    "Tech Device Time": tue,
                    "Smoker": smoke,
                    "Calorie Monitoring": scc,
                    "Between Meal Eating": caec,
                    "Alcohol Consumption": calc,
                    "Transportation": mtrans
                }
                st.json(input_summary)
                
                # Show prediction with mapped label
                readable_label = obesity_levels.get(predicted_class, predicted_class)
                st.write(f"**Predicted Obesity Level:** `{readable_label}`")
                
                # Add some interpretation
                st.info(f"""
                **Obesity Level Interpretation:**
                - {readable_label}
                """)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                if 'prediction' in locals():
                    st.error(f"Raw prediction value: {prediction[0]}")
                st.error("Please check your input values and try again.")

if __name__ == '__main__':
    main()