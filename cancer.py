import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st
import requests
import zipfile
import io

# Download and extract the dataset
url = 'https://archive.ics.uci.edu/static/public/225/ilpd+indian+liver+patient+dataset.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# Assuming the CSV file is named 'Indian Liver Patient Dataset (ILPD).csv'
file_name = 'Indian Liver Patient Dataset (ILPD).csv'

# Load the dataset from the extracted file
data = pd.read_csv(file_name, sep=',', header=None)

# Check the first few rows to ensure it is read correctly
print(data.head())

# Assign column names based on the dataset's features
data.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins',
                'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']

# Display the first few rows and basic statistics
st.write("### Dataset Preview")
st.write(data.head())
st.write("### Dataset Statistics")
st.write(data.describe())

# Handle missing values by dropping rows with missing data
data = data.dropna()

# Encode categorical features (Gender)
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Visualizations (Exploratory Data Analysis)
st.write("### Age Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Age'], ax=ax)
st.pyplot(fig)

st.write("### Total Bilirubin by Dataset")
fig, ax = plt.subplots()
sns.boxplot(x='Dataset', y='Total_Bilirubin', data=data, ax=ax)
st.pyplot(fig)

# Feature selection and Train-test split
X = data.drop('Dataset', axis=1)
y = data['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'liver_cancer_model.pkl')

# Load the trained model
model = joblib.load('liver_cancer_model.pkl')

# Model Prediction Function
def preprocess_input(data):
    data = scaler.transform([data])
    return data

# Streamlit UI
st.title('Liver Cancer Detection')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0, value=1.0)
direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0, value=0.1)
alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0, value=100)
alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0, value=20)
aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0, value=30)
total_proteins = st.number_input('Total Proteins', min_value=0.0, value=6.0)
albumin = st.number_input('Albumin', min_value=0.0, value=3.0)
albumin_and_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, value=1.0)

# Gender encoding
gender_encoded = 1 if gender == 'Male' else 0

# Prediction
if st.button('Predict'):
    input_data = [age, gender_encoded, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                  alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                  albumin, albumin_and_globulin_ratio]
    
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    prediction_label = 'Diseased' if prediction[0] == 1 else 'Not Diseased'

    st.write(f'Prediction: {prediction_label}')

