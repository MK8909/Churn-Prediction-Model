import pickle
import pandas as pd
import streamlit as st

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data['rfc'], encoders, model_data['feature_names']

loaded_model, encoders, feature_names = load_artifacts()

def churn_prediction(input_data):
    """Function to predict customer churn"""
    input_data_df = pd.DataFrame([input_data])

    for column in input_data_df.columns:
        if column in encoders:
            input_data_df[column] = encoders[column].transform(input_data_df[column])

    input_data_df = input_data_df[feature_names]

    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    return {
        'prediction': 'Churn' if prediction[0] == 1 else 'No Churn',
        'probability': {
            'No Churn': float(pred_prob[0][0]),
            'Churn': float(pred_prob[0][1])
        }
    }

def main():
    st.title("📊 Customer Churn Prediction")
    st.write("Enter customer details to predict churn probability")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])

        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        col3, col4 = st.columns(2)
        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        with col4:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=29.85)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=29.85)

        submitted = st.form_submit_button("Predict Churn")

        if submitted:
            input_data = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            result = churn_prediction(input_data)

            st.subheader("Prediction Results")
            st.metric("Churn Prediction", result['prediction'])
            st.progress(result['probability']['Churn'])
            st.write(f"🔴 Probability of Churn: {result['probability']['Churn']:.1%}")
            st.write(f"🟢 Probability of No Churn: {result['probability']['No Churn']:.1%}")

if __name__ == "__main__":
    main()

























'''import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# 1. Load the model file

with open("customer_churn_model.pkl", "rb") as f:
  model_data = pickle.load(f)
# 2. Check what's in the loaded file
print("Keys in model file:", model_data.keys())  # This shows it has 'rfc' and 'feature_names'

# 3. Extract the actual model - using the correct key 'rfc' instead of 'classifier'
loaded_model = model_data['rfc']

# 4. Load encoders - make sure the path is correct

with open("encoders.pkl","rb") as f:    
    encoders = pickle.load(f)

# 5. Prepare input data
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

input_data_df = pd.DataFrame([input_data])

# 6. Transform categorical features - ensure all columns exist in encoders
for column in input_data_df.columns:
    if column in encoders:
        input_data_df[column] = encoders[column].transform(input_data_df[column])

# 7. Reorder columns to match training data if needed
if 'feature_names' in model_data:
    input_data_df = input_data_df[model_data['feature_names']]

# 8. Make prediction
try:
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)
    
    print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    print(f"Prediction Probability: {pred_prob[0]}")
except Exception as e:
    print("Prediction failed:", e)
    print("Model type:", type(loaded_model))
    print("Does model have predict method?", hasattr(loaded_model, 'predict'))'''







'''import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle


####DATA LOA
data_path ="WA_Fn-UseC_-Telco-Customer-Churn.csv"


df = pd.read_csv(data_path, header=0)  # Add header names later if needed




# Preview

dfcopy=df.copy()
df.head(5)

# Create a proper copy of the DataFrame (note the parentheses)
df = dfcopy

# Check the shape of the DataFrame
print(df.shape)

# Set pandas to display all columns (you need to specify None or a number)


# Display the first 5 rows
df.head(5)
df.info()
df=dfcopy

df.drop('customerID', axis=1, inplace=True)  # Modifies df directly, returns None
df.head(3)  # Now works because df was modified in place


## printing unique values of a column
print(df['gender'].unique())



df.columns

numerical_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges']

for col in df.columns:
 if col not in numerical_columns:
  print(col,df[col].unique())
  print("-"*50)

df.isnull().sum() ## to check how mnay null or missing


## Option 1: Handle errors during conversion
#df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype('float')

## Option 2: Replace blanks/whitespace first
df['TotalCharges'] = df['TotalCharges'].str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype('float')


print(df['TotalCharges'].dtype)  # Should show 'float64'
print(df['TotalCharges'].head())  

def plot_histogram(df, column_name):
    plt.figure(figsize=(5, 3))
    # Plot histogram with KDE
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")

    # Calculate mean and median
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()

    # Add vertical lines for mean and median
    plt.axvline(col_mean, color="red", linestyle="dashed", label="Mean")
    plt.axvline(col_median, color="green", linestyle="dashed", label="Median")

    # Add legend
    plt.legend()
    plt.show()

object_cols=df.select_dtypes(include="object").columns
object_cols

object_cols=df.select_dtypes(include="object").columns.to_list()
object_cols=["SeniorCitizen"]+object_cols

for col in object_cols:
  plt.figure(figsize=(5, 3))
  sns.countplot(x=df[col])
  plt.title(f"Distribution of {col}")
  plt.show()

df["Churn"]=df["Churn"].replace({"Yes":1,"No":0})
df.head(3)

encoders={}

#apply label encoding and store the encoders

for columns in object_cols:
  label_encoder=LabelEncoder()
  df[columns]=label_encoder.fit_transform(df[columns])
  encoders[columns]=label_encoder

## save encoder to pickle file

with open("encoders.pkl","wb") as f:
  pickle.dump(encoders,f)


x=df.drop(columns=["Churn"])
y=df["Churn"]

print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


smote=SMOTE(random_state=42)
# Drop rows with any missing values
x_train_clean = x_train.dropna()
y_train_clean = y_train[x_train.index.isin(x_train_clean.index)]

x_train=x_train_clean
y_train=y_train_clean
# Apply SMOTE
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

from imblearn.over_sampling import SMOTE
print("Resampled features shape:", x_train_smote.shape)
print("Resampled target shape:", y_train_smote.shape)

models={
    "DecisionTree":DecisionTreeClassifier(random_state=42),
    "RandomForest":RandomForestClassifier(random_state=4),
    "XGBoost":XGBClassifier(random_state=4)
}


from sklearn.model_selection import cross_val_score

## dictionary to store cross-validation results
cv_scores = {}

## perform s-fold cross validation for each model
for model_name, model in models.items():
    print(f"Training {model_name} with default parameter")
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f}")





rfc=RandomForestClassifier(random_state=42)
rfc.fit(x_train_smote,y_train_smote)


y_test_pred=rfc.predict(x_test)
print(y_test.value_counts())  ## from confusion matrix-->outof 1036 zeros->873 correctly identified as zero and out of 373 1's-.273 correctly identifired
print("Accuracy score:\n",accuracy_score(y_test,y_test_pred))
print("confusion matrix:\n",confusion_matrix(y_test,y_test_pred))
print("classification report:\n",classification_report(y_test,y_test_pred))


model_data={"rfc":rfc,"feature_names":x.columns.tolist()}


with open("customer_churn_model.pkl","wb") as f:
  pickle.dump(model_data,f)


##load saved model and feature names
with open("customer_churn_model.pkl","rb") as f:
  model_data=pickle.load(f)

loaded_model=model_data["rfc"]
feature_names=model_data["feature_names"]









import pandas as pd
import pickle

input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Only transform columns that exist in both DataFrames
for column in input_data_df.columns:
    if column in encoders:
        input_data_df[column] = encoders[column].transform(input_data_df[column])

print(input_data_df.head())



##make prediction

prediction=loaded_model.predict(input_data_df)
print(prediction)

pred_prob=loaded_model.predict_proba(input_data_df)
print(pred_prob)

##result

print(f"Prediction:{'Churn' if prediction[0]==1  else 'No Churn' }")
print(f"Prediction Probablity: {pred_prob}")








import pandas as pd
import pickle

input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Only transform columns that exist in both DataFrames
for column in input_data_df.columns:
    if column in encoders:
        input_data_df[column] = encoders[column].transform(input_data_df[column])

print(input_data_df.head())

##make prediction

prediction=loaded_model.predict(input_data_df)
print(prediction)

pred_prob=loaded_model.predict_proba(input_data_df)
print(pred_prob)

##result

print(f"Prediction:{'Churn' if prediction[0]==1  else 'No Churn' }")
print(f"Prediction Probablity: {pred_prob}")








import pickle
import pandas as pd
import streamlit as st

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data['rfc'], encoders

loaded_model, encoders = load_artifacts()

def churn_prediction(input_data):
    """Function to predict customer churn"""
    input_data_df = pd.DataFrame([input_data])
    
    # Transform categorical features
    for column in input_data_df.columns:
        if column in encoders:
            input_data_df[column] = encoders[column].transform(input_data_df[column])
    
    # Make prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)
    
    return {
        'prediction': 'Churn' if prediction[0] == 1 else 'No Churn',
        'probability': {
            'No Churn': float(pred_prob[0][0]),
            'Churn': float(pred_prob[0][1])
        }
    }

def main():
    st.title("Customer Churn Prediction")
    st.write("Enter customer details to predict churn probability")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            
        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        col3, col4 = st.columns(2)
        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        with col4:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=29.85)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=29.85)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            # Prepare input data
            input_data = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }
            
            # Make prediction
            result = churn_prediction(input_data)
            
            # Display results
            st.subheader("Prediction Results")
            st.metric("Churn Prediction", result['prediction'])
            
            st.progress(result['probability']['Churn'])
            st.write(f"Probability of Churn: {result['probability']['Churn']:.1%}")
            st.write(f"Probability of No Churn: {result['probability']['No Churn']:.1%}")



if __name__ == "__main__":
    main()'''

