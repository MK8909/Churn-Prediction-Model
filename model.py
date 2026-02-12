import pickle
import pandas as pd
import streamlit as st

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open("customer_churn_model_new.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders_new.pkl", "rb") as f:
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


























