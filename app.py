import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import tempfile
import xgboost
import sklearn.ensemble

# Initialize list to store temporary files for delayed cleanup
temp_files = []

# Set title
st.header("Streamlit Machine Learning App")

# Get user input
race = st.selectbox("Race", ("1", "2", "3"))
sex = st.selectbox("Sex", ("1", "2"))
site = st.selectbox("Site", ("1", "2", "3", "4", "5", "6"))
grade = st.selectbox("Grade", ("1", "2", "3", "4"))
T_stage = st.selectbox("T Stage", ("1", "2", "3", "4", "5", "6"))
N_stage = st.selectbox("N Stage", ("1", "2", "3"))
surgery = st.selectbox("Surgery", ("0", "1"))
radiation = st.selectbox("Radiation", ("0", "1"))
chemotherapy = st.selectbox("Chemotherapy", ("0", "1"))
marital = st.selectbox("Marital Status", ("1", "2"))
residence = st.selectbox("Residence", ("1", "2"))
income = st.selectbox("Income", ("1", "2"))
size = st.number_input("Tumor Size (mm)", min_value=0.0, step=0.1)
age = st.number_input("Age (years)", min_value=0, step=1)

# Submit button
if st.button("Submit"):
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")


        if not os.path.exists(model_path):
            st.error("Model file 'model.pkl' not found at D:\\Code\\streamlit")
        else:
            # Load model
            model = joblib.load(model_path)
            st.write(f"Model type: {type(model)}")  # Debug model type

            # Extract best_estimator_ if GridSearchCV
            if hasattr(model, "best_estimator_"):
                best_model = model.best_estimator_
            else:
                best_model = model

            # Build input features
            feature_names = [
                "race", "sex", "site", "grade", "T_stage", "N_stage",
                "surgery", "radiation", "chemotherapy", "marital",
                "residence", "income", "size", "age"
            ]
            feature_values = [
                int(race), int(sex), int(site), int(grade), int(T_stage),
                int(N_stage), int(surgery), int(radiation), int(chemotherapy),
                int(marital), int(residence), int(income), float(size), float(age)
            ]
            X = pd.DataFrame([feature_values], columns=feature_names)

            # Prediction
            prediction = best_model.predict(X)[0]
            st.success(f"Prediction result: {prediction}")

            # Probability prediction (if supported)
            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(X)[0]
                probability = proba[int(prediction)] * 100
                st.info(f"Probability for class {prediction}: {probability:.1f}%")
                st.info(f"Probability for class 0: {proba[0] * 100:.1f}%")  # Show class 0 probability
                st.info(f"Probability for class 1: {proba[1] * 100:.1f}%")  # Show class 1 probability

                # Provide advice
                if int(prediction) == 1:
                    advice = (
                        f"Based on the model prediction, you are in a high-risk group (probability: {probability:.1f}%).\n\n"
                        "It is recommended to seek medical attention promptly for further examination and to develop an appropriate treatment plan."
                    )
                else:
                    advice = (
                        f"Based on the model prediction, you are in a low-risk group (probability: {probability:.1f}%).\n\n"
                        "Although the risk is low, regular check-ups and a healthy lifestyle are recommended to maintain good health."
                    )
                st.write(advice)

            # SHAP model explanation (using waterfall plot, fixed to class 0)
            try:
                # Select appropriate SHAP explainer based on model type
                if isinstance(best_model, (xgboost.XGBClassifier, sklearn.ensemble.RandomForestClassifier)):
                    explainer = shap.TreeExplainer(best_model)
                else:
                    explainer = shap.KernelExplainer(best_model.predict, X)

                shap_values = explainer(X)
                st.write(f"SHAP values shape: {shap_values.shape}")  # Debug SHAP values shape

                st.subheader("Model Explanation (SHAP Waterfall Plot - Class 0)")

                # Select SHAP values for class 0
                shap_values_for_plot = shap_values[0][:, 0]  # Fixed to class 0

                # Save SHAP waterfall plot to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    shap.plots.waterfall(shap_values_for_plot, show=False)
                    plt.savefig(tmp_file.name, bbox_inches='tight', dpi=300)
                    plt.close()
                    if os.path.exists(tmp_file.name):
                        st.image(tmp_file.name, caption="SHAP Waterfall Plot (Class 0)")
                        # Store temporary file path for delayed cleanup
                        temp_files.append(tmp_file.name)
                    else:
                        st.error("SHAP plot file was not generated.")

            except Exception as e:
                st.error(f"Failed to generate SHAP plot, model may not be supported: {e}")
                import traceback
                st.text(traceback.format_exc())  # Display full stack trace for debugging

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.text(traceback.format_exc())  # Display full stack trace for debugging

# Clean up temporary files (attempt at program end)
for temp_file in temp_files:
    try:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    except PermissionError:
        st.warning(f"Unable to delete temporary file {temp_file}, it may be in use.")