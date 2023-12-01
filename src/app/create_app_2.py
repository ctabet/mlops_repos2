import json
import yaml
import joblib
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

class ConfigObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
                return cls(config_dict)
            except yaml.YAMLError as exc:
                print(exc)

config = ConfigObject.from_yaml("C:/charbel tabet/charbel/mlops_repos/mlops_training_repo/config/main.yaml")

# with open("C:/charbel tabet/charbel/mlops_repos/mlops_training_repo/config/main.yaml", "r") as stream:
#     try:
#         config_dict = yaml.safe_load(stream)
#         config = ConfigObject(config_dict)
#     except yaml.YAMLError as exc:
#         print(exc)
def preproc_data(df):
    categorical_val = getattr(config.variables, "categorical_variables", [])
    col_to_scale = getattr(config.variables, "continuous_variables", [])
    # Check if 'target' is in categorical_val list before removal
    if 'target' in categorical_val:
        categorical_val.remove('target')  # Remove 'target' from categorical columns list

    # Check if 'target' key is in the dictionary before removal
    if 'target' in df:
        df.pop('target')  # Remove 'target' column from the dictionary

    # Restructure the dictionary to contain lists as values
    for key in df:
        df[key] = [df[key]]  # Convert scalar value to a list containing the value

    # Create a DataFrame from the updated dictionary
    dataset = pd.DataFrame(df)

    # Perform one-hot encoding for categorical columns
    dataset = pd.get_dummies(dataset, columns=categorical_val)
    # Check if the columns exist after one-hot encoding
    existing_cols = [col for col in col_to_scale if col in dataset.columns]
    s_sc = joblib.load("C:/charbel tabet/charbel/mlops_repos/mlops_training_repo/data/processed/scaler.pkl")
    if existing_cols:
        custom_means = s_sc.mean_
        custom_scales = s_sc.scale_
        # Transform the data using the custom mean and scale values
        scaled_data = (dataset[existing_cols] - custom_means) / custom_scales
        common_columns = dataset.columns.intersection(scaled_data.columns)        # Scale numeric columns using StandardScaler
        dataset[common_columns] = scaled_data[common_columns]
    else:
        print("Columns for scaling not found after one-hot encoding.")
    return dataset.to_json(orient='records')

def preproc_data_2(df, categorical_val):
    if "target" in categorical_val:
        categorical_val.remove("target")

    dataset = pd.get_dummies(df, columns=categorical_val)
    s_sc = StandardScaler()
    col_to_scale = config.raw.Numeric_cols_to_scale
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

    # Convert DataFrame to a list of dictionaries
    records = dataset.to_dict(orient='records')
    return records
def get_inputs():
    """Get inputs from users on streamlit"""
    st.title("Predict Heart Disease")

    data = {}

    data["age"] = st.number_input("Current Age", min_value=15, step=1, value=20)
    data["sex"] = st.selectbox("Gender", options=[1, 0], help="Male: 1: Female: 2",)
    data["cp"] = st.selectbox("CP", options=[0, 1, 2, 3])
    data["trestbps"] = st.number_input("trestbps", min_value=94, step=1, value=150)
    data["chol"] = st.number_input("chol", min_value=126, step=1, value=500)
    data["fbs"] = st.selectbox("fbs", options=[1, 0])
    data["restecg"] = st.selectbox("restecg", options=[1, 0])
    data["thalach"] = st.number_input("thalach", min_value=70, step=1, value=150)
    data["exang"] = st.selectbox("exang", options=[1, 0])
    data["oldpeak"] = st.number_input("oldpeak", min_value=0.0, step=0.1, value=1.6)
    data["slope"] = st.selectbox("slope", options=[0, 1, 2])
    data["ca"] = st.selectbox("ca", options=[0, 1, 2])
    data["thal"] = st.selectbox("thal", options=[0, 1, 2, 3])
    return data


def write_predictions(data: dict):
    if st.button("Will the patient have heart disease?"):
        data_json = json.dumps(data)
        response = requests.post(
            "https://patient-predict-1.herokuapp.com/predict",
            headers={"content-type": "application/json"},
            data=data_json,
        )
        st.write(data_json)

        if response.status_code == 200:
            prediction = response.json()
            st.write(
                "Prediction received:", prediction
            )  # Check what the full response looks like

            # Extracting the prediction (modify this based on the actual response structure)
            if prediction.get("prediction") == "0":
                st.write("This patient is predicted to not have heart disease.")
            else:
                st.write("This patient is predicted to have heart disease.")
        else:
            st.write("Error:", response.text)  # Display any errors from the API

def cat_cont_variables(df):
    categorical_val = []
    continuous_val = []
    for column in df.columns:
        if len(df[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    return categorical_val, continuous_val

def main():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        categorical_val, _ = cat_cont_variables(df)
        dataset = preproc_data_2(df, categorical_val)
        write_predictions(dataset)
    else:
        data = get_inputs()
        data = preproc_data(data)
        write_predictions(data)

if __name__ == "__main__":
    main()
