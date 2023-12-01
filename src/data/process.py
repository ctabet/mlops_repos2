import os
import joblib
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ConfigObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    def add_variables(self, categorical_val, continuous_val):
        if not hasattr(self, 'variables'):
            self.variables = ConfigObject({})  # Create a 'variables' attribute if it doesn't exist

        self.variables.categorical_variables = categorical_val
        self.variables.continuous_variables = continuous_val

        # Save back to YAML
        with open("mlops_training_repo/config/main.yaml", "w") as stream:
            yaml.dump(self.to_dict(), stream)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


with open("mlops_training_repo/config/main.yaml", "r") as stream:
    try:
        config_dict = yaml.safe_load(stream)
        config = ConfigObject(config_dict)
    except yaml.YAMLError as exc:
        print(exc)

def get_data(raw_path: str):
    df = pd.read_csv(raw_path)
    return df

def cat_cont_variables(df):
    categorical_val = []
    continuous_val = []
    for column in df.columns:
        if len(df[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    return categorical_val, continuous_val

def preproc_data(df, categorical_val):
    categorical_val.remove('target')
    dataset = pd.get_dummies(df, columns = categorical_val)
    s_sc = StandardScaler()
    col_to_scale = config.raw.Numeric_cols_to_scale
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
    joblib.dump(
        s_sc,
        "C:/charbel tabet/charbel/mlops_repos/mlops_training_repo/data/processed/scaler.pkl",
    )
    return dataset

def split_df(df):
    X = df.drop(config.raw.Label, axis=1)
    y = df[config.raw.Label]
    return train_test_split(X, y.values.ravel(), test_size=0.3, random_state=42)

def process_data():
    df = get_data(config.raw.path)
    categorical_val, continuous_val = cat_cont_variables(df)
    config.add_variables(categorical_val, continuous_val)
    df_processed = preproc_data(df, categorical_val)
    x_train, x_test, y_train, y_test = split_df(df_processed)

    # Accessing directory and file names from the config
    processed_dir = config.processed.dir
    x_train_name = config.processed.x_train.name
    y_train_name = config.processed.y_train.name
    x_test_name = config.processed.x_test.name
    y_test_name = config.processed.y_test.name

    # Constructing the file path
    x_train_path = os.path.join(processed_dir, x_train_name)
    y_train_path = os.path.join(processed_dir, y_train_name)
    x_test_path = os.path.join(processed_dir, x_test_name)
    y_test_path = os.path.join(processed_dir, y_test_name)

    # Save data
    pd.DataFrame(x_train).to_csv(x_train_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(x_test).to_csv(x_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

if __name__ == '__main__':

    process_data(config)