import os
import joblib
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ConfigObject:
    def __init__(self, d):
        self.variables = None
        self.processed = None
        self.raw = None
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    def add_variables(self, categorical_val, continuous_val, root_path):
        if not hasattr(self, "variables"):
            self.variables = ConfigObject(
                {}
            )  # Create a 'variables' attribute if it doesn't exist

        self.variables.categorical_variables = categorical_val
        self.variables.continuous_variables = continuous_val
        self.variables.root_path = root_path

        # Save back to YAML
        with open(root_path + "/config/main.yaml", "w") as cf:
            yaml.dump(self.to_dict(), cf)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

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

def preproc_data(df, categorical_val,config):
    categorical_val.remove("target")
    dataset = pd.get_dummies(df, columns=categorical_val)
    s_sc = StandardScaler()
    col_to_scale = config.raw.Numeric_cols_to_scale
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
    joblib.dump(
        s_sc,
        config.variables.root_path + "/data/processed/scaler.pkl",
    )
    return dataset


def split_df(df,config):
    X = df.drop(config.raw.Label, axis=1)
    y = df[config.raw.Label]
    return train_test_split(X, y.values.ravel(), test_size=0.3, random_state=42)

def replace_placeholders(config_dict, root_path):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            replace_placeholders(value, root_path)
        elif isinstance(value, str) and "${root_path}" in value:
            config_dict[key] = value.replace("${root_path}", root_path)

def process_data():
    # Define default paths
    local_root_path = "C:/charbel tabet/charbel/mlops_repos/mlops_training_repo"
    github_actions_root_path = "mlops_training_repo"

    # Check if running in GitHub Actions or local environment
    is_github_actions = os.getenv("GITHUB_ACTIONS")

    # Access the appropriate root path based on the environment
    root_path = github_actions_root_path if is_github_actions else local_root_path
    # Load the YAML file
    with open(root_path + "/config/main.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Replace placeholders with the root path
    replace_placeholders(config, root_path)

    config = ConfigObject(config)
    config.add_variables(None, None, root_path)
    df = get_data(config.raw.path)
    categorical_val, continuous_val = cat_cont_variables(df)
    config.add_variables(categorical_val, continuous_val, root_path)
    df_processed = preproc_data(df, categorical_val, config)
    x_train, x_test, y_train, y_test = split_df(df_processed, config)
    data_frames = {
        'x_train': pd.DataFrame(x_train),
        'y_train': pd.DataFrame(y_train),
        'x_test': pd.DataFrame(x_test),
        'y_test': pd.DataFrame(y_test)
    }

    # Save data frames
    for name, df in data_frames.items():
        file_name = getattr(config.processed, name).name
        file_path = os.path.join(config.processed.dir, file_name)
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    process_data()
