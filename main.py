from mlops_training_repo.src.data.process import process_data
from mlops_training_repo.src.models.train_evaluate import *


if __name__ == '__main__':
    process_data()
    train_evaluate()