import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

logger = logging.getLogger('data_ingestion_logger')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# params

def load_params(params_path: str) -> float:

    test_size = yaml.safe_load(open(params_path, 'r'))['data_ingestion']['test_size']
    logger.debug('Test Size retrieved')
    return test_size


# Load Data
def load_data(url: str) -> pd.DataFrame:

    df = pd.read_csv(url)
    logger.info('Data Loaded')
    return df


# Process Data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df.drop(columns=['tweet_id'], inplace=True)
    final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
    final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
    logger.info('Data Processed Successfully')
    return final_df


# Save Data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    data_path = os.path.join("data", "raw")
    os.makedirs(data_path)
    train_data.to_csv(os.path.join(data_path, 'train.csv'))
    test_data.to_csv(os.path.join(data_path, 'test.csv'))
    logger.info('Data Saved Successfully')

# Main Function
def main() -> None:

    test_size = load_params("C:\Data Analyst\CampusX Study Material\MLOPS\Practice Modules\session 11/versioning-demo\params.yaml")

    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

    final_df = process_data(df)

    # Train Test Split
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    save_data(train_data, test_data)

if __name__ == '__main__':
    main()


