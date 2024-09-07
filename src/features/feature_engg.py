import numpy as np
import pandas as pd
import yaml
import os
from sklearn.feature_extraction.text import CountVectorizer


def load_params(params_path: str) -> float:

    max_features = yaml.safe_load(open(params_path, 'r'))['feature_engg']['max_features']
    
    return max_features


# fetch the data from data/processed
def load_train_data(url):
    
    train_data = pd.read_csv(url)
    
    return train_data


def load_test_data(url):
    
    test_data = pd.read_csv(url)
    
    return test_data

def convert_to_df(train_con_df, test_con_df):
    
    train_df = pd.DataFrame(train_con_df.toarray())
    test_df = pd.DataFrame(test_con_df.toarray())

    return train_df, test_df


def save_data(train_df, test_df):
    
    data_path = os.path.join("data","processed")
    os.makedirs(data_path)
    train_df.to_csv(os.path.join(data_path,"train_bow.csv"))
    test_df.to_csv(os.path.join(data_path,"test_bow.csv"))



def main():
    max_features = load_params("C:\Data Analyst\CampusX Study Material\MLOPS\Practice Modules\session 11/versioning-demo\params.yaml")

    train_data = load_train_data('./data/interim/train_processed.csv')
    test_data = load_test_data('./data/interim/test_processed.csv')

    train_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)
    
    # Splitting Data
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=max_features)

    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    train_df, test_df = convert_to_df(train_con_df=X_train_bow, test_con_df=X_test_bow)

    train_df['label'] = y_train
    test_df['label'] = y_test

    # store the data inside data/features
    save_data(train_df=train_df, test_df=test_df)


if __name__ == '__main__':
    main()