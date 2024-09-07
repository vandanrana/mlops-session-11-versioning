import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml

# Load Params
def load_params(params_path: str) -> float:
    
    params_file = yaml.safe_load(open(params_path,'r'))
    n_estimators = params_file['model_building']['n_estimators']
    learning_rate = params_file['model_building']['learning_rate']

    return n_estimators, learning_rate

# Load Data
def load_data(url: str) -> pd.DataFrame:
    
    train_data = pd.read_csv(url)
    
    return train_data



# Save Model
def save_model(model_object) -> None:
    pickle.dump(model_object, open('models/model.pkl','wb')) 


# Main Function
def main():
    # load parameters
    n_estimators, learning_rate = load_params("C:\Data Analyst\CampusX Study Material\MLOPS\Practice Modules\session 11/versioning-demo\params.yaml")

    # Loading Training Data
    train_data = load_data('./data/processed/train_bow.csv')

    # Splitting Data
    X_train = train_data.iloc[:,0:-1].values
    y_train = train_data.iloc[:,-1].values
    
    # Training Data
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)

    # Saving the Model
    save_model(model_object=clf)

if __name__ == '__main__':
    main()