import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle

class Preprocessing:
    def __init__(self, filepath):
        self.filepath= filepath
        self.df= None
        self.label_encoder= LabelEncoder()
        self.label_encoders = {}

    def read_data(self):
        self.df= pd.read_csv(self.filepath)

    def check_null_values(self):
        print("Total null values:\n", self.df.isnull().sum())

    def check_duplicate_data(self):
        print("\nTotal duplicated data before:", self.df.duplicated().sum())

    def remove_duplicate_data(self):
        self.df = self.df.drop_duplicates()
        print("\nTotal duplicated data after:",self.df.duplicated().sum())

    def plot_weight_height(self):
        categories= self.df['NObeyesdad'].unique()
        colors = plt.cm.get_cmap('tab10', len(categories))

        plt.figure(figsize=(8, 6))
        for i, category in enumerate(categories):
            subset = self.df[self.df['NObeyesdad'] == category]
            plt.scatter(subset['Weight'], subset['Height'], color=colors(i), label=category, alpha=0.6)

        plt.xlabel('Weight')
        plt.ylabel('Height')
        plt.title('Distribution of Weight and Height by NObeyesdad')
        plt.legend(title='NObeyesdad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def encoding(self):
        label_enc_cols = ['CAEC', 'CALC']
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        
        for col in label_enc_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        for col in binary_cols:
            self.df[col] = self.df[col].map({'no': 0, 'yes': 1, 'Female': 0, 'Male': 1})
        
        transport_mapping = {
            'Public_Transportation': 0, 
            'Walking': 1, 
            'Automobile': 2, 
            'Motorbike': 3, 
            'Bike': 4
        }
        self.df['MTRANS'] = self.df['MTRANS'].map(transport_mapping)
    
    def define_x_y(self):
        x= self.df.drop(columns=["NObeyesdad"])
        y= self.df["NObeyesdad"]
        return x,y

class Modeling:
    def __init__(self, x, y):
        self.x= x
        self.y= y
        
        self.categorical_cols = []
        self.numerical_cols = []
        self.identify_columns()
        
        self.scaler = RobustScaler()
        
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_pred, self.best_model= [None] * 6
        self.model= RandomForestClassifier(criterion= 'gini', max_depth= 4)
        
    def identify_columns(self):
        for col in self.x.columns:
            if 'int' in str(self.x[col].dtype) or 'float' in str(self.x[col].dtype):
                self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)
        print(f'Categorical Columns: {self.categorical_cols}')
        print(f'Numerical Columns: {self.numerical_cols}')
        
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test= train_test_split(self.x, self.y, test_size= 0.2, random_state= 42)

    def scale_data(self):
        self.x_train[self.numerical_cols] =  self.scaler.fit_transform(self.x_train[self.numerical_cols])
        self.x_test[self.numerical_cols] =  self.scaler.transform(self.x_test[self.numerical_cols])

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        self.y_pred= self.model.predict(self.x_test)
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_pred))

    def hyperparameter_tuning(self):
        parameter = {
            'n_estimators': [100, 200, 300],
            'max_depth': [2, 4, 6, 8],
            'min_samples_leaf': [1, 3, 5]
        }
        model_tune= GridSearchCV(self.model, parameter, cv= 5, verbose=2, scoring= "f1_weighted", n_jobs= -1)
        model_tune.fit(self.x_train, self.y_train)
        self.best_model= model_tune.best_estimator_
        
    def evaluate_best_model(self):
        self.y_pred= self.best_model.predict(self.x_test)
        print("Classification Report for Best Model\n")
        print(classification_report(self.y_test, self.y_pred))

    def model_save(self, filepath):
        with open(filepath, "wb") as f:
          pickle.dump(self.best_model, f)

Preprocessing= Preprocessing("ObesityDataSet_raw_and_data_sinthetic.csv")
Preprocessing.read_data()
Preprocessing.check_null_values()
Preprocessing.check_duplicate_data()
Preprocessing.remove_duplicate_data()
Preprocessing.plot_weight_height()
Preprocessing.encoding()
x, y= Preprocessing.define_x_y()

Modeling= Modeling(x, y)
Modeling.split_data()
Modeling.scale_data()
Modeling.train()
Modeling.evaluate()
Modeling.hyperparameter_tuning()
Modeling.evaluate_best_model()

Modeling.model_save("Obesity_model.pkl")