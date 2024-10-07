import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

from src.exception import Customexception
from src.logger import logging

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for Data tranformation
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )
            cat_pipline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Ctegorical Columns : {categorical_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                    ("cat_pipelines", cat_pipline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise Customexception(e, sys)

        
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            traget_column_name = "math_score"
            numerical_columns = ["writing score", "reading score"]
            
            input_feature_train_df=train_df.drop(columns=[traget_column_name], axis=1)
            target_feature_train_df=train_df[traget_column_name]
            
            input_feature_test_df=test_df.drop(columns=[traget_column_name], axis=1)
            target_feature_test_df=test_df[traget_column_name]
            
            logging_info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)          
            '''
            fit_transform: The magic box learns from the training data and changes it.
            transform: The magic box uses what it learned to change the test data in the same way.
            '''
            
            train_arr = np.c_[                              
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[                              
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            ''' 
            np.c_:
            This is a shorthand for concatenation along the second axis (columns) in NumPy. It is used to combine arrays column-wise.
            ''' 
            logging.info("saved preprocessing object.")
            
            return{
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            }
        except:
            pass