import numpy as np
import pandas as pd

class DataFrameInfo:

    def __init__(self, df_info):
        self.df_info = df_info

    def check_column_datatypes(self): 
        return self.df_info.dtypes

    def extract_statistical_values(self, columns):
        # Populate list of rows
        for col in columns:
            print(f'Statistics for Col: {col}')
            return self.df_info[col].describe()

    def count_unique_categories(self, columns):
        self.columns = self.df_info.select_dtypes(include=['category']).columns
        df_unique = self.df_info[columns[:]]
        return df_unique.nunique()

    def shape_of_dataframe(self):
        print(f'Shape of DataFrame: [{self.df_info.shape[0]} rows x {self.df_info.shape[1]} columns]\n')

    def num_of_nulls(self):
        cols = self.df_info.columns
        # Define an empty list of null column rows
        null_coulmn_rows = []
        # Populate list of null column rows
        for col in cols:
            # if self.df[col].isnull().sum() > 0:
            null_coulmn_rows.append([col, self.df_info[col].count(), 100*(self.df_info[col].isnull().sum()/len(self.df_info))])
        
        # Convert the list into dataframe rows
        data = pd.DataFrame(null_coulmn_rows)
        # Add columns headers
        data.columns = ['column', 'count', '% null count']  
        return data


if __name__ == "__main__":
    table_of_loans = pd.read_csv('eda.csv')
    df_df = DataFrameInfo(table_of_loans)

    print(df_df.check_column_datatypes())
    print(df_df.extract_statistical_values(['loan_amount']))
    print(f"The no. of Unique Items in {df_df.count_unique_categories(['grade'])}")
    print(df_df.shape_of_dataframe())
    print(df_df.num_of_nulls())