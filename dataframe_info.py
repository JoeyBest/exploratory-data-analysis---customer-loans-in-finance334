import numpy as np
import pandas as pd


class DataFrameInfo:
    '''
    This class is used to dive into the a dataframe and begin to understand the information it contains.
    eda.csv being the dataframe that we are going to be loooking into.
    '''

    def __init__(self, df_info):
        '''
        This method is used to initialise this instance of the DataFrameInfo class.

        Parameters:
        ----------
        df_info: list
            loan payments information in a pandas dataframe list
        '''
        self.df_info = df_info

    def check_column_datatypes(self): 
        '''
        This method is used to check the column types of the df.

        Returns:
        ----------
        data-type
            List of of all the data types of each column.
        '''
        return self.df_info.dtypes

    def extract_statistical_values(self, columns):
        '''
        This method is used to loop through a specified column to provide a print of the statistics that .describe() gives us.
        
        Parameters:
        ----------
        columns: int
            The integer column value within the table.

        Returns:
        ----------
            Returns the statistics (count, mean, std, min, LQR, median, UQR and max) of the values in the column provided.
        '''
        for col in columns:
            print(f'Statistics for Col: {col}')
            return self.df_info[col].describe()

    def count_unique_categories(self, columns):
        '''
        This method is used to count the number of distinct/unique values within a column.
        
        Parameters:
        ----------
        columns: int
            The column value within the table.
        columns: str
            The column value within the table.

        Returns:
        ----------
        int
            The number of values that are unique/distinct, including the number of unique categorical values.
        '''
        self.columns = self.df_info.select_dtypes(include=['category']).columns
        df_unique = self.df_info[columns[:]]
        return df_unique.nunique()

    def shape_of_dataframe(self):
        '''
        This method is used to provide an insight to the shape of the df.
        For example, how many rows and columns there are.

        Returns:
        ----------
        tuple: 
            The number of rows and columns within the DataFrame.
        '''
        print(f'Shape of DataFrame: [{self.df_info.shape[0]} rows x {self.df_info.shape[1]} columns]\n')

    def num_of_nulls(self):
        '''
        This method is used to count the number of Null or NaN values there are in the df.
        It the prints these into a pandas chart, with the headings: column, count and %null count.

        Returns:
        ----------
        str
            The name of each column within the data and placed in a table.
        float
            Provides the percentage of the null count in the column.
        int
            Provides the number of values in the column that aren't null or NaN.
        '''
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

    # specific ['columns'] can be changed to any you make like
    print(df_df.check_column_datatypes())
    print(df_df.extract_statistical_values(['loan_amount']))
    print(f"The no. of Unique Items in {df_df.count_unique_categories(['grade'])}")
    print(df_df.shape_of_dataframe())
    print(df_df.num_of_nulls())