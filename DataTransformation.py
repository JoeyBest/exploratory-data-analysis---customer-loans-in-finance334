import numpy as np
import pandas as pd
from db_utils import RDSDatabaseConnector

class DataTransform:
    '''
    This class is used to apply transformations to columns within the data.
    ''' 

    def __init__(self, df_info):
        self.df_info = df_info

    def to_interger(self, column):
        '''
        This method converts the datatype of the listed columns to 'Int32'.
        It also fills NULL values with 0.
        '''
        for col in column:
            self.df_info[col] = self.df_info[col].fillna(0).astype('int32')

    def to_boolean(self, column_name):
        '''
        This method transforms data into a boolean using a mask.
        Data that shows 'n' is converted to false and data that shows 'y' is converted to true in column.
        The datatype of the listed columns is converted to 'bool'.
        A interger of the number of unique values is then printed.
        '''
        mask = {'n': False, 'y': True}
        self.df_info[column_name].map(mask)
        self.df_info[column_name] = self.df_info[column_name].astype('bool')
        print(self.df_info[column_name].unique())

    def to_object(self, column):
        '''
        This method converts the datatype of the listed columns to 'object'.
        '''            
        for col in column:
            self.df_info[col] = self.df_info[col].astype(object)
    
    def to_float(self, column):
        '''
        This method converts the datatype of the listed columns to 'float64'.        
        '''
        for col in column:
            self.df_info[col] = self.df_info[col].astype('float64')

    def to_rounded_float(self, column, decimal_places):
        '''
        This method rounds the floats in the listed columns to the users selected number of decimal places.        
        '''
        self.df_info[column] = self.df_info[column].apply(lambda x: round(x, decimal_places))

    def to_category(self, column):
        '''
        This method converts the datatype of the listed columns to 'category'.
        '''
        for col in column:
            self.df_info[col] = self.df_info[col].astype('category')

    def to_numerical_column(self, column):
        '''
        This method changes the values of the listed columns to a numerical value that pandas can read.
        '''    
        for col in column:
            pd.to_numeric(self.df_info[col])

    def extract_integer_from_string(self, column):
        '''
        This method is used to extract integers that are contained within strings in columns.
        It loops through the specified column extracting one or more (\d+) intergers orginally formatted as a string
        '''
        for col in column:
            self.df_info[col] = self.df_info[col].str.extract('(\d+)')

    def strings_to_dates(self, column):
        '''
        This method is used to convert dates within a string into a period format date as the loan database only includes month and year.
        '''    
        for col in column:
            self.df_info[col] = pd.to_datetime(self.df_info[col], errors='coerce', format="%b-%Y")#.dt.to_period('M') - method converts the datetime to a period (M) which is a date that contains only the month and year since this is the resolution of the data provided.

    def replace_string_text(self, column_name, original_string: str, new_string: str):
        '''
        This method is used to replace string text with a newly created string text.
        Parameters:
            column_name: The name of the column to which this method will be applied.
            original_string (str): the string that will be replaced.
            new_string (str): the string that will replace the original_string.
        '''
        self.df_info[column_name].replace(original_string, new_string)

    def rename(self, column_name, new_column_name):
        '''
        This method is used to replace a columns name with a newly created column name.
        Parameters:
            column_name: the column name that will be replaced.
            new_column_name: the column name that will replace the original_string.
        '''
        self.df_info.rename(columns={column_name: new_column_name})

    def drop_column(self, column):
        '''
        This method removes the listed columns from the dataframe.
        '''
        for col in column:
            self.df_info.drop(col, axis=1, inplace=True)

    def remove_null_rows(self, column_name):
        '''
        This method is used to remove rows within the dataframe where data points from a specified column are null.
        '''
        self.df_info.dropna(subset=column_name, inplace=True)

    def save_transformed_data(self, filename='full_loan_data.csv'):
        '''
        This method savves the dataframe to the current device and working directory as a CSV file called 'transformed_data.csv'.
        '''
        self.df_info.to_csv(filename, index=False)

if __name__ ==  "__main__":
    table_of_loans = pd.read_csv('eda.csv')
    Transform = DataTransform(table_of_loans)

    Transform.to_boolean('payment_plan')
    # transforms n to false and y to true in the payment plan column
    # prints [True] as the only unique value in the dataframe column

    to_object_columns = ['id', 'member_id', 'policy_code']
    Transform.to_object(to_object_columns)
    
    convert_categories = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'employment_length']
    Transform.to_category(convert_categories)
    # transforms column values to catagories

    string_month_and_year = ['last_credit_pull_date', 'next_payment_date', 'last_payment_date', 'earliest_credit_line', 'issue_date']
    Transform.strings_to_dates(string_month_and_year)
    # transforms strings to a datetime format MonthYear

    string_numbers = ['term']
    Transform.extract_integer_from_string(string_numbers)

    numericals = ['term', 'mths_since_last_record', 'mths_since_last_major_derog', 'mths_since_last_delinq', 'mths_since_last_record']
    Transform.to_numerical_column(numericals)

    drop_cols = ['funded_amount', 'application_type', 'policy_code', 'out_prncp_inv', 'total_payment_inv']
    Transform.drop_column(drop_cols)
    # funded_amount is missing some data but contains the same data as funded_amount_inv, so we can drop it
    # out_prncp_inv contains the same data as out_prncp, so we can drop it
    # total_payment_inv contains the same data as total_payment, so we can drop it
    # application_type and policy_code are the same for everyone and doesn't provide us with much info
    
    int_numbers = ['loan_amount', 'funded_amount_inv', 'annual_inc', 'term', 'open_accounts', 'total_accounts', 'collections_12_mths_ex_med', 'mths_since_last_delinq', 'mths_since_last_major_derog']
    Transform.to_interger(int_numbers)
    
    Transform.to_rounded_float('collection_recovery_fee', 2)

    print(Transform.df_info.dtypes)
    print(Transform.df_info.info())

    # saves a new CSV of the df called 'full_loan_data.csv'
    Transform.save_transformed_data('full_loan_data.csv')


    