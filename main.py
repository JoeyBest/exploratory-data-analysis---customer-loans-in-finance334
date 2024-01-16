# %%
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import yaml
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
from sqlalchemy import create_engine
from sqlalchemy import inspect

# %%
def credentials_reader():
    '''
    This function is used to extract the credentials from yaml to a dictionary to establish connection with the RDS.
    '''
    with open('credentials.yaml', 'r') as file:
        credentials = yaml.safe_load(file)
        return credentials


class RDSDatabaseConnector:
    '''
    This class is used to establish a connection with the AiCore RDS containing loan payments information.
    
    Credentials being the dictionary containing the 'Host', 'Password', 'User', 'Database' and 'Port' required for the sqlalchemy to establish a connection with the RDS
    '''

    def __init__(self, credentials):
        '''
        This method is used to initialise this instance of the RDSDatabaseConnector class.
        '''
        self.credentials = credentials
        
    def initiate_engine(self):
        '''
        This method is used to create the SQLAlchemy engine which is used to connect to the AiCore RDS.
        '''
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'
        self.HOST = 'eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com'
        self.USER = 'loansanalyst'
        self.PASSWORD = 'EDAloananalyst'
        self.DATABASE = 'payments'
        self.PORT = 5432

        self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}")
        self.engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        
        inspector = inspect(self.engine)
        self.table_inspector = inspector.get_table_names()
        print(self.table_inspector)

    def database_to_dataframe(self):
        '''
        This method connects to the RDS in order to extract the 'loan_payments' table into a pandas dataframe.
        '''
        ######### maybe add with self.engine.connect() as connection: (only if it doesn't work like this)
        self.loans = pd.read_sql_table('loan_payments', self.engine)
        return self.loans.head(10), self.loans.tail(10)
        
    # Saves the data to your current pathway as eda.csv
    def saves_data_locally(self):
        '''
        This method savves the dataframe to the current device and working directory as a CSV file called 'eda.csv'.
        '''
        self.loans.to_csv('eda.csv', sep=',', index=False, encoding='utf-8')

    def load_localdata_to_dataframe(self):
        '''
        This method uses the saved CSV file to load the data into a pandas dataframe.
        '''
        # Defining the columns to read
        usecols = ["id", "member_id","loan_amount", "funded_amount", "funded_amount_inv", "term", "int_rate", "instalment", "grade", "sub_grade", "employment_length", "home_ownership", "annual_inc", "verification_status", "issue_date", "loan_status", "payment_plan", "purpose", "dti", "delinq_2yrs", "earliest_credit_line", "inq_last_6mths", "mths_since_last_record", "open_accounts", "total_accounts", "out_prncp", "out_prncp_inv", "total_payment", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_date", "last_payment_amount", "next_payment_date", "last_credit_pull_date", "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code", "application_type"]
    # Read data with subset of columns
        loan_data_df = pd.read_csv("/Users/joeybest/Ai Core/EDA/exploratory-data-analysis---customer-loans-in-finance334/eda.csv", index_col="id", usecols=usecols)
        return loan_data_df

# %%
if __name__ == "__main__":

# reads credentials from yaml file
    credentials = credentials_reader()

    RDSDatabaseConnector(credentials)
    # next creates instance of the rds connector
    loan_data = RDSDatabaseConnector(credentials)
    loan_data.initiate_engine()
    loan_data.database_to_dataframe()
    
    extracted_data_frame = loan_data.database_to_dataframe()
    # print(extracted_data_frame)

    # was after extracted dataframe variable but it seems to work without it:
    # pd.DataFrame
    
    loan_data.save_to_csv("eda.csv")
    # saves CSV file

    table_of_loans = loan_data.load_localdata_to_dataframe()
    print(table_of_loans)
# %%
class DataFrameInfo:
    '''
    This class is used to dive into the a dataframe and begin to understand the information it contains.
    eda.csv being the dataframe that we are going to be loooking into.
    '''

    def __init__(self, df_info):
        '''
        This method is used to initialise this instance of the DataFrameInfo class.
        '''
        self.df_info = df_info

    def check_column_datatypes(self): 
        '''
        This method is used to check the column types of the df.
        '''
        return self.df_info.dtypes

    def extract_statistical_values(self, columns):
        '''
        This method is used to loop through a specified column to provide a print of the statistics that .describe() gives us.
        '''
        # Populate list of rows
        for col in columns:
            print(f'Statistics for Col: {col}')
            return self.df_info[col].describe()

    def count_unique_categories(self, columns):
        '''
        This method is used to count the number of distinct/unice values within a column.
        '''
        self.columns = self.df_info.select_dtypes(include=['category']).columns
        df_unique = self.df_info[columns[:]]
        return df_unique.nunique()

    def shape_of_dataframe(self):
        '''
        This method is used to provide an insight to the shape of the df.
        For example, how many rows and columns there are.
        '''
        print(f'Shape of DataFrame: [{self.df_info.shape[0]} rows x {self.df_info.shape[1]} columns]\n')

    def num_of_nulls(self):
        '''
        This method is used to count the number of Null or NaN values there are in the df.
        It the prints these into a pandas chart, with the headings: column, count and %null count.
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


# %%
if __name__ == "__main__":
    table_of_loans = pd.read_csv('eda.csv')
    df_df = DataFrameInfo(table_of_loans)

    # specific ['columns'] can be changed to any you make like
    print(df_df.check_column_datatypes())
    print(df_df.extract_statistical_values(['loan_amount']))
    print(f"The no. of Unique Items in {df_df.count_unique_categories(['grade'])}")
    print(df_df.shape_of_dataframe())
    print(df_df.num_of_nulls())

# %%
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

# %%
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

# %%

class DataFrameTransform:
    '''
    This class is used to apply transformations such as imputing or removing columns with missing data, to the dataframe.
    '''

    def __init__(self, df_transform):
        '''
        This method is used to initalise this instance of the DataFrameTransform class.
        '''
        self.df_transform = df_transform

    def num_of_nulls(self):
        '''
        This method is used to count the number of missing values in each of the columns in the data.
        It is also made into a percentage and then returned.
        '''
        cols = self.df_transform.columns
        # Define an empty list of null column rows
        null_coulmn_rows = []
        # Populate list of null column rows
        for col in cols:
            # if self.df[col].isnull().sum() > 0:
            null_coulmn_rows.append([col, self.df_transform[col].count(), 100*(self.df_transform[col].isnull().sum()/len(self.df_transform))])
        
        # Convert the list into dataframe rows
        data = pd.DataFrame(null_coulmn_rows)
        # Add columns headers
        data.columns = ['column', 'count', '% null count']  
        return data

    def drop_column(self, column):
        '''
        This method removes the listed columns from the dataframe.
        '''
        for col in column:
            self.df_transform.drop(col, axis=1, inplace=True)

    def drop_null_rows(self, column):
        '''
        This method is used to remove rows containing null or missing values.
        '''
        self.df_transform.dropna(subset=column, inplace=True)
    
    def impute_zeros(self, column):
        '''
        This method is used to impute values to a value of zero.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(0)

    def impute_median(self, column):
        '''
        This method is used to impute values to the median value of the column.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(self.df_transform[col].median())

    def impute_mean(self, column):
        '''
        This method is used to impute values to the mean value of the column.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(self.df_transform[col].mean())

    def log_transform(self, column):
        '''
        This method is used to transform the column values using a log transformation.
        This is in hopes to make the data less skewed.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].map(lambda i: np.log(i) if i > 0 else 0)
    
    def box_cox_transform(self, column):
        '''
        This method is used to transform the column values using a boxcox transformation.
        This is in hopes to make the data less skewed.
        '''
        for col in column:
            boxcox_column = self.df_transform[col] +0.01
            a, b = stats.boxcox(boxcox_column)
            self.df_transform[col] = a 

    def remove_outliers_iqr(self, data, threshold=1.5):
        '''
        This method is used to remove outliers of data.
        This uses the Inter Quartile Range data to remove the data past the lower and upper quatiles.
        '''
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    def remove_outliers_iqr_dataframe(self, column, threshold=1.5):
        '''
        This method is used to remove outliers of column data after a log or boxcox transformation.
        This uses the Inter Quartile Range data to remove the data past the lower and upper quatiles.
        '''
        filtered_dataframe = self.df_transform
        for col in column:
            filtered_dataframe[col] = self.remove_outliers_iqr(filtered_dataframe[col], threshold)
        return filtered_dataframe
    
    def save_transformed_data(self, filename='transformed_loan_data.csv'):
        '''
        This method is used to save the current dataframe as a new CSV file called 'transformed_loan_data.csv'
        '''
        self.df_transform.to_csv(filename, index=False)

    def save_untransformed_data(self, filename='untransformed_loan_data.csv'):
        '''
        This method is used to save the current dataframe as a new CSV file called 'transformed_loan_data.csv'
        '''
        self.df_transform.to_csv(filename, index=False)

class Plotter:

    def __init__(self, df_transform):
        '''
        This method is used to initalise this instance of the Plotter class.
        '''
        self.df_transform = df_transform

    def agostino_k2_test(self, col):
        '''
        This method is used to calculate the result of the agostino k2 test on a specific column.
        '''
        stat, p = normaltest(self.df_transform[col], nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def qq_plot(self, col):
        '''
        This method is used to plot a qqplot of a specific column.
        '''
        self.df_transform.sort_values(by=col, ascending=True)
        qq_plot = qqplot(self.df_transform[col], scale=1, line='q', fit=True)
        plt.show()

    def histogram(self, col):
        '''
        This method is used to plot a histogram of a specific column.
        '''
        self.df_transform[col].hist(bins=40)

    def density_plot(self, col):
        '''
        This method is used to plot a density_plot of a specific column.
        '''
        sns.histplot(data=self.df_transform[col], kde=True)
        sns.despine()

    def boxplot(self, col):
        '''
        This method is used to plot a boxplot of a specific column.
        '''
        fig = px.box(self.df_transform[col],width=600, height=500)
        fig.show()

    def scatter(self, col):
        '''
        This method is used to plot a scatter of a specific column.
        '''
        sns.scatterplot(self.df_transform[col])
        
    def show_missing_nulls(self):
        '''
        This method is used to plot the missing values of the dataframe.
        '''
        msno.matrix(self.df_transform)
        plt.show()

    def multi_plot(self, col):
        '''
        This method is used to calculate the mean, median, skewness and result of the agostino k2 test.
        It also plots a blended histogram and density plot as well as plotting a qqplot of a specific column.
        '''
        print(f'The median of {[col]} is {table_of_loans[col].median()}')
        print(f'The mean of {[col]} is {table_of_loans[col].mean()}')
        print(f"Skew of {[col]} column is {table_of_loans[col].skew()}")
        stat, p = normaltest(self.df_transform[col], nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        self.df_transform[col].hist(bins=40)
        sns.histplot(data=self.df_transform[col], kde=True)
        sns.despine()
        self.df_transform.sort_values(by=col, ascending=True)
        qq_plot = qqplot(self.df_transform[col], scale=1, line='q', fit=True)
        plt.show()

    def find_skew(self, col):
        '''
        This method is used to calculate if a columns skewness is above or below the threshold 0.85.
        '''
        if self.df_transform[col].skew() >= 0.86:
            print('Skewed!')
            print(f' {self.df_transform[col].skew()} is over 0.85')
        else:
            print(f' {self.df_transform[col].skew()} is under 0.85')

    def multi_hist_plot(self, num_cols):
        '''
        This method is used to plot a histogram of all specified columns.
        '''
        sns.set(font_scale=0.7)
        f = pd.melt(self.df_transform, value_vars=num_cols)
        g = sns.FacetGrid(f, col="variable", col_wrap=4,
                          sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        plt.show()

    def multi_qq_plot(self, cols):
        '''
        This method is used to plot a qqplot of all specified columns.
        '''
        remainder = 1 if len(cols) % 4 != 0 else 0
        rows = int(len(cols) / 4 + remainder)

        fig, axes = plt.subplots(
            ncols=4, nrows=rows, sharex=False, figsize=(12, 6))
        for col, ax in zip(cols, np.ravel(axes)):
            sm.qqplot(self.df_transform[col], line='s', ax=ax, fit=True)
            ax.set_title(f'{col} QQ Plot')
        plt.tight_layout()

    def show_outliers(self):
        '''
        This method is used to plot a boxplot of all specified columns in order to visualise what outliers are present.
        '''
        #select only the numeric columns in the DataFrame
        df = self.df_transform.select_dtypes(include=['float64'])
        plt.figure(figsize=(18,14))

        for i in list(enumerate(df.columns)):
            fig_cols = 4
            fig_rows = int(len(df.columns)/fig_cols) + 1
            plt.subplot(fig_rows, fig_cols, i[0]+1)
            sns.boxplot(data=df[i[1]]) 

        # Show the plot
        plt.tight_layout()
        plt.show()

    def show_outliers_after_removal(self, dataframe, columns):
        '''
        This method is used to plot a boxplot of all specified columns after outliers have been removed. 
        This is in order to visualise if the previous method worked.
        '''
        plt.figure(figsize=(18, 14))

        for i, col in enumerate(columns):
            fig_cols = 4
            fig_rows = len(columns) // fig_cols + 1
            plt.subplot(fig_rows, fig_cols, i + 1)
            sns.boxplot(data=dataframe[col])

        plt.tight_layout()
        plt.show()


# %%
if __name__ ==  "__main__":
    # load the data in
    table_of_loans = pd.read_csv('full_loan_data.csv.csv')
    df_cols = DataFrameTransform(table_of_loans)
    plot = Plotter(table_of_loans)

    plot.show_missing_nulls
    # A number of missing values across the df visualised

    print(df_cols.num_of_nulls())
    # COLUMN                        COUNT    % NULL COUNT
    # funded_amount                 51224      5.544799
    # term                          49459      8.799395
    # int_rate                      49062      9.531449
    # employment_length             52113      3.905515
    # mths_since_last_delinq        23229      57.166565
    # mths_since_last_record        6181       88.602460
    # last_payment_date             54158      0.134609
    # next_payment_date             21623      60.127971
    # last_credit_pull_date         54224      0.012908
    # collections_12_mths_ex_med    54180      0.094042
    # mths_since_last_major_derog   7499       86.172116

    df_cols.impute_zeros(['employment_length'])
    df_cols.impute_median(['int_rate'])
    df_cols.drop_null_rows(['last_payment_date', 'last_credit_pull_date'])
    df_cols.drop_column(['mths_since_last_delinq', 'next_payment_date', 'mths_since_last_record', 'mths_since_last_major_derog'])
    # Drops/imputes the Null values or rows

    print(df_cols.num_of_nulls())
    #re-visualise data to see no more Null values
    
    df_cols.save_untransformed_data('untransformed_loan_data.csv')
    # to save a version of untransformed data for use in Milestone 4 later on

    # Skewness of the data needs to be visualised:
    numerical_cols = ['loan_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'dti', 'annual_inc', 'total_payment', 'total_accounts', 'open_accounts', 'last_payment_amount']
    plot.multi_hist_plot(numerical_cols)
    plot.multi_qq_plot(numerical_cols)
    # which will print the density plot for all columns in the numerical_cols variable
    # and then will print the qq plot for all columns in the numerical_cols variable 

    # or use # following code for the:
    # mean, median, agostino_k2_test, qqplot, histogram and density plot together of a single column
    # plot.multi_plot('instalment')
    # plot.multi_plot('open_accounts')
    # plot.multi_plot('total_rec_prncp')
    # plot.multi_plot('total_payment')
    # plot.multi_plot('total_rec_int')
    # plot.multi_plot('out_prncp')
    # plot.multi_plot('last_payment_amount')
    # plot.multi_plot('inq_last_6mths')
    # plot.multi_plot('annual_inc')
    # plot.multi_plot('delinq_2yrs')
    # plot.multi_plot('total_rec_late_fee')
    # plot.multi_plot('recoveries')
    # plot.multi_plot('collection_recovery_fee') 
    # add/change columns as you see fit 

    # perform the transformations on skewed columns:
    boxcox_cols = ['loan_amount', 'instalment', 'int_rate', 'dti', 'funded_amount_inv', 'total_payment']
    df_cols.boxcox_transform(boxcox_cols)

    logt_cols = ['annual_inc', 'total_accounts', 'open_accounts', 'last_payment_amount']
    df_cols.log_transform(logt_cols)

    # May not need to transfor below columns, but is here just in case:
    #    df_cols.log_transform(['total_rec_prncp'])
    #    df_cols.log_transform(['total_rec_int'])
    #    df_cols.log_transform(['out_prncp'])
    #    df_cols.log_transform(['inq_last_6mths'])
    #    df_cols.log_transform(['delinq_2yrs'])
    #    df_cols.log_transform(['total_rec_late_fee'])
    #    df_cols.log_transform(['recoveries'])
    #    df_cols.log_transform(['collection_recovery_fee'])
    # outliers have been used through performing these transformations

    # code saves file to working directory
    df_cols.save_transformed_data('transformed_loan_data.csv')

    transformed_loans = pd.read_csv('transformed_loan_data.csv')
    df_cols = DataFrameTransform(transformed_loans)
    plot = Plotter(transformed_loans)

    plot.multi_qq_plot(numerical_cols)
    # prints qqplot of all columns in 'numerical_cols', some outliers can be seen, lets look closer
    plot.show_outliers()
    # boxplot of columns showing outliers present past the IQR
    plot.boxplot(numerical_cols)
    #visualised for the numerical_cols

    # Removing outliers and re-visualising boxplots
    filtered_df = df_cols.remove_outliers_iqr_dataframe(column= numerical_cols, threshold=1.5)
    plot.show_outliers_after_removal(dataframe=filtered_df, columns=numerical_cols)

    # Making the filtered data a variable to use the classes
    df_without_outliers = DataFrameTransform(filtered_df)
    plot_without_outliers = Plotter(filtered_df)


    # Checking data again
    plot_without_outliers.show_missing_nulls()
    df_without_outliers.num_of_nulls()
    # removing the outliers has left Null values, so we will either transform or remove the rows

    # COLUMN                COUNT   % NULL COUNT
    # loan_amount           54145   0.011080
    # funded_amount_inv     53992   0.293623
    # int_rate              54107   0.081254
    # instalment            54111   0.073868
    # annual_inc            53000   2.125538
    # open_accounts         53631   0.960278
    # total_accounts        53282   1.604772
    # total_payment         53971   0.332404
    # last_payment_amount   53950   0.371184

    # values are all very low
    # when imputing outliers, the median is more robust and less influenced by extreme values.
    to_be_median_imputed = ['loan_amount', 'funded_amount_inv',  'int_rate', 'instalment', 'annual_inc', 'open_accounts', 'total_accounts', 'total_payment', 'last_payment_amount']
    df_without_outliers.impute_median(to_be_median_imputed)

    # Checking data again
    plot_without_outliers.show_missing_nulls()
    df_without_outliers.num_of_nulls()
    # No Null values

    # checking skewness and outliers
    plot_without_outliers.multi_hist_plot(numerical_cols)
    plot_without_outliers.multi_qq_plot(numerical_cols)
    # data improved!


    # corrolation matrix: checking for multicollinearity issues, will drop overly corrolated columns
    plot_without_outliers.heatmap(numerical_cols)

    ## Threshold for highly correlated columns is 0.85
    # Multi-linearity between 'loan_amount', 'instalment' & 'funded_amount_inv'

    # funded_amount_inv and loan_amount corrolation = 0.96
    # instalmannt and loan_amount corrolation = 0.96
    # instalmannt and funded_amount_inv corrolation = 0.93

    # total_payment is highly corrolated with loan_amount, funded_amount_inv and instalment too, but only at 0.81, 0.78 and 0.81 respectively
    # therefore not passed the 0.85 threshold

    # desptie 'loan_amount', 'instalment' & 'funded_amount_inv' being past the threshold, they're all important for the analysis stage, so we wont be dropping any of them!



#%%
# Beginning of Milestone 4 Tasks
filtered_df = pd.read_csv('untransformed_loan_data.csv')

# loan recovered (total_payment) againts invstor funding (funded_amount_inv) & total amount funded (loan_amount)
total_amount_funded = filtered_df['loan_amount'].sum()
invstor_funding = filtered_df['funded_amount_inv'].sum()
total_payment_sum = filtered_df['total_payment'].sum()

# same as above but rounded and prints together
totals = round(filtered_df[['loan_amount', 'funded_amount_inv', 'total_payment']].sum(), 0)

# % of loans recovered
per_of__inv_loan_recovered = round((total_payment_sum/invstor_funding)*100, 2)
per_of__total_loan_recovered = round((total_payment_sum/total_amount_funded)*100, 2)

# Calculating percentage of investor funding recovered
pct_invetor_rec = round(100 * totals.total_payment/totals.funded_amount_inv, 2)
# Calculating percentage of funded amount recovered
pct_total_rec = round(100 * totals.total_payment/totals.loan_amount, 2)

# total left to be paid
remaining_amount = round(total_amount_funded - total_payment_sum, 2)

# 6 month projection:
# sum of the total_payment and the instalment and multiply this by 6. 
# Then divide this by the sum of the funded amount inv. 
# times By 100 to get this as a percentage
six_month_projection = round((filtered_df["total_payment"].sum()+(filtered_df["instalment"].sum()*6))/(filtered_df["funded_amount_inv"].sum())*100, 2)


# Filter the DataFrame to include only charged off loans
charged_off_loans = filtered_df[filtered_df['loan_status'] == 'Charged Off']
# Calculate the percentage of charged off loans
charged_off_percentage = (charged_off_loans.shape[0] / filtered_df.shape[0]) * 100
# Calculate the total amount paid towards charged off loans
total_payment_charged_off = charged_off_loans['total_payment'].sum()

# Charged loan calculations:
charged_off_loans_loan_amount_sum = round(charged_off_loans['loan_amount'].sum(),2)
charged_off_loans_total_payment_sum = round(charged_off_loans['total_payment'].sum(),2)
charged_off_loans_loss = round(charged_off_loans_loan_amount_sum - charged_off_loans_total_payment_sum, 2)

# creating new columns in the dataframe:
filtered_df['num_of_payments_made'] = filtered_df['total_payment'] / filtered_df['instalment']
filtered_df['months_left_to_pay'] = filtered_df['term'] - filtered_df['num_of_payments_made']


# Filtering the DataFrame to include only late loans
risk_loans = filtered_df[(filtered_df['loan_status'] == 'Late (31-120 days)') | (filtered_df['loan_status'] == 'Late (16-30 days)')]
# Calculating the percentage of risk loans
risk_percentage = round((risk_loans.shape[0] / filtered_df.shape[0]) * 100, 2)
# Calculating the total amount paid towards charged off loans
total_customers_in_risk_bracket = round(risk_loans.shape[0],0)

# Risk loan calculations:
risk_loans_loan_amount_sum = round(risk_loans['loan_amount'].sum(),2)
risk_loans_total_payment_sum = round(risk_loans['total_payment'].sum(),2)
risk_loans_loss = round(risk_loans_loan_amount_sum - risk_loans_total_payment_sum, 2)

risk_amount_left_to_pay_ = risk_loans['months_left_to_pay'] * risk_loans['instalment']
projected_loss = round(risk_amount_left_to_pay_.sum(), 2)

risk_charged_off_rev_pct = risk_percentage + charged_off_percentage 

# creating subsets of the df for specific loan status'
paid_loans = filtered_df[(filtered_df['loan_status'] == 'Fully Paid')]
on_time_loans = filtered_df[(filtered_df['loan_status'] == 'Current')]
grace_period_loans = filtered_df[(filtered_df['loan_status'] == 'In Grace Period')]


# %%
if __name__ =="__main__":
    filtered_df = pd.read_csv('untransformed_loan_data.csv')
    # untransformed data, specifically for milestone 4
    df_without_outliers = DataFrameTransform(filtered_df)
    plot_without_outliers = Plotter(filtered_df)
    Transform = DataTransform(filtered_df)

    filtered_df['term'] = filtered_df['term'].replace(0, 36)
    # replaces previously imputed 0's in ['term'] with median value(36)

    print(f' Total loan amount funded is {total_amount_funded}')
    print(f' Total amount invested is {invstor_funding}')
    print(f' Total payment recovered is {total_payment_sum}')

    print(totals)

    # % of loans recovered
    print(f'Percentage of the loans recovered against the investor_funding is {(total_payment_sum/invstor_funding)*100}')
    print(f'Percentage of the loans recovered against the total_amount_funded is {(total_payment_sum/total_amount_funded)*100}')
    
    print(f' {per_of__inv_loan_recovered}%')
    print(f' {per_of__total_loan_recovered}%')

    data = {
    'funded_amount_inv': (filtered_df['total_payment'].sum()/invstor_funding)*100,
    'loan_amount': (filtered_df['total_payment'].sum()/total_amount_funded)*100
    }
    plot_without_outliers.bar(data)

    data = {
    'Funding': ['Investor', 'Total'],
    'Percent': [pct_invetor_rec, pct_total_rec]
    }
    # Creating a Pandas DataFrame
    df = pd.DataFrame(data)
    display(df)

    df.plot(x="Funding", y="Percent", kind="bar", ylabel="% Recovered") 

    df.plot(y="Percent", kind="pie", ylabel="% Recovered", labels=df['Funding'], startangle=90, xlabel='Funding')
    # Adding a title and xlabel
    plt.title("Recovery Percentage by Funding Source")
    plt.legend(title="Funding Source")
    # Displaying the chart
    plt.show()

    # whats left to be paid overall
    print(remaining_amount)

    # Creating a DataFrame with the recovered and remaining amounts
    data = {'Amount': [total_payment_sum, remaining_amount]}
    df = pd.DataFrame(data, index=['Recovered', 'Remaining'])

    # Plotting the pie chart
    df.plot(y='Amount', kind='pie', autopct='%1.1f%%', startangle=90)
    # Adding a title and legend title
    plt.title("Percentage of Amount Recovered")
    plt.legend(title="Key")
    # Displaying the chart
    plt.show()


    data = {'Funding': ['Total'],
            'Percent': [six_month_projection]}
    # Create a Pandas DataFrame
    df = pd.DataFrame(data)
    display(df)

    df.plot(x="Funding", y="Percent", kind="bar", ylabel="% Recovered")  

    # Charged loan calculations:
    print(f"Percentage of charged off loans: {charged_off_percentage:.2f}%")
    print(f"Total amount paid towards charged off loans: {total_payment_charged_off:.2f}")
    print(' ')
    
    print(f' Total amount to be paid: {charged_off_loans_loan_amount_sum}')
    print(f' Total that has been paid: {charged_off_loans_total_payment_sum}')
    print(f' Total that has been lost: {charged_off_loans_loss}')
    print(' ')

    data = {'Paid': [charged_off_loans_total_payment_sum],
        'Loss': [charged_off_loans_loss]}
    df = pd.DataFrame(data)
    display(df)

    # Creating a pie chart
    labels = ['Paid', 'Loss']
    sizes = [charged_off_loans_total_payment_sum, charged_off_loans_loss]
    colors = ['#1f77b4', '#ff7f0e']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    # Setting the title of the pie chart
    plt.title('Paid vs. Loss')
    # Displaying the pie chart
    plt.show()

    # Creating a pie chart
    labels = ['Paid', 'Loss']
    sizes = [charged_off_loans_total_payment_sum, charged_off_loans_loss]
    colors = ['#1f77b4', '#ff7f0e']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    # Setting the title of the pie chart
    plt.title('Paid vs. Loss')
    # Adding a legend
    plt.legend()
    # Displaying the pie chart
    plt.show()

    print(' ')
    print('Risk loan calculations:')
    print(f"Percentage of risk loans: {risk_percentage}%")
    print(f"Total number of customers in risk bracket: {total_customers_in_risk_bracket}")
    print(' ')

    
    # print(f' Total amount to be paid: {risk_loans_loan_amount_sum}')
    # print(f' Total that has been paid: {risk_loans_total_payment_sum}')
    # print(f'Percentage: {round((risk_loans_loss/risk_loans_loan_amount_sum)*100,2)}%')
    # print(f'Total still to be paid (exclusive of int_rate): {risk_loans_loss}')

    print(f"Projected Loss if Switched to Charged Off: ${projected_loss}")
    print(f'Percentage of late and charged off revenue: {risk_charged_off_rev_pct:.2f}%')
    print(' ')
    # compare 'grade', 'purpose', 'home_ownership', 'employment_length', 'sub_grade' and 'annual_inc' with:
    # customers who have already stopped paying     &    customers who are currently behind on payments.

    # Does the grade of the loan have effect on customers not paying?
    # Is the purpose for the loan likely to have an effect?
    # Does the home_ownership value contribute to the likelihood a customer won't pay?


    data = {'Paid Grades': paid_loans['grade'].value_counts(),
        # 'On Time Grades': on_time_loans['grade'].value_counts(),
        # 'Grace Period Grades': grace_period_loans['grade'].value_counts(),
        'Risk Loan Grades': risk_loans['grade'].value_counts(),
        'Charged Off Grades': charged_off_loans['grade'].value_counts()
        }

    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        print(f"Paid Loans {index} Grade: {row['Paid Grades']}")
        # print(f"On time {index} Grades: {row['On Time Grades']}")
        # print(f"Grace Period {index} Grades: {row['Grace Period Grades']}")
        print(f"Risk Loan {index} Grades: {row['Risk Loan Grades']}")
        print(f"Charged Off Loan {index} Grades: {row['Charged Off Grades']}") 

    df.plot(kind='bar', stacked=True)

    plt.title('Counts of Grades')
    plt.xlabel('Grade')
    plt.ylabel('Count')
    plt.legend()
    # Displaying the plot
    plt.show()
    print(' ')


    data = {'Paid sub grade': paid_loans['sub_grade'].value_counts(),
        # 'On Time sub grade': on_time_loans['sub_grade'].value_counts(),
        # 'Grace Period sub grade': grace_period_loans['sub_grade'].value_counts(),
        'Risk Loan sub grade': risk_loans['sub_grade'].value_counts(),
        'Charged Off sub grade': charged_off_loans['sub_grade'].value_counts()
        }

    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        print(f"Paid Loans sub grade {index}: {row['Paid sub grade']}")
        # print(f"On time sub grade {index}: {row['On Time sub grade']}")
        # print(f"Grace Period sub grade {index}: {row['Grace Period sub grade']}")
        print(f"Risk Loan sub grade {index}: {row['Risk Loan sub grade']}")
        print(f"Charged Off Loan sub grade {index}: {row['Charged Off sub grade']}")

    df = pd.DataFrame(data)

    df.plot(kind='bar', stacked=True)

    plt.title('Count of Sub Grade')
    plt.xlabel('Sub Grades')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    print(' ')

    # next plot
    data = {'Paid purpose': paid_loans['purpose'].value_counts(),
        # 'On Time purpose': on_time_loans['purpose'].value_counts(),
        # 'Grace Period purpose': grace_period_loans['purpose'].value_counts(),
        'Risk Loan purpose': risk_loans['purpose'].value_counts(),
        'Charged Off purpose': charged_off_loans['purpose'].value_counts()
        }

    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        print(f"Paid Loans purpose: {index} {row['Paid purpose']}")
        # print(f"On time purpose: {index} {row['On Time purpose']}")
        # print(f"Grace Period purpose: {index} {row['Grace Period purpose']}")
        print(f"Risk Loan purpose: {index} {row['Risk Loan purpose']}")
        print(f"Charged Off Loan purpose: {index} {row['Charged Off purpose']}")

    df.plot(kind='bar', stacked=True)

    plt.title('Counts of Purposes')
    plt.xlabel('Loan Purpose')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    print(' ')

    # next plot
    data = {'Paid home ownership': paid_loans['home_ownership'].value_counts(),
        # 'On Time home ownership': on_time_loans['home_ownership'].value_counts(),
        # 'Grace Period home ownership': grace_period_loans['home_ownership'].value_counts(),
        'Risk Loan home ownership': risk_loans['home_ownership'].value_counts(),
        'Charged Off home ownership': charged_off_loans['home_ownership'].value_counts()
        }
    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        print(f"Paid Loans home ownership: {index} {row['Paid home ownership']}")
        # print(f"On time home ownership: {index} {row['On Time home ownership']}")
        # print(f"Grace Period home ownership: {index} {row['Grace Period home ownership']}")
        print(f"Risk Loan home ownership: {index} {row['Risk Loan home ownership']}")
        print(f"Charged Off Loan home ownership: {index} {row['Charged Off home ownership']}")

    df.plot(kind='bar', stacked=True)

    plt.title('Counts of Home Ownership Status')
    plt.xlabel('Home Ownership')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    print(' ')


    # next plot 
    data = {'Paid employment length': paid_loans['employment_length'].value_counts(),
        # 'On Time employment length': on_time_loans['employment_length'].value_counts(),
        # 'Grace Period employment length': grace_period_loans['employment_length'].value_counts(),
        'Risk Loan employment length': risk_loans['employment_length'].value_counts(),
        'Charged Off employment length': charged_off_loans['employment_length'].value_counts()
        }

    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        print(f"Paid Loans employment length {index}: {row['Paid employment length']}")
        # print(f"On time employment length {index}: {row['On Time employment length']}")
        # print(f"Grace Period employment length {index}: {row['Grace Period employment length']}")
        print(f"Risk Loan employment length {index}: {row['Risk Loan employment length']}")
        print(f"Charged Off Loan employment length {index}: {row['Charged Off employment length']}")

    df.plot(kind='bar', stacked=True)

    plt.title('Length of Employment')
    plt.xlabel('Employment Length')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    print(' ')

    # next plot
    data = {
    'Paid annual income': paid_loans['annual_inc'],
    'Risk Loan annual income': risk_loans['annual_inc'],
    'Charged Off annual income': charged_off_loans['annual_inc']
    }

    df = pd.DataFrame(data)

    # Print the counts for each loan category
    for column in df.columns:
        counts = df[column].value_counts()
        print(f"{column} counts:")
        print(counts)
        print()

    # Plotting histogram-density plots using seaborn
    for column in df.columns:
        plt.figure(figsize=(10, 6))  # Set the figure size as needed
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f"Histogram-Density Plot - {column}")
        plt.xlabel("Annual Income")
        plt.ylabel("Density")
        plt.xlim(0, 700000)  # Adjust the x-axis limits as needed

        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[column])
        plt.title(f"Box Plot - {column}")
        plt.ylabel("Annual Income")

        # Customising y-axis tick labels
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
        plt.tight_layout()  # Adjust the spacing between subplots
        plt.show()
        print(' ')

    # different view of box plots for annual_inc
    data = {'Paid annual income': paid_loans['annual_inc'].value_counts(),
            # 'On Time annual income': on_time_loans['annual_inc'].value_counts(),
            # 'Grace Period annual income': grace_period_loans['annual_inc'].value_counts(),
            'Risk Loan annual income': risk_loans['annual_inc'].value_counts(),
            'Charged Off annual income': charged_off_loans['annual_inc'].value_counts()
            }

    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        print(f"Paid Loans annual income {index}: {row['Paid annual income']}")
        # print(f"On time annual income {index}: {row['On Time annual income']}")
        # print(f"Grace Period annual income {index}: {row['Grace Period annual income']}")
        print(f"Risk Loan annual income {index}: {row['Risk Loan annual income']}")
        print(f"Charged Off Loan annual income {index}: {row['Charged Off annual income']}")

    df = pd.DataFrame(data)

    # Plotting the box plot
    df.boxplot()
    plt.title('Distribution of Annual Income')
    plt.ylabel('Annual Income')
    plt.xticks(rotation=45)
    print(' ')


    # another view
    data = {
        'Paid annual income': paid_loans['annual_inc'],
        'Risk Loan annual income': risk_loans['annual_inc'],
        'Charged Off annual income': charged_off_loans['annual_inc']
    }

    df = pd.DataFrame(data)

    # Print the counts for each loan category
    for column in df.columns:
        counts = df[column].value_counts()
        print(f"{column} counts:")
        print(counts)
        print()

    df.boxplot()
    plt.title('Distribution of Annual Income')
    plt.ylabel('Annual Income')
    plt.xticks(rotation=45)
    plt.show()
    print(' ')


# %%
# Summary:

# Grade / Sub Grade:
# Those with a higher Grade or Sub grade were more likely to pay back the full loan, than those with lower grades
# As the grade decends from 'A', there is a decrease in fully paid loans and an increase in risk and charged off loans between 'B1-E2'.
# Thus, the lower the grade the more risky the loan becomes.

# Annual Income:
# in comparison to the paid loans, charged off and risk loans are drastically smaller
# Thus indicating that a higher income is suggestive of being able to pay the full loan back
# And smaller annual incomes are suggestive of needing more time/assistance or likely to be charged off 

# Purpose:
# debt_consolidation Has a significant increase within the risk and charged off loans as a the primary/majority purpose for their loans
# The majority of paid loans was for this purpose too
# Meaning purpose alone may not directly be linked to likelyhood to be able to pay back the loans, an analysis of all indicators would need to be observed
# Other than that small businesses and credit_card purposes also poses a strong likelyhood of being charged off

# Home Ownership:
# Morgage owners are very likly to be able to pay the full loan
# Renters are slightly more likely to be charged off, but are still hold a good place for repaying their loans in full
# however, people who own their property seem to be not as likely as the previous two to pay the loan in full
# but there is less data on that point
# so in summary people who rent their property should be accepted with some caution 

# Employment Length:
# The largest amount of paid, charged off and risk loans all come frome people who have been in employment for 10 years+
# But the paid loans are significantly higher, suggesting that people in long term employment are more likely to pay their loans in full.
# However, ignoring the 10years+ most of the columns seem largely similar
# This could suggest that employment length doesnt have a significant impact on ability to repay loans
# but it should still be noted that people who have worked less than a year have the 2nd largest amount of charged off loans after the 10years+

