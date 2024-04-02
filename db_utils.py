import pandas as pd
import yaml
from sqlalchemy import create_engine
from sqlalchemy import inspect


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

        Parameters:
        ----------
        credentials: dict
        Dictionary containing 'Host', 'Password', 'User', 'Database' and 'Port' required for the sqlalchemy to establish a connection with the RDS.
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

        Returns:
        ----------
        list
            returns the top 10 and bottom 10 rows and columns in the loan payments.
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
        
        Returns:
        ----------
        list
            returns a shortened version of the loan payments file information.
        '''
        loan_data_df = pd.read_csv("/Users/joeybest/Ai Core/EDA/exploratory-data-analysis---customer-loans-in-finance334/eda.csv", index_col="id", usecols=usecols)
        return loan_data_df


if __name__ == "__main__":

# reads credentials from yaml file
    credentials = credentials_reader()

    RDSDatabaseConnector(credentials)
    # next creates instance of the rds connector
    loan_data = RDSDatabaseConnector(credentials)
    loan_data.initiate_engine()
    loan_data.database_to_dataframe()
    
    extracted_data_frame = loan_data.database_to_dataframe()
    
    loan_data.saves_data_locally()
    # saves CSV file

    table_of_loans = loan_data.load_localdata_to_dataframe()
    print(table_of_loans)
