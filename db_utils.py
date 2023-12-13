import pandas as pd
import yaml
from sqlalchemy import create_engine

class RDSDatabaseConnector:
    def __init__(self, credential_input: credential_output):
        self.credential_input = credential_input

    def yaml():
        with open('credentials.yaml', 'r') as file:
            credentials = yaml.safe_load(file)
            print(credentials)
            return
        
    def data_extraction(engine):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = 'eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com'
        USER = 'loansanalyst'
        PASSWORD = 'EDAloananalyst'
        DATABASE = 'payments'
        PORT = 5432
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        return engine

    def database_to_dataframe(data_extraction, engine):
        loans = pd.read_sql_table('loan_payments', data_extraction(engine))
        loans.head(10)
        loans.tail(10)
        return

    def saves_data_locally(database_to_dataframe, loans):
        database_to_dataframe(loans).to_csv('eda.csv', sep=',', index=False, encoding='utf-8')

    def load_localdata_to_dataframe():
        # Defining the columns to read
        usecols = ["id", "member_id","loan_amount", "funded_amount", "funded_amount_inv", "term", "int_rate", "instalment", "grade", "sub_grade", "employment_length", "home_ownership", "annual_inc", "verification_status", "issue_date", "loan_status", "payment_plan", "purpose", "dti", "delinq_2yr", "earliest_credit_line", "inq_last_6mths", "mths_since_last_record", "open_accounts", "total_accounts", "out_prncp", "out_prncp_inv", "total_payment", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_payment_date", "last_payment_amount", "next_payment_date", "last_credit_pull_date", "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code", "application_type"]
# Read data with subset of columns
        loan_data = pd.read_csv("/Users/joeybest/Ai Core/EDA/exploratory-data-analysis---customer-loans-in-finance334/eda.csv",\
        index_col="id", usecols=usecols)




