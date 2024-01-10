# Exploratory Data Analysis -
## Customer Loans in Finance Project

## Table of Contents:
| Syntax | Description |
| ----------- | ----------- |
| 1 | Table of Contents |
| 2 | Project description |
| 5 | What I learned |
| 6 | Installation instructions |
| 7 | Usage instructions |
| 8 | File structure of the project |
| 9 | License information |

## Project Description

### <What is the EDA project?>
#### This project is based on a scenario of a large financial institution, where managing loans is a critical component of business operations.
#### Understanding the loan portfolio data is essential in order to help the business make informed decisions are made about loan approvals, pricing and risk is efficiently managed.
#### To do so I will perform exploratory data analysis (EDA) on the loan portfolio, using various statistical and data visualisation techniques, 
#### which will allow me to report patterns, relationships, and anomalies in the loan data.

### <Aim of the project:>
- Help make informed decisions within the business
- Gain a deeper understanding of the risk and return associated with the business' loans
- Improve the performance and profitability of the loan portfolio
- Be able to read and understand results from statistical analysis
- Report any anomalies in the data

### <What I learned:>
- I have leant how to use classes effectively
- I have leant how to transform data and how to impute/drop values
- I have learnt how to create plots for data visualisation
- I have leant how to analyse a columns importance
- I ahve learnt how to remove outliers that are affecting data skewness
- I learnt how to interpret data from a financial portfolio


## Installation instructions:

1. You will need to download and clone the git repository (Repo):
  #### By clicking *'< > code'* you can copy the github https code
  #### Paste is into you CLI after the following: *git clone*
  
2. You may want to create a new conda environment for all the libraries that you will need to use for this project.
  - conda create --name *'insert conda env name'*

#### Below can be seen a list of libraries needed to complete the project:
- import missingno as msno
- import numpy as np
- import pandas as pd
- import plotly.express as px
- import seaborn as sns
- import statsmodels.api as sm
- import sqlalchemy
- import yaml
- from matplotlib import pyplot
- from scipy import stats
- from scipy.stats import normaltest
- from statsmodels.graphics.gofplots import qqplot
- from sqlalchemy import create_engine
- from sqlalchemy import inspect

## Usage instructions:

1. Firstly activate the conda environment created to ensure the appropriate libraries are avalible and ready to use.
2. Run the 'db_utils.py' file to extract the data from an AWS Relational Database (RDS) and write it into the appropriate csv file.
   - This step uses the confidental file credential.yaml to access the csv file.
   - The CSV file will now be saved in your working directory as 'eda.csv'.
     
3. If code is going to be run from 'main.py' or 'main.ipynb', then running it once perform all actions needed on the data and save more up to dat versions of it.
   - including: 'transformed_loan_data.csv' and 'filtered_loan_data.csv'

#### COME BACK TO THE ABOVE AND COMPLETE!!!!!

4. Running 'dataframe_info.py'  will provide you with a print of the column datatypes, a statistical description of the column 'loan_amount' (*feel free to change the columns*),
   the no. of Unique Items in in the column 'grade', the shape_of_dataframe and the number of null values in the data.
   This is just to get an understanding of some of the data

5. 
   #### ##################
5. This contains the exploratory data analysis where the data is transformed to remove and impute nulls, visualise skewness, remove outliers and identify correlation.
   This provides insights, conclusions and visualisations from the transformed data. Analysis on the current state of loans, current and potential losses as well as identifying risk      indicating variables are provided in this notebook.
#### ################## come back to this!!!!!

## File structure of the project:
.
└── /Users/
    └── joeybest/
        └── Ai Core/
            └── EDA/
                └── exploratory-data-analysis---customer-loans-in-finance334/
                    ├── credentials.yaml
                    ├── eda.csv
                    ├── db_utils.py
                    ├── dataframe_info.py
                    ├── dataframe_info.ipynb
                    ├── DataTransformation.py
                    ├── DataTransformation.ipynb
                    ├── plot.py
                    ├── plot.ipynb
                    ├── transformed_loan_data.csv
                    ├── README.md
                    └── __pycache__/
                        ├── DataTransformation.cpython-311.pyc
                        └── db_utils.cpython-311.pyc
## License information:
