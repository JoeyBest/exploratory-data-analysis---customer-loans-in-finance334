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

### What is the EDA project?
#### This project is based on a scenario of a large financial institution, where managing loans is a critical component of business operations.
#### Understanding the loan portfolio data is essential in order to help the business make informed decisions are made about loan approvals, pricing and risk is efficiently managed.
#### To do so I will perform exploratory data analysis (EDA) on the loan portfolio, using various statistical and data visualisation techniques, 
#### which will allow me to report patterns, relationships, and anomalies in the loan data.

### Aim of the project:
- Help make informed decisions within the business
- Gain a deeper understanding of the risk and return associated with the business' loans
- Improve the performance and profitability of the loan portfolio
- Be able to read and understand results from statistical analysis
- Report any anomalies in the data

### What I learned:
- How to use classes effectively
- How to transform data and how to impute/drop values
- How to create plots for data visualisation
- How to analyse a columns importance
- How to remove outliers that are affecting data skewness
- How to interpret data from a financial portfolio
- How to to identify indicators that can show risky loans
- How to calculate future repayment predictions
- How to create new columns for data I will use again

## Installation instructions:

1. You will need to download and clone the git repository (Repo):
  #### By clicking *'< > code'* you can copy the github https code
  #### Paste is into you CLI after the following: *git clone*
  
2. You may want to create a new conda environment for all the libraries that you will need to use for this project.
  - conda create --name *'insert conda env name'*

## Usage instructions:

1. Firstly activate the conda environment created to ensure the appropriate libraries are avalible and ready to use.
  
2. User can choose to use the code from main.ipynb or follow through the .py files (the .ipynb file may be better for those wanting a good visual representation of all the data).
   - The order of .py files for user ease: db_utils.py, dataframe_info.py, DataTransformation.py, plot.py
  
3. The difference between the two is readability and speed.
   The .py files contain all the docstrings for users who want to use the same code and understand it better. 
   The .py files uses 
   ```
   if __name__ == "__main__":
   ``` 
   to encapsulate important parts of code, but it doesnt contain everything needed to analyse, clean and visualise all partsof the data. 
   Whereas the main.ipynb goes through the code in sections, so each code needs running individually. But there is still explainations in the notes and markdown boxes along the way.

4. Running the chosen 'main.ipynb' file will extract the data from an AWS Relational Database (RDS) and write it into the appropriate csv file.
   - This step uses the confidental file credential.yaml to access the csv file.
   - The CSV file will now be saved in your working directory as 'eda.csv'.
     
5. If code is going to be run from the .py files, then running each one once, will perform all actions needed on the data and save more up to date versions of the data.
   - including: 'transformed_loan_data.csv', 'filtered_loan_data.csv', 'full_loan_data.csv' and 'untransformed_loan_data.csv'

6. Running the code from 'main.ipynb' will take you chronolgically through the sections:
   - Accessing the df
   - Exploring the data
   - Transforming pt 1
   - Transforming pt 2
   - Visualising the data
   - Outlier Stage
   - Analysis and Visualisation
   - Summary

## File structure of the project:
```
.
└── /Users/
    └── joeybest/
        └── Ai Core/
            └── EDA/
                └── exploratory-data-analysis---customer-loans-in-finance334/
                    ├── credentials.yaml
                    ├── dataframe_info.ipynb
                    ├── dataframe_info.py
                    ├── DataTransformation.ipynb
                    ├── DataTransformation.py
                    ├── db_utils.py
                    ├── eda.csv
                    ├── filtered_loan_data.csv
                    ├── full_loan_data.csv
                    ├── main.ipynb
                    ├── main.py
                    ├── plot.ipynb
                    ├── plot.py
                    ├── README.md
                    ├── transformed_loan_data.csv
                    ├── untransformed_loan_data.csv
                    └── __pycache__/
                        ├── DataTransformation.cpython-311.pyc
                        └── db_utils.cpython-311.pyc
```

## License information:
