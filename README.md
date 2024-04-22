# Exploratory Data Analysis -
## Customer Loans in Finance Project

## Table of Contents:
| Syntax | Description |
| ----------- | ----------- |
| 1 | Table of Contents |
| 2 | Project description |
| 3 | Installation instructions |
| 4 | Usage instructions |
| 5 | File structure of the project |
| 6 | License information |

## Project Description

### What is the EDA project?
#### This project is based on a scenario of a large financial institution, where managing loans is a critical component of business operations.
#### Understanding the loan portfolio data is essential in order to help the business make sure informed decisions are made about loan approvals, pricing and risk is efficiently managed.
#### To do so I will perform exploratory data analysis (EDA) on the loan portfolio, using various statistical and data visualisation techniques. Which will allow me to report patterns, relationships, and anomalies in the loan data.

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
  - All libraries used can be found in 'requirements.txt'


## Usage instructions:

1. Firstly activate the conda environment created to ensure the appropriate libraries are avalible and ready to use.

1. The user will need to run the db_utils.py file first as this will extract the data from an AWS Relational Database (RDS) and write it into the appropriate csv file.
   - This step uses the confidental file credential.yaml to access the csv file.
   - The CSV file will now be saved in your working directory as 'eda.csv'.
  
1. Now the user can use the main.ipynb file to access the saved 'eda.csv' file. Allowing for investigations of the data to begin. 
  
1. The difference between main.ipynb and the .py files is readability and speed.
   - The .ipynb file is better for those wanting a good visual representation of all the data. It goes through the code in sections, so each block of code is running individually. But there are still explainations in the notes and markdown boxes along the way.

   - The .py files are better for those who want to use the same code and understand it better by reading the docstrings (the order of .py files for user ease: db_utils.py, dataframe_info.py, DataTransformation.py, plot.py). The .py files also use 
   ```
   if __name__ == "__main__":
   ``` 
   to encapsulate important parts of code, but it doesn't contain everything needed to analyse, clean and visualise all parts of the data. 

1. Running the code from 'main.ipynb' will take you chronolgically through the sections:
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
                    ├── dataframe_info.py
                    ├── DataTransformation.py
                    ├── db_utils.py
                    ├── plot.py
                    ├── main.ipynb
                    ├── README.md
                    ├── requirements.txt
                    ├── __pycache__/
                    └── Data/
                        ├── eda.csv
                        ├── full_loan_data.csv
                        ├── untransformed_loan_data.csv
                        ├── transformed_loan_data.csv
                        └── filtered_loan_data.csv
```

## License information:
