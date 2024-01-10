# %%
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
#%% 
class DataFrameTransform:

    def __init__(self, df_transform):
        self.df_transform = df_transform

    def num_of_nulls(self):
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
        self.df_transform.dropna(subset=column, inplace=True)
    
    def impute_zeros(self, column):
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(0)

    def impute_median(self, column):
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(self.df_transform[col].median())

    def impute_mean(self, column):
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(self.df_transform[col].mean())

    def log_transform(self, column):
        for col in column:
            self.df_transform[col] = self.df_transform[col].map(lambda i: np.log(i) if i > 0 else 0)
    
    def box_cox_transform(self, column):
        for col in column:
            boxcox_column = self.df_transform[col] +0.01
            a, b = stats.boxcox(boxcox_column)
            self.df_transform[col] = a 

    def remove_outliers_iqr(self, data, threshold=1.5):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    def remove_outliers_iqr_dataframe(self, column, threshold=1.5):
        filtered_dataframe = self.df_transform
        for col in column:
            filtered_dataframe[col] = self.remove_outliers_iqr(filtered_dataframe[col], threshold)
        return filtered_dataframe
    
    def save_transformed_data(self, filename='transformed_data.csv'):
        self.df_transform.to_csv(filename, index=False)

class Plotter:

    def __init__(self, df_transform):
        self.df_transform = df_transform

    def agostino_k2_test(self, col):
        stat, p = normaltest(self.df_transform[col], nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def qq_plot(self, col):
        self.df_transform.sort_values(by=col, ascending=True)
        qq_plot = qqplot(self.df_transform[col], scale=1, line='q', fit=True)
        pyplot.show()

    def histogram(self, col):
        self.df_transform[col].hist(bins=40)

    def density_plot(self, col):
        sns.histplot(data=self.df_transform[col], kde=True)
        sns.despine()

    def boxplot(self, col):
        fig = px.box(self.df_transform[col],width=600, height=500)
        fig.show()

    def scatter(self, col):
        sns.scatterplot(self.df_transform[col])
        
    def show_missing_nulls(self):
        msno.matrix(self.df_transform)
        pyplot.show()

    def multi_plot(self, col):
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
        pyplot.show()

    def find_skew(self, col):
        if self.df_transform[col].skew() >= 0.86:
            print('Skewed!')
            print(f' {self.df_transform[col].skew()} is over 0.85')
        else:
            print(f' {self.df_transform[col].skew()} is under 0.85')

    def multi_hist_plot(self, num_cols):
        sns.set(font_scale=0.7)
        f = pd.melt(self.df_transform, value_vars=num_cols)
        g = sns.FacetGrid(f, col="variable", col_wrap=4,
                          sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        pyplot.show()

    def multi_qq_plot(self, cols):
        remainder = 1 if len(cols) % 4 != 0 else 0
        rows = int(len(cols) / 4 + remainder)

        fig, axes = pyplot.subplots(
            ncols=4, nrows=rows, sharex=False, figsize=(12, 6))
        for col, ax in zip(cols, np.ravel(axes)):
            sm.qqplot(self.df_transform[col], line='s', ax=ax, fit=True)
            ax.set_title(f'{col} QQ Plot')
        pyplot.tight_layout()

    def show_outliers(self):
        #select only the numeric columns in the DataFrame
        df = self.df_transform.select_dtypes(include=['float64'])
        pyplot.figure(figsize=(18,14))

        for i in list(enumerate(df.columns)):
            fig_cols = 4
            fig_rows = int(len(df.columns)/fig_cols) + 1
            pyplot.subplot(fig_rows, fig_cols, i[0]+1)
            sns.boxplot(data=df[i[1]]) 

        # Show the plot
        pyplot.tight_layout()
        pyplot.show()

    def show_outliers_after_removal(self, dataframe, columns):
        pyplot.figure(figsize=(18, 14))

        for i, col in enumerate(columns):
            fig_cols = 4
            fig_rows = len(columns) // fig_cols + 1
            pyplot.subplot(fig_rows, fig_cols, i + 1)
            sns.boxplot(data=dataframe[col])

        pyplot.tight_layout()
        pyplot.show()
# %%
if __name__ ==  "__main__":
    table_of_loans = pd.read_csv('eda.csv')
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
    
    # Skewness of the data needs to be visualised:
    numerical_cols = ['loan_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'dti', 'annual_inc', 'total_payment', 'total_accounts', 'open_accounts', 'last_payment_amount']
    plot.multi_hist_plot(numerical_cols)
    plot.multi_qq_plot(numerical_cols)
    # which will print the density plot for all columns in the numerical_cols variable
    # and then will print the qq plot for all columns in the numerical_cols variable 

    # or use # following code for the:
    # mean, median, agostino_k2_test, qqplot, histogram and density plot together on a single column
    plot.multi_plot('instalment')
    plot.multi_plot('open_accounts')
    plot.multi_plot('total_rec_prncp')
    plot.multi_plot('total_payment')
    plot.multi_plot('total_rec_int')
    plot.multi_plot('out_prncp')
    plot.multi_plot('last_payment_amount')
    plot.multi_plot('inq_last_6mths')
    plot.multi_plot('annual_inc')
    plot.multi_plot('delinq_2yrs')
    plot.multi_plot('total_rec_late_fee')
    plot.multi_plot('recoveries')
    plot.multi_plot('collection_recovery_fee')
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


# %%
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


