import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt

class DataFrameTransform:
    '''
    This class is used to apply transformations such as imputing or removing columns, to the data.
    '''

    def __init__(self, df_transform):
        '''
        This method is used to initalise this instance of the DataFrameTransform class.

        Parameters:
        ----------
        df_transform: list
            loan payments information in a list.
        '''
        self.df_transform = df_transform

    def num_of_nulls(self):
        '''
        This method is used to count the number of missing values in each of the columns in the data.

        Returns:
        ----------
            The percentage of the number of missing values from each column.
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

        Parameters:
        ----------
        column: list
            List of specific column names that will be removed from the data.
        '''
        for col in column:
            self.df_transform.drop(col, axis=1, inplace=True)

    def drop_null_rows(self, column):
        '''
        This method is used to remove rows containing null or missing values.

        Parameters:
        ----------
        column: list
            List of specific column names that will have null rows removed from the data.
        '''
        self.df_transform.dropna(subset=column, inplace=True)
    
    def impute_zeros(self, column):
        '''
        This method is used to impute values to a value of zero.

        Parameters:
        ----------
        column: list
            List of specific column names that will have empty values imputed to 0.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(0)

    def impute_median(self, column):
        '''
        This method is used to impute values to the median value of the column.

        Parameters:
        ----------
        column: list
            List of specific column names that will have empty values imputed to the median value.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(self.df_transform[col].median())

    def impute_mean(self, column):
        '''
        This method is used to impute values to the mean value of the column.

        Parameters:
        ----------
        column: list
            List of specific column names that will have empty values imputed to the mean value.
        '''
        for col in column:
            self.df_transform[col] = self.df_transform[col].fillna(self.df_transform[col].mean())

    def log_transform(self, column):
        '''
        This method is used to transform the column values using a log transformation, to make the data less skewed.

        Parameters:
        ----------
        column: list
            List of specific column names that will have a log transformation performed on them.
        '''
        for col in column:
            log_sample = self.df_transform[col] = self.df_transform[col].map(lambda i: np.log(i) if i > 0 else 0)
            t=sns.histplot(log_sample,label="Skewness: %.2f"%(log_sample.skew()) )
            t.legend()
    
    def box_cox_transform(self, column):
        '''
        This method is used to transform the column values using a boxcox transformation, to make the data less skewed.

        Parameters:
        ----------
        column: list
            List of specific column names that will have a boxcox transformation performed on them.
        '''
        for col in column:
            boxcox_column = self.df_transform[col] + 0.01
            a, b = stats.boxcox(boxcox_column)
            self.df_transform[col] = a 
            t=sns.histplot(boxcox_column,label="Skewness: %.2f"%(boxcox_column.skew()) )
            t.legend()

    def remove_outliers_iqr(self, data, threshold=1.5):
        '''
        This method is used to remove outliers of data, using the Inter Quartile Range data to remove the data past the lower and upper quatiles.
        
        Parameters:
        ----------
        data: list
            The data set that will have outliers removed from.
        threshold: int
            The limit to calculate the upper/lower bounds.

        Returns:
        ----------
            Returns the data without the outliers.
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
        This method is used to remove outliers of column data after a log or boxcox transformation, using
        the Inter Quartile Range data to remove the data past the lower and upper quatiles.
        
        Parameters:
        ----------
        column: list
            List of specific column names that will have a outliers removed from them.
        threshold: int
            The limit to calculate the upper/lower bounds.

        Returns:
        ----------
            Returns the data without the outliers.
        '''
        filtered_dataframe = self.df_transform
        for col in column:
            filtered_dataframe[col] = self.remove_outliers_iqr(filtered_dataframe[col], threshold)
        return filtered_dataframe
    
    def save_transformed_data(self, filename='transformed_loan_data.csv'):
        '''
        This method is used to save the current dataframe as a new CSV file called 'transformed_loan_data.csv'.

        Parameters:
        ----------
        filename: str
            The name that will be given to the saved CSV file.
        '''
        self.df_transform.to_csv(filename, index=False)

    def save_untransformed_data(self, filename='untransformed_loan_data.csv'):
        '''
        This method is used to save the current dataframe as a new CSV file called 'transformed_loan_data.csv'
        
        Parameters:
        ----------
        filename: str
            The name that will be given to the saved CSV file.
        '''
        self.df_transform.to_csv(filename, index=False)


class Plotter:
    '''
    This class is used to find skewness of the data and plot the data in various forms.
    '''

    def __init__(self, df_transform):
        '''
        This method is used to initalise this instance of the Plotter class.
        
        Parameters:
        ----------
        df_transform: list
            loan payments information in a list.
        '''
        self.df_transform = df_transform

    def agostino_k2_test(self, col):
        '''
        This method is used to calculate the result of the agostino k2 test on a specific column.
        
        Parameters:
        ----------
        col: list
            The column that will be tested using the agostino k2 test.

        Returns:
        ----------
            The Statistics value and the p-value of the column after running the test.
        '''
        stat, p = normaltest(self.df_transform[col], nan_policy='omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def qq_plot(self, col):
        '''
        This method is used to plot a qqplot of a specific column.

        Parameters:
        ----------
        col: list
            The column that will be plotted using the qqplot.

        Returns:
        ----------
            A pyplot of the the columns qqplot.
        '''
        self.df_transform.sort_values(by=col, ascending=True)
        qq_plot = qqplot(self.df_transform[col], scale=1, line='q', fit=True)
        plt.show()

    def histogram(self, col):
        '''
        This method is used to plot a histogram of a specific column.

        Parameters:
        ----------
        col: list
            The column that will be plotted using a histogram.

        Returns:
        ----------
            A histogram plot of the data given.
        '''
        self.df_transform[col].hist(bins=40)

    def density_plot(self, col):
        '''
        This method is used to plot a density_plot of a specific column.

        Parameters:
        ----------
        col: list
            The column that will be plotted using a density plot.

        Returns:
        ----------
            A density plot of the data given.
        '''
        sns.histplot(data=self.df_transform[col], kde=True)
        sns.despine()

    def boxplot(self, col):
        '''
        This method is used to plot a boxplot of a specific column.

        Parameters:
        ----------
        col: list
            The column that will be plotted using a boxplot.

        Returns:
        ----------
            A boxplot of the data given.
        '''
        fig = px.box(self.df_transform[col],width=600, height=500)
        fig.show()
    
    def bar(self, data):
        '''
        This method is used to plot a bar plot of the data.

        Parameters:
        ----------
        data:
            The dataset that will be plotted using a bar plot.

        Returns:
        ----------
            A bar plot of the data given.
        '''
        df = pd.DataFrame(data, index=[0])
        sns.barplot(data=df)

    def scatter(self, col):
        '''
        This method is used to plot a scatter of a specific column.

        Parameters:
        ----------
        col: list
            The column that will be plotted using a scatter plot.

        Returns:
        ----------
            A scatter plot of the data given.
        '''
        sns.scatterplot(self.df_transform[col])

    def pie(self, col):
        '''
        This method is used to create a pie chart of a specific column.

        Parameters:
        ----------
        col: list
            The column that will be plotted using a pie chart.

        Returns:
        ----------
            A pie chart of the data given.
        '''
        data = self.df_transform[col].value_counts()

        fig = px.pie(values=data.values, names=data.index, title= 'Pie Chart of {self.df_transform[col]}', width=600)
        fig.show()
        
    def show_missing_nulls(self):
        '''
        This method is used to plot the missing values of the dataframe.

        Returns:
        ----------
            A matrix of the number of NUll/missing values.
        '''
        msno.matrix(self.df_transform)
        plt.show()

    def heatmap(self, col):
        '''
        This method is used to generate a heatmap. 
        It visualises the correlation between a specified column and other columns in the dataset.
        
        Parameters:
        ----------
        col: list
            The column that will have a corrolation visualised against the other columns in the dataset.
        '''
        corr = self.df_transform[col].corr() # Calculate the correlation matrix
        mask = np.zeros_like(corr) # Create a mask for the upper triangle of the heatmap
        mask[np.triu_indices_from(mask)] = True # Set the upper triangle values to True in the mask
        
        plt.figure(figsize=(10, 8)) # Set the figure size

        cmap = sns.diverging_palette(220, 10, as_cmap=True) # Define the color map

        sns.heatmap(corr, mask=mask, square=True, linewidths=5, annot=True, cmap=cmap) # Generate the heatmap

        # Show the plot
        plt.show()

    def multi_plot(self, col):
        '''
        This method is used to calculate the mean, median, skewness and result of the agostino k2 test.
        It also plots a blended histogram and density plot as well as plotting a qqplot of a specific column.
        
        Parameters:
        ----------
        col: list
            The column that will be plotted and measures of central tendency displayed.
        '''
        table_of_loans = pd.read_csv('untransformed_loan_data.csv')
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
        
        Parameters:
        ----------
        col: list
            The column that will have its skew measured.

        Returns:
        ----------
        int
            Description of skewness integer from specified column.
        '''
        if self.df_transform[col].skew() >= 0.86:
            print('Skewed!')
            print(f' {self.df_transform[col].skew()} is over 0.85')
        else:
            print(f' {self.df_transform[col].skew()} is under 0.85')

    def multi_hist_plot(self, num_cols):
        '''
        This method is used to plot a histogram of all specified columns.
        
        Parameters:
        ----------
        num_cols: list
            The columns that will be plotted in a histogram.
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

        Parameters:
        ----------
        cols: list
            The columns that will be plotted in a qqplot.
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
        
        Parameters:
        ----------
        dataframe: list
            loan payments information in a list.

        columns: list
            The columns that are avalible in the dataframe to be selected.
        '''
        plt.figure(figsize=(18, 14))

        for i, col in enumerate(columns):
            fig_cols = 4
            fig_rows = len(columns) // fig_cols + 1
            plt.subplot(fig_rows, fig_cols, i + 1)
            sns.boxplot(data=dataframe[col])

        plt.tight_layout()
        plt.show()

if __name__ ==  "__main__":
    # load the data in
    table_of_loans = pd.read_csv('full_loan_data.csv')
    df_cols = DataFrameTransform(table_of_loans)
    plot = Plotter(table_of_loans)

    plot.show_missing_nulls
    print(df_cols.num_of_nulls())

    df_cols.impute_zeros(['employment_length'])
    df_cols.impute_median(['int_rate'])
    df_cols.drop_null_rows(['last_payment_date', 'last_credit_pull_date'])
    df_cols.drop_column(['mths_since_last_delinq', 'next_payment_date', 'mths_since_last_record', 'mths_since_last_major_derog'])

    print(df_cols.num_of_nulls())