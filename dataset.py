import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class GetDataset:
    def __init__(self):
        self.dataset = self.get_dataset()

    def get_dataset(self) -> pd.DataFrame:
        """ Method for obtaining a dataset """
        loan_df = pd.read_csv('dataset/Loan_Data.csv')
        loan_df.info()

        # Displaying metadata and dataset characteristics
        print('', 'First 5 rows:', loan_df.head().to_markdown(), sep='\n', end='\n\n')
        print('Property Area: ', loan_df.Property_Area.unique())
        print('Dependents: ', loan_df.Dependents.unique(), end='\n\n')

        # Handle dataset
        loan_df = self._handled_dataset(loan_df)
        print('Handled dataset (first 5 rows):', loan_df.head().to_markdown(), sep='\n', end='\n\n')
        print(loan_df.dtypes.to_markdown(), end='\n\n')

        # create and save diagrams
        self.visualizations(loan_df)

        return loan_df

    @staticmethod
    def _handled_dataset(df) -> pd.DataFrame:
        """ Method for processing a dataset """
        # Drop rows with missing values and drop column with ID
        df = df.drop(['Loan_ID'], axis=1)           # Drop column with Loan ID (not required)
        df = df.dropna().reset_index(drop=True)     # Drop rows with missing values
        print('Count total NaN at each column in a DataFrame:', df.isnull().sum(), sep='\n', end='\n\n')
        return df

    @staticmethod
    def visualizations(df) -> None:
        """ Method for additional visualizations (to cover all things) """
        os.makedirs('diagrams') if not os.path.exists(os.path.abspath('diagrams')) else False
        for file in os.scandir('diagrams'):
            os.remove(file.path)

        string_fields = df.select_dtypes(include=['object']).columns.tolist()
        num_fields = df.select_dtypes(include=['number']).columns.tolist()

        # Data distribution and missing value detection
        df[num_fields].hist(figsize=(10, 8))
        plt.suptitle('Data distribution and missing value detection')
        plt.tight_layout()
        plt.savefig('diagrams/distrib_data.png')
        plt.close()

        for feature in string_fields:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=feature, data=df)
            plt.title(f'Distribution of {feature}')
            plt.savefig(f'diagrams/distrib_{feature.lower()}.png')
        plt.close()

        # The relationship between 'ApplicantIncome' and 'LoanAmount' with a breakdown by 'Loan_Status'
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df)
        plt.title('Relationship between ApplicantIncome and LoanAmount by Loan Status')
        plt.xlabel('ApplicantIncome')
        plt.ylabel('LoanAmount')
        plt.legend(title='Loan Status')
        plt.savefig('diagrams/relationship_applicantincome_loanamount.png')
        plt.close()

        # Heatmap correlation for a set of numerical data
        numeric_data = df[num_fields]
        matrix_corr = numeric_data.corr()
        mask = np.triu(np.ones_like(matrix_corr, dtype=bool))

        plt.subplots(figsize=(10, 3))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(matrix_corr, mask=mask, cmap=cmap, annot=True)
        plt.xticks(rotation=45)
        plt.title('Теплова карта кореляції для набору числових даних')
        plt.tight_layout()
        plt.savefig('diagrams/correlation_num_data.png')
        plt.close()
