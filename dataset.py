import pandas as pd


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

        return loan_df

    @staticmethod
    def _handled_dataset(df) -> pd.DataFrame:
        """ Method for processing a dataset """
        # Drop rows with missing values and drop column with ID
        df = df.drop(['Loan_ID'], axis=1)  # Drop column with Loan ID (not required)
        df = df.dropna().reset_index(drop=True)  # Drop rows with missing values
        print('Count total NaN at each column in a DataFrame:', df.isnull().sum(), sep='\n', end='\n\n')
        return df
