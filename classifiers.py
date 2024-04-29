from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataset import GetDataset


class LoanClassifier(GetDataset):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.pipelines = self._build_pipeline(classifiers_dict)

    def _build_pipeline(self, classifiers_dict: dict) -> {Pipeline}:
        """
        Build the machine learning pipeline.
        Preprocesses the data, sets up feature transformations, and defines the classifier.
        """
        # Separate features and target variable
        x = self.dataset.drop('Loan_Status', axis=1)
        y = self.dataset['Loan_Status']

        # Split data into training and test sets
        (self.x_train, self.x_test,
         self.y_train, self.y_test) = train_test_split(x, y, test_size=0.2, random_state=42)

        # Pre-processing pipeline for numeric features
        numeric_features = x.select_dtypes(include=['number']).columns.tolist()
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        # Pre-processing pipeline for categorical features
        categorical_features = x.select_dtypes(include=['object']).columns.tolist()
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

        # Combine pre-processing pipeline
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                       ('cat', categorical_transformer, categorical_features)])
        # Create the full pipelines with specified classifier
        pipelines = dict()
        for name, model in classifiers_dict.items():
            pipelines[name] = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        return pipelines
