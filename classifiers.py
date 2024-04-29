import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import GetDataset


class LoanClassifier(GetDataset):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.pipelines = self._build_pipeline(classifiers_dict)
        self.param_grid = {
            'RandomForest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'classifier__p': [1, 2]
            }
        }

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

        # Use it to transform the training and test target variables to number
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.y_train)
        self.y_test = label_encoder.transform(self.y_test)

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

    def train_and_validate(self) -> None:
        """ Train and validate the machine learning models on the full data """
        for name, pipeline in self.pipelines.items():
            # Train model
            pipeline.fit(self.x_train, self.y_train)
            # Validate model
            cv_scores = cross_val_score(pipeline, self.x_train, self.y_train, cv=5)
            print(f'Cross-validation scores for {name}: {cv_scores.mean()}', cv_scores, sep='\n', end='\n\n')

    def fine_tune(self) -> None:
        """ Fine-tune the RandomForestClassifier model using GridSearchCV """
        for name, pipeline in self.pipelines.items():
            # Define the parameter grid for GridSearchCV
            if name in self.param_grid:
                print(self.param_grid[name])
                # Initialize GridSearchCV with the pipeline and parameter grid
                grid_search = GridSearchCV(pipeline, self.param_grid[name], cv=5)
                # Perform the grid search to find the best parameters
                grid_search.fit(self.x_train, self.y_train)

                # Get the best parameters found by GridSearchCV
                best_params = grid_search.best_params_
                print(f'Best parameters for {name}: {best_params}')

                # Update the pipeline with the best parameters and
                # retrain the RandomForestClassifier model with the best parameters on the full data
                pipeline.set_params(**best_params)
                pipeline.fit(self.x_train, self.y_train)

                joblib.dump(pipeline, f'pipelines/{name.lower()}.pkl')

    def evaluate_model(self) -> None:
        """ Evaluate the machine learning models on the test data and print performance metrics """
        for name, pipeline in self.pipelines.items():
            y_pred = pipeline.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print('Model: ', name)
            print('Accuracy: ', accuracy, end='\n\n')
            print('Classification Report:', classification_report(self.y_test, y_pred), sep='\n', end='\n\n')
            print('Confusion Matrix:', confusion_matrix(self.y_test, y_pred), sep='\n', end='\n\n')
