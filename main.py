import pandas as pd
import joblib
import subprocess

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from classifiers import LoanClassifier


def select_classifier():
    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }
    selected_val = dict()

    while True:
        print('> Select Model')
        options_count = len(classifiers)
        options_list = list(classifiers.keys()) + ['All', 'Exit']

        for index, key in enumerate(options_list, 1):
            print(f'[{index}] {key}')
        val = input('> ')

        if val.isdigit():
            val_int = int(val)
            if 1 <= val_int <= options_count:
                selected_key = list(classifiers.keys())[val_int - 1]
                selected_val[selected_key] = classifiers[selected_key]
                return selected_val
            elif val_int == options_count + 1:
                return classifiers
            elif val_int == options_count + 2:
                return dict()
            else:
                print(inv_inp)
        else:
            print(inv_inp)


def run_predict(data_df: pd.DataFrame, model_name: str = 'RandomForest'):
    """
    Make predictions on new data using the trained model.
    Args: data (DataFrame): New data to make predictions on.
    Returns: ndarray: Predicted labels for the new data.
    """
    read_model = joblib.load(f'pipelines/{model_name.lower()}.pkl')
    return read_model.predict(data_df)[0]


if __name__ == '__main__':
    inv_inp = 'Invalid Input!'
    while True:
        print('', '> Menu', '[1] Training Models', '[2] Predict Data', '[3] Show Diagrams', '[4] Exit', sep='\n')
        val = input('> ')

        match val:
            case '1':
                if selected_classifier := select_classifier():
                    # Create pipeline and run the classifier
                    classifier = LoanClassifier(selected_classifier)
                    classifier.train_and_validate()
                    classifier.fine_tune()
                    classifier.evaluate_model()
                    del classifier
                else:
                    break

            case '2':
                y = '[1] Yes'
                n = '[2] No'
                data = dict()

                if models := select_classifier():
                    try:
                        print('Select Gender:', '[1] Male', '[2] Female', sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Gender'] = 'Male'
                            case '2':
                                data['Gender'] = 'Female'
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Married:', y, n, sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Married'] = 'Yes'
                            case '2':
                                data['Married'] = 'No'
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Dependents:', '[1] One', '[2] Two', '[3] Three +', '[4] Null', sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Dependents'] = '1'
                            case '2':
                                data['Dependents'] = '2'
                            case '3':
                                data['Dependents'] = '3+'
                            case '4':
                                data['Dependents'] = '0'
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Education:', '[1] Graduate', '[2] Not Graduate', sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Education'] = 'Graduate'
                            case '2':
                                data['Education'] = 'Not Graduate'
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Self Employed:', y, n, sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Self_Employed'] = 'Yes'
                            case '2':
                                data['Self_Employed'] = 'No'
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Credit History:', y, n, sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Credit_History'] = 1
                            case '2':
                                data['Credit_History'] = 0
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Property Area:', '[1] Urban', '[2] Rural', '[3] Semiurban', sep='\n')
                        val = input('> ')
                        match val:
                            case '1':
                                data['Property_Area'] = 'Urban'
                            case '2':
                                data['Property_Area'] = 'Rural'
                            case '3':
                                data['Property_Area'] = 'Semiurban'
                            case _:
                                print(inv_inp)
                                continue

                        print('Select Applicant Income (150 - 81_000):', sep='\n')
                        val = int(input('> '))
                        if 150 <= val <= 81_000:
                            data['ApplicantIncome'] = val
                        else:
                            print(inv_inp)
                            continue

                        print('Select Coapplicant Income (0 - 50_000):', sep='\n')
                        val = int(input('> '))
                        if 0 <= val <= 50_000:
                            data['CoapplicantIncome'] = val
                        else:
                            print(inv_inp)
                            continue

                        print('Select Loan Amount (9 - 700):', sep='\n')
                        val = int(input('> '))
                        if 9 <= val <= 700:
                            data['LoanAmount'] = val
                        else:
                            print(inv_inp)
                            continue

                        print('Select Loan Amount Term (12 - 480):', sep='\n')
                        val = int(input('> '))
                        if 12 <= val <= 480:
                            data['Loan_Amount_Term'] = val
                        else:
                            print(inv_inp)
                            continue

                    except ValueError:
                        print(inv_inp)
                        data = dict()

                    if data:
                        print('Selected Parameters: ', data, sep='\n')
                        for name in models.keys():
                            result = run_predict(pd.DataFrame(data, index=[0]), name)
                            result = True if result else False
                            print(f'Loan Status ({name}): {result}')
            case '3':
                subprocess.Popen(['explorer', 'diagrams'])
            case '4':
                break
            case _:
                print(inv_inp)
