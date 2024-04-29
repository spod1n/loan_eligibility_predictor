from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from classifiers import LoanClassifier

if __name__ == '__main__':
    RANDOM_STATE = 42
    classifiers = {
        'RandomForest': RandomForestClassifier(RANDOM_STATE),
        'KNeighbors': KNeighborsClassifier(RANDOM_STATE)
    }

    # Create and run the classifier
    classifier = LoanClassifier(classifiers)
    print(classifier)