# test_azure_job.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Generate a random classification dataset
    logging.info("Generating random dataset")
    X, y = make_classification(n_samples=1000,      # total samples
                               n_features=20,       # number of features
                               n_informative=15,    # informative features
                               n_redundant=5,       # redundant features
                               n_classes=2,         # binary classification
                               random_state=42)

    # Split the dataset into training and testing sets
    logging.info("Splitting dataset into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the Logistic Regression model
    logging.info("Initializing and training the model")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Perform predictions on the test set
    logging.info("Performing predictions on the test set")
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info("Model performance:")
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
