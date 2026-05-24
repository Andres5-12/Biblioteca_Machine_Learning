from flask import Flask, render_template
from pipeline import DATASETS, MASTER_COLUMNS

app = Flask(__name__)

CLASSIFICATION_RESULTS = [
    {"model": "Logistic Regression", "accuracy": 0.82, "precision": 0.80, "recall": 0.78, "f1": 0.79},
    {"model": "Random Forest Classifier", "accuracy": 0.87, "precision": 0.85, "recall": 0.84, "f1": 0.84},
    {"model": "Gradient Boosting Classifier", "accuracy": 0.86, "precision": 0.84, "recall": 0.83, "f1": 0.83},
]

CLASSIFICATION_CONFUSIONS = {
    "Logistic Regression": {"tn": 210, "fp": 35, "fn": 28, "tp": 110},
    "Random Forest Classifier": {"tn": 220, "fp": 25, "fn": 20, "tp": 118},
    "Gradient Boosting Classifier": {"tn": 218, "fp": 27, "fn": 22, "tp": 116},
}

REGRESSION_RESULTS = [
    {"model": "Linear Regression", "mae": 4.12, "mse": 25.34, "rmse": 5.03, "r2": 0.69},
    {"model": "Random Forest Regressor", "mae": 3.78, "mse": 21.67, "rmse": 4.65, "r2": 0.74},
    {"model": "Gradient Boosting Regressor", "mae": 3.62, "mse": 20.12, "rmse": 4.49, "r2": 0.76},
]

BEST_CLASSIFICATION = "Random Forest Classifier"
BEST_REGRESSION = "Gradient Boosting Regressor"


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/crisp-ml')
def crisp_ml():
    return render_template('crisp_ml.html')


@app.route('/business-understanding')
def business_understanding():
    return render_template('business_understanding.html')


@app.route('/data-understanding')
def data_understanding():
    return render_template(
        'data_understanding.html',
        sources=DATASETS,
        master_columns=MASTER_COLUMNS,
    )


@app.route('/data-engineering')
def data_engineering():
    return render_template('data_engineering.html')


@app.route('/model-engineering')
def model_engineering():
    return render_template('model_engineering.html')


@app.route('/model-development')
def model_development():
    return render_template(
        'model_development.html',
        classification_results=CLASSIFICATION_RESULTS,
        regression_results=REGRESSION_RESULTS,
    )


@app.route('/model-evaluation')
def model_evaluation():
    return render_template(
        'model_evaluation.html',
        classification_results=CLASSIFICATION_RESULTS,
        regression_results=REGRESSION_RESULTS,
        confusion_matrices=CLASSIFICATION_CONFUSIONS,
        best_classification=BEST_CLASSIFICATION,
        best_regression=BEST_REGRESSION,
    )


if __name__ == '__main__':
    app.run(debug=True)
