from flask import Flask, render_template
from pipeline import DATASETS, MASTER_COLUMNS

app = Flask(__name__)


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


if __name__ == '__main__':
    app.run(debug=True)