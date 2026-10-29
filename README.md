# Stock Price Prediction Using Python and Machine Learning

A web-based **Stock Price Predictor** built using Python, Flask, and Scikit-learn. The application uses Ridge Regression to estimate a stock's closing price based on its Open, High, Low, and Volume values.

## Project Overview

The project demonstrates how machine learning can be integrated into a Flask web application for stock market data analysis and price estimation. Users can enter a stock ticker, retrieve historical market data from Yahoo Finance, train a regression model, and explore predictions through an interactive interface.

**Note:** This application estimates the closing price for a trading day using that same day's Open, High, Low, and Volume values. It is not designed to forecast future stock prices.

## Features

- **Stock Ticker Support:** Enter supported ticker symbols such as `AAPL`, `TCS.NS`, or `GC=F`.
- **Live Market Data:** Retrieves available historical market data using Yahoo Finance.
- **Machine Learning:** Uses Ridge Regression to estimate closing prices.
- **Model Evaluation:** Displays evaluation metrics such as R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- **Interactive Predictions:** Adjust input values using interactive sliders to explore different estimates.
- **Price History Visualization:** Displays historical price data alongside the prediction.
- **Recent Predictions:** Shows a table of recent predictions.
- **Input Validation:** Validates user inputs before processing predictions.
- **JSON API:** Provides endpoints for prediction, stock data, and application health checks.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Market Data:** Yahoo Finance through the project's configured data-fetching library
- **Frontend:** HTML, CSS, JavaScript
- **Visualization:** The charting library configured in the project
- **Deployment:** Gunicorn

## Project Structure

```text
Stock-Price-Prediction-Using-Python-and-Machine-Learning/
├── app.py
├── train_model.py
├── utils.py
├── model.pkl
├── model_params.json
├── requirements.txt
├── Procfile
├── Stock_Price_Predictor.ipynb
├── README.md
├── .gitignore
├── static/
│   ├── app.js
│   └── style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── prediction.html
    └── 404.html
```

*This structure reflects the project files shown in the repository. Keep the listing synchronized with your actual files.*

## Installation and Setup

### Prerequisites

- Python 3.10 or another Python version compatible with the dependencies in `requirements.txt`
- pip package manager
- Internet connection for retrieving online market data

### 1. Clone the Repository

```bash
git clone https://github.com/Shidita/Stock-Price-Prediction-Using-Python-and-Machine-Learning.git
```

### 2. Open the Project Directory

```bash
cd Stock-Price-Prediction-Using-Python-and-Machine-Learning
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

Open the following address in your browser:

`http://127.0.0.1:5000`

## Model Training

To retrain the model using the project's training script, run:

```bash
python train_model.py --ticker GC=F
```

Replace `GC=F` with another supported ticker if the training script accepts it. The training process depends on the availability of historical market data and the implementation in `train_model.py`.

## API Endpoints

The application provides the following endpoints, as described by the project:

| Endpoint | Purpose |
|---|---|
| `/api/predict` | Accepts prediction inputs and returns a prediction response. |
| `/api/stock/<ticker>` | Retrieves stock information for the specified ticker. |
| `/health` | Checks application health. |

Refer to `app.py` for the exact HTTP methods, request parameters, response formats, and validation rules.

## Model Evaluation

The application reports the following metrics:

- **R² Score:** Indicates how well the model explains variation in the evaluated target values.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted closing prices.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more strongly.

Evaluation results depend on the selected ticker, dataset, and training procedure. No fixed accuracy or return is guaranteed.

## Deployment

The project includes a `Procfile` for deployment using Gunicorn.

The configured start command is:

```bash
gunicorn app:app
```

Use a hosting platform that supports Python web applications and configure its Python version and dependencies appropriately. Online market-data access may be subject to provider availability and rate limits.

## Limitations

- The estimated closing price uses inputs from the same trading day, including its High, Low, and Volume. These values may not all be known before the market closes.
- The application does not inherently predict the next trading day's closing price.
- Historical data quality and online data availability can affect results.
- Model evaluation metrics do not guarantee future performance.
- Predictions should not be treated as guaranteed or reliable investment outcomes.

## Future Improvements

- Develop a genuine next-day forecasting model using only information available at prediction time.
- Add time-series features and compare multiple machine learning models.
- Improve validation using chronological train/test splits and walk-forward evaluation.
- Add more visualization and historical performance analysis.
- Improve error handling and caching for online market-data requests.

## Disclaimer

This project is intended for **educational and demonstration purposes only**. It is not financial advice, and its predictions should not be used as the sole basis for investment decisions.

## Author

**Shidita Tiwari**

GitHub: [@Shidita](https://github.com/Shidita)
