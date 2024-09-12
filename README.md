# Bitcoin Price Prediction using CNN + LSTM and Random Forest

This project demonstrates how to predict Bitcoin prices using a combination of CNN + LSTM neural network models and a Random Forest Regressor. The project leverages historical data from the CoinRanking API to train and evaluate models for accurate predictions.

<br>

## 📑 Table of Contents
- [📃 Project Overview](#project-overview)
- [📊 Data Source](#data-source)
- [📈 Models Used](#models-used)
  - [CNN + LSTM](#cnn--lstm)
  - [Random Forest Regressor](#random-forest-regressor)
- [⚙️ Model Training Process](#model-training-process)
- [🚀 Installation](#installation)
- [💻 Usage](#usage)
- [🏆 Results](#results)
- [🤝 Contributing](#contributing)

<br>

## 📃 Project Overview

The goal of this project is to predict the closing price of Bitcoin using two different models: 
1. A hybrid CNN + LSTM model, which is well-suited for time series data.
2. A Random Forest Regressor, a robust ensemble learning method.

These models are trained using historical Bitcoin price data retrieved from the CoinRanking API over a period of 5 years. Predictions are made using a combination of these models, and their performance is evaluated based on common metrics like MAE, MSE, and MAPE.

<br>

## 📊 Data Source

The data is fetched using the CoinRanking API, which provides historical price data for Bitcoin. The historical data includes:
- **Timestamps**
- **Closing prices**

### API Details:
- **Endpoint:** `/coin/{coin_id}/history`
- **Time Period:** 5 years
- **Coin ID for Bitcoin:** `Qwsogvtv82FCd`

<br>

## 📈 Models Used

### CNN + LSTM
This model is particularly effective for sequential data. The architecture includes:
- **LSTM Layer**: To capture temporal dependencies in the data.
- **Dense Layers**: To perform the final regression after feature extraction.
- **Dropout and Batch Normalization**: To prevent overfitting and ensure stable training.
- **Weighted MSE Loss**: The custom loss function emphasizes recent data points.

### Random Forest Regressor
An ensemble method that operates by constructing multiple decision trees during training and outputting the mean prediction of individual trees.

## ⚙️ Model Training Process

### Data Preprocessing
- **Scaling**: The `MinMaxScaler` is used to normalize the Bitcoin prices for better training performance.
- **Data Splitting**: 60% of the data is used for training, and 40% is reserved for testing.
- **Sliding Window Approach**: A window of 60 days' worth of data is used to predict the next day's closing price.

### CNN + LSTM Model
1. **Architecture**: 
   - LSTM layers with 50 units.
   - Dense layers with `ReLU` activation and regularization.
   - Dropout layers to reduce overfitting.
2. **Optimizer**: Stochastic Gradient Descent (SGD) with dynamic learning rate and momentum.
3. **Loss Function**: A custom weighted MSE loss is used to prioritize recent data points.

### Random Forest Model
- **Parameters**: 
  - 100 trees.
  - Maximum depth of 10.
  - Minimum samples split of 5.

### Cross-Validation
- **K-Fold Cross-Validation** (with `k=10`) is employed to ensure the robustness of the Random Forest model.

### Evaluation Metrics
The following metrics are used to evaluate the performance of both models:
- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**

<br>
  
## 🚀 Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bitcoin-price-prediction.git
   cd bitcoin-price-prediction
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the CoinRanking API**:
   Update the `x-access-token` in the code with your own API token from [CoinRanking](https://developers.coinranking.com).


<br>

## 💻 Usage

1. **Run the model training**:
   ```bash
   python model_training.py
   ```

2. **View the plots**:
   - The real vs predicted prices will be plotted and saved as PNG files in the project directory.

3. **Analyze Results**:
   - The CNN + LSTM model and the Random Forest model's performance metrics will be printed in the console.

<br>

## 🏆 Results

### CNN + LSTM
- **MAE**: 0.1460
- **MSE**: 0.0267
- **MAPE**: 1.8463
- **ACCURACY**: 98.15%

### Random Forest
- **MAE**: 0.0935
- **MSE**: 0.0121
- **MAPE**: 0.8317
- **ACCURACY**: 99.16%


![rf_cnn_lstm_price_prediction](https://github.com/user-attachments/assets/31f7cde9-5b7b-48d0-a7db-bf473920b018)


The comparison between the two models is visualized in the generated plot that displays real prices, CNN + LSTM predictions, and Random Forest predictions.

<br>

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
