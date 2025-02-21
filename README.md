# FirstPythonProj

# Stock Price Prediction

## Overview
This project focuses on predicting stock prices using a Long Short-Term Memory (LSTM) neural network. The dataset consists of stock price data from various companies over five years, and the model is trained to predict the closing price of Apple (AAPL) stocks.

## Technologies Used
- **Python**
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Matplotlib & Seaborn** for data visualization
- **TensorFlow & Keras** for deep learning model implementation
- **Scikit-Learn** for data preprocessing

## Dataset
The dataset is loaded from a CSV file containing stock prices over five years. The structure includes columns such as:
- `date`: The date of the stock data.
- `open`: Opening price of the stock.
- `close`: Closing price of the stock.
- `volume`: Number of stocks traded.
- `Name`: Stock ticker symbol.

## Data Preprocessing
1. **Load Data**: The dataset is read using `pandas.read_csv()`.
2. **Check Dataset Shape**: `data.shape` is used to determine the number of rows and columns.
3. **Random Sampling**: `data.sample(7)` selects seven random rows for review.
4. **Convert Date Column**: `pd.to_datetime(data["date"])` ensures the date column is in datetime format.
5. **Filtering Apple Stock Data**:
   - Stocks from Apple (`AAPL`) are extracted for model training.
   - Data is filtered to include only stock prices from 2013 to 2018.

## Exploratory Data Analysis (EDA)
- Line plots of stock price trends are created for various companies.
- A separate plot visualizes the trade volume over time.
- The `.loc[]` method is used for filtering data based on conditions.
- The `enumerate()` function is used for iterating over multiple companies while plotting.

## Data Scaling and Preparation
- The `MinMaxScaler` from `sklearn.preprocessing` scales stock prices between 0 and 1.
- The dataset is split into training (95%) and testing (5%).
- LSTM requires sequential data, so the last 60 days are used as features to predict the next day’s stock price.
- The data is reshaped into a 3D format `(samples, time steps, features)` for LSTM input.

## LSTM Model Architecture
- **Sequential Model**: A stack of layers with one input and one output tensor per layer.
- **Layers Used**:
  - Two LSTM layers with 64 units each.
  - A Dense layer with 32 units.
  - A Dropout layer (0.5) to prevent overfitting.
  - A final Dense layer with 1 unit for the stock price prediction.
- **Compilation**:
  - Optimizer: `adam`
  - Loss function: `mean_squared_error`
- **Training**:
  - The model is trained using `model.fit(x_train, y_train, epochs=10)`.

## Model Evaluation
- **Testing Data Preparation**:
  - The last 60 days of training data are included to maintain the correct sequence.
  - The test data is scaled similarly to training data.
  - Predictions are generated using `model.predict(x_test)`.
- **Performance Metrics**:
  - Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are computed to evaluate accuracy.

## Visualization of Predictions
- The model’s predictions are plotted against the actual stock prices.
- A line plot shows training data, test data, and predicted prices.

## Key Concepts Explained
### **Why Use LSTM?**
LSTMs are effective for time series forecasting because they retain long-term dependencies and prevent the vanishing gradient problem.

### **Understanding Shape Parameters**
- `.shape[0]`: Represents the number of rows (samples).
- `.shape[1]`: Represents the number of columns (features or time steps).

### **Why Use Dropout?**
Dropout randomly disables neurons during training, reducing overfitting and improving model generalization.

### **What is Overfitting?**
Overfitting occurs when the model learns noise in the training data, performing well on training but poorly on new data.

## Acknowledgments
- Code is based on an example project modified by Susobhan Akhuli.
- Explanations are derived from GeeksforGeeks and additional research on LSTM-based stock prediction.

## Future Improvements
- Extend the dataset to include more recent stock prices.
- Optimize the model architecture for better performance.
- Implement additional evaluation metrics beyond MSE and RMSE.
- Explore other deep learning models such as GRU or Transformer-based architectures.
- Add classes.
- Add testing modules for test driven development.
- Add the MSE and RMSE values to the page and see how they would change with a different dropout rate as 0.5 seems quite excessive.

---
This README provides a structured explanation of the stock price prediction project, covering data preparation, model training, evaluation, and key concepts.


