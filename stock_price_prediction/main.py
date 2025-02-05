import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

warnings.filterwarnings("ignore")

data = pd.read_csv(
    r"..\data\raw\all_stocks_5yr\all_stocks_5yr.csv",
    delimiter=",",
    # The delimiter tells pandas how to split each row of data into different columns.
    # In this case, it tells the function that each value in a row is separated by a comma.
    on_bad_lines="skip",
    # By setting on_bad_lines="skip", you're telling pandas to ignore any problematic rows in the CSV and continue reading the rest of the file.
)
print(data.shape)
# It’s a quick way to check how large the dataset is
# The shape attribute of a DataFrame returns a tuple that represents the dimensions of the DataFrame:

# The first value of the tuple is the number of rows in the DataFrame.
# The second value of the tuple is the number of columns in the DataFrame.
print(data.sample(7))
#  The sample() method is used to randomly select a specified number of rows from the DataFrame.
# 7: This argument tells pandas to select 7 random rows from the data DataFrame.
# By default, sample() selects random rows without replacement, meaning that the same row won’t be selected twice. If you want to sample with replacement, you can set replace=True.
# print(data.sample(7)): This will print 7 random rows from the DataFrame data to the console. It's useful when you want to quickly view a small, random subset of your dataset.

data["date"] = pd.to_datetime(data["date"])
# data.info() is used to give us info and visualise our data in the interactive window

# date vs open
# date vs close

# Define the list of companies you want to plot
companies = ["AAPL", "AMD", "FB", "GOOGL", "AMZN", "NVDA", "EBAY", "CSCO", "IBM"]

plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    # enumerate(companies, 1) pairs each company in the list with an index starting at 1.
    # In each loop iteration, index holds the counter, and company holds the company name from the list.
    # So, enumerate() is a useful way to get both the index and the item in a loop.
    plt.subplot(3, 3, index)
    c = data[data["Name"] == company]
    plt.plot(c["date"], c["close"], c="r", label="close", marker="+")
    plt.plot(c["date"], c["open"], c="g", label="open", marker="^")
    plt.title(company)
    plt.legend()
    plt.tight_layout()

# Plots the volume of trade for these 9 stocks as well as a function of time.
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data["Name"] == company]
    plt.plot(c["date"], c["volume"], c="purple", marker="*")
    plt.title(f"{company} Volume")
    plt.tight_layout()

# Analyses the data for Apple Stocks from 2013 to 2018
apple = data[data["Name"] == "AAPL"]
prediction_range = apple.loc[
    # The .loc[] method in Pandas is used to access a group of rows and columns by labels or a boolean array. It allows you to select data by label (the row or column names) instead of the index positions.
    # datetime(year, month, day)
    (apple["date"] > datetime(2013, 1, 1)) & (apple["date"] < datetime(2018, 1, 1))
]
plt.plot(apple["date"], apple["close"])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")
plt.show()

close_data = apple.filter(["close"])
dataset = close_data.values
training = int(np.ceil(len(dataset) * 0.95))
# This calculates 95% of the total length of the dataset. The idea here is to split the dataset into a training set (95%) and a testing set (5%).
print(training)

scaler = MinMaxScaler(
    feature_range=(0, 1)
)  # Uses a fairly simple linear scaling formula that you can look at online
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[
    0 : int(training), :
]  # ":" means all of the columns are looked at
# Prepare feature and labels
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(
        train_data[i - 60 : i, 0]
    )  # Looking at the last 60 days of stock data
    y_train.append(train_data[i, 0])  # The next day's stock data

# Reshaping for LSTM: The input features are reshaped into the proper format for an LSTM model, which expects a 3D array.
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(
    keras.layers.LSTM(
        units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)
    )
)
# LSTM stands for Long Short-Term Memory, which is a type of Recurrent Neural Network (RNN) useful for sequential data (such as time series, text, etc.).
# units=64: This defines the number of LSTM units (or neurons) in this layer. It determines how much information the model can store and process in memory. More units can allow the model to learn more complex patterns.
# return_sequences=True: Since the LSTM layer is followed by another LSTM, this setting ensures that it returns the entire sequence of outputs (instead of just the final output) to pass to the next LSTM layer. If this is set to False, only the output from the last timestep would be passed on.
# input_shape=(x_train.shape[1], 1): This defines the shape of the input data. x_train.shape[1] corresponds to the number of time steps (60 in your case), and 1 is the number of features at each time step (since we're using a single feature, the 'close' stock price).
model.add(keras.layers.LSTM(units=64))
# This is another LSTM layer with 64 units, but without return_sequences=True. This means it only outputs the last hidden state (the output from the final time step), which will be passed to the next layer.
model.add(keras.layers.Dense(32))
# This is a Dense (fully connected) layer with 32 neurons. Each neuron in this layer is connected to all neurons in the previous layer (the LSTM layer). It is typically used to reduce or increase the dimensionality of the data.
# The model will learn weights for each connection between the neurons.
model.add(keras.layers.Dropout(0.5))
# Dropout is a regularization technique used to prevent overfitting. During training, dropout randomly sets a fraction (0.5 in this case) of the input units to zero at each update to prevent overfitting. This forces the model to not rely too heavily on any single neuron and generalizes better.
model.add(keras.layers.Dense(1))
# This is the output layer of the model with a single neuron (since you're predicting one value, the stock closing price). The activation function is implicitly linear (no activation function), which is typical for regression tasks where the output can be any real number.
model.summary()
# This gives an overview of the architecture

# optimizer – This is the method that helps to optimize the cost function by using gradient descent.
# loss – The loss function by which we monitor whether the model is improving with training or not.
# metrics – This helps to evaluate the model by predicting the training and the validation data.
model.compile(optimizer="adam", loss="mean_squared_error")
# Specifies how the model will be trained (which optimizer and loss function to use).
history = model.fit(x_train, y_train, epochs=10)
# Starts the training process by using the training data and iterating over it for a certain number of epochs.

test_data = scaled_data[training - 60 :, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i, 0])

x_test = np.array(x_test)
# The shape (x_test.shape[0], x_test.shape[1], 1) reshapes x_test to be a 3D array because LSTM layers expect input in this form: (samples, timesteps, features)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the testing data
predictions = model.predict(x_test)
# Applies the inverse transformation to convert the predictions back to their original range (real stock prices).
predictions = scaler.inverse_transform(predictions)


# Evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
# calculates the Mean Squared Error (MSE) between the predicted and true values.
print("MSE", mse)
print("RMSE", np.sqrt(mse))
# This calculates and prints the Root Mean Squared Error (RMSE).

# Visualising results
train = apple[:training]
test = apple[training:]
test["Predictions"] = predictions

plt.figure(figsize=(10, 8))
plt.plot(train["date"], train["close"])
plt.plot(test["date"], test[["close", "Predictions"]])
plt.title("Apple Stock Close Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.legend(["Train", "Test", "Predictions"])

# This code is modified by Susobhan Akhuli
