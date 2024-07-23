This automated trading bot was developed to operate in the financial market using LSTM neural networks with data provided by IqOption. It is designed to make price predictions and execute buying and selling operations in an automated way, with the ability to continuously learn and adjust based on the results of operations.

Functionalities
Data Preprocessing:

The bot collects data from various financial assets and applies a series of transformations, including normalization and calculation of technical indicators such as moving averages (MA), Relative Strength Index (RSI), and Stochastic (%K and %D).
Prediction with LSTM:

It uses an LSTM neural network to make predictions based on sequences of historical data. The network architecture is configured to capture temporal patterns and market trends.
Execution of Operations:

Based on the predictions, the bot executes buy (CALL) or sell (PUT) operations on IqOption. Execution is done automatically, using integrated functions to place bets and check the results of operations.
Continuous Learning:

The bot is designed to continually learn from operation data. It stores the results of operations and updates the model periodically, allowing it to adjust and improve its accuracy over time.
Periodic Model Update:

A separate thread is responsible for periodically updating the model with new operation data. This ensures that the model is always updated and improved based on the latest results.
Code Structure
Preprocessing:

The preprocess_prediction(iq) function collects and processes asset data, calculating the necessary indicators and normalizing the data for input into the LSTM model.
Model and Training:

The LSTM model is defined and trained using historical data. The train_data() function is responsible for initializing and training the model with historical data.
The update_model_periodically() function periodically updates the model with new operation data.
Execution and Monitoring:

The bot's main loop (main()) performs buy and sell operations based on model predictions and stores the results for future model updating.
How to use
Initial setting:

Clone this repository and install the necessary dependencies.
Configure IqOption credentials in the script.
Model Training:

Run the script to train the initial model using historical data.
Bot Execution:

Launch the bot so that it automatically starts operating in the market, making predictions and executing buy and sell operations.
Continuous Update:

The bot will continue to learn and update the model based on the results of operations, improving its accuracy over time.
Requirements
Python 3.7 or higher
TensorFlow
Pandas
NumPy
scikit-learn
Contributions
Feel free to contribute improvements, bug fixes or new features. Open an issue or send a pull request with your suggestions.
