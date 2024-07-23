import threading
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import datetime
import time
import tensorflow as tf
from iq import fast_data, higher, lower, login, get_checkwin
from training import train_data, update_model
import sys

# Configure GPU memory growth
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
except Exception as e:
    print(e)

# Constants
SEQ_LEN = 5  # Sequence length for RNN
FUTURE_PERIOD_PREDICT = 2  # Prediction period

def preprocess_prediction(iq):
    Actives = ['EURUSD','GBPUSD','EURJPY','AUDUSD']
    main = pd.DataFrame()
    
    for active in Actives:
        if active == "EURUSD":
            main = fast_data(iq, active).drop(columns=['from', 'to'], errors='ignore')
        else:
            current = fast_data(iq, active)
            current = current.drop(columns=['from', 'to', 'open', 'min', 'max'], errors='ignore')
            current.columns = [f'close_{active}', f'volume_{active}']
            main = main.join(current, how='outer')
    
    df = main

    df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    
    if 'min' in df.columns and 'max' in df.columns:
        df['L14'] = df['min'].rolling(window=14).min()
        df['H14'] = df['max'].rolling(window=14).max()
        df['%K'] = 100 * ((df['close'] - df['L14']) / (df['H14'] - df['L14']))
        df['%D'] = df['%K'].rolling(window=3).mean()
    
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    rsi_period = 14
    chg = df['close'].diff(1)
    gain = chg.mask(chg < 0, 0)
    loss = chg.mask(chg > 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = abs(avg_gain / avg_loss)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Drop columns if they exist
    columns_to_drop = ['open', 'min', 'max', 'avg_gain', 'avg_loss', 'L14', 'H14', 'gain', 'loss', 'future']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns, inplace=True)
    
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    pred = pd.DataFrame(df_scaled, index=indexes)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in pred.iloc[len(pred) - SEQ_LEN:].values:
        prev_days.append(i)
        if len(prev_days) == SEQ_LEN:
            sequential_data.append(np.array(prev_days))
    
    X = np.array(sequential_data)  # Ensure X is a numpy array

    return X

def update_model_periodically(model, new_data, interval=3600):
    """Re-train and update the model periodically with new data"""
    while True:
        time.sleep(interval)
        if len(new_data) > 0:
            print("Updating model with new data...")
            X, y = preprocess_data(new_data)  # Supondo que você tenha uma função preprocess_data
            model = update_model(model, X, y)  # Atualize o modelo com os novos dados
            model.save("models/LSTM-best.keras")  # Salve o modelo atualizado

def preprocess_data(new_data):
    """Processa novos dados para o treinamento do modelo"""
    X = []
    y = []
    for data, operation_type, result in new_data:
        X.append(data)
        y.append([operation_type, result])
    return np.array(X), np.array(y)

def main():
    if len(sys.argv) == 1:
        bet_money = 1
        ratio = 'EURUSD'
    elif len(sys.argv) != 3:
        print("Usage: python testing.py CURRENCY INITIAL_BET")
        print("Example: python testing.py EURUSD 1")
        exit(-1)
    else:
        ratio = sys.argv[1]
        bet_money = int(sys.argv[2])
    
    # Get model name from training
    model_path = train_data()  # Certifique-se de que train_data() retorna um caminho
    model = tf.keras.models.load_model(model_path)
    
    iq = login()

    new_data = []
    
    update_thread = threading.Thread(target=update_model_periodically, args=(model, new_data))
    update_thread.start()

    while True:
        time_taker = time.time()
        pred_ready = preprocess_prediction(iq)
        pred_ready = pred_ready.reshape((pred_ready.shape[0], SEQ_LEN, pred_ready.shape[2]))
        result = model.predict(pred_ready)
        # Formatação das probabilidades para x.xx
        put_prob = format(result[0][0] * 100, ".2f")
        call_prob = format(result[0][1] * 100, ".2f")

        print(f'Probability of PUT: ', put_prob + '%')
        print(f'Probability of CALL: ', call_prob + "%")
        print(f'Time taken : {int(time.time() - time_taker)} seconds')
        
        if 50 <= datetime.datetime.now().second <= 59:
            if result[0][0] > 0.5:
                print('PUT')
                id = lower(iq, bet_money, ratio)
                
                while True:
                    trade_result, taxa = iq.check_win_v4(id)
                    if trade_result:
                        break
                if trade_result == 'win':
                    print('Trade result: WIN')
                    new_data.append((pred_ready[-1], 0, 1))  # 0 para SHORT, 1 para WIN
                else:
                    print('Trade result: LOSS')
                    new_data.append((pred_ready[-1], 0, 0))  # 0 para SHORT, 0 para LOSS

            elif result[0][0] < 0.5:
                print('CALL')
                id = higher(iq, bet_money, ratio)
                while True:
                    trade_result, taxa = iq.check_win_v4(id)
                    if trade_result:
                        break
                if trade_result == 'win':
                    print('Trade result: WIN')
                    new_data.append((pred_ready[-1], 1, 1))  # 1 para LONG, 1 para WIN
                else:
                    print('Trade result: LOSS')
                    new_data.append((pred_ready[-1], 1, 0))  # 1 para LONG, 0 para LOSS
            # Atualize o modelo com os novos dados periodicamente
        if len(new_data) > 2:  # Por exemplo, atualize após 100 dados novos
            new_X = np.array([data[0] for data in new_data])
            new_y = np.array([data[2] for data in new_data])
            model = update_model(model, new_X, new_y)
            new_data = []  # Limpa os dados após atualização

if __name__ == "__main__":
    main()
