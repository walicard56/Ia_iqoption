import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from iq import get_data_needed, login
import time

# Ensure TensorFlow uses the GPU if available and sets memory growth
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
    print(e)

SEQ_LEN = 5  # How long the sequence is
FUTURE_PERIOD_PREDICT = 2  # How far into the future to predict

def classify(current, future):
    return 1 if float(future) > float(current) else 0

def preprocess_df(df):
    df = df.drop("future", axis=1)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    
    df = pd.DataFrame(df_scaled, index=indexes)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def train_data(initial=True):
    iq = login()
    df = get_data_needed(iq)
    
    df.ffill(inplace=True)
    df = df.loc[~df.index.duplicated(keep='first')]
    
    df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
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
    columns_to_drop = ['open', 'min', 'max', 'avg_gain', 'avg_loss', 'L14', 'H14', 'gain', 'loss']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns, inplace=True)
    
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    main_df = df
    main_df.ffill(inplace=True)
    main_df.dropna(inplace=True)

    main_df['target'] = list(map(classify, main_df['close'], main_df['future']))
    main_df.dropna(inplace=True)

    times = sorted(main_df.index.values)
    last_10pct = sorted(main_df.index.values)[-int(0.1 * len(times))]

    validation_main_df = main_df[(main_df.index >= last_10pct)]
    main_df = main_df[(main_df.index < last_10pct)]

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"sells: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION sells: {validation_y.count(0)}, buys : {validation_y.count(1)}")

    train_y = np.asarray(train_y)
    validation_y = np.asarray(validation_y)

    LEARNING_RATE = 0.001
    EPOCHS = 40
    BATCH_SIZE = 16
    NAME = f"{LEARNING_RATE}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-{EPOCHS}-{BATCH_SIZE}-PRED-{int(time.time())}"
    print(NAME)

    earlyStoppingCallback = EarlyStopping(monitor='loss', patience=3)
    model = Sequential([
        LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(128, return_sequences=True),
        Dropout(0.1),
        BatchNormalization(),
        LSTM(128),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    checkpoint = ModelCheckpoint("models/LSTM-best.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    if initial:
        history = model.fit(
            train_x, train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint, earlyStoppingCallback]
        )

    return "models/LSTM-best.keras"

def update_model(model, X, y):
    LEARNING_RATE = 0.001
    EPOCHS = 20  # Menor número de épocas para atualização
    BATCH_SIZE = 16
    
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    earlyStoppingCallback = EarlyStopping(monitor='loss', patience=3)

    # Ajustar o modelo com os novos dados
    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[earlyStoppingCallback]
    )

    return model

if __name__ == "__main__":
    model = train_data()
