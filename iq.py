import logging
import time
from iqoptionapi.stable_api import IQ_Option
import pandas as pd

def login(verbose=False, iq=None, checkConnection=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    
    if iq is None:
        print("Trying to connect to IqOption")
        iq = IQ_Option('emelybeatriz695@gmail.com', 'wali1234')  # Replace with your username and password
        iq.connect()
    
    if iq is not None:
        while True:
            if not iq.check_connect():
                print('Error when trying to connect')
                print(iq)
                print("Retrying")
                iq.connect()
            else:
                if not checkConnection:
                    print('Successfully Connected!')
                break
            time.sleep(3)
    
    iq.change_balance("PRACTICE")  # or "REAL"
    return iq

def higher(iq, Money, Actives):
    done, id = iq.buy(Money, Actives, "call", 1)
    return id

def lower(iq, Money, Actives):
    done, id = iq.buy(Money, Actives, "put", 1)
    return id

def get_checkwin(iq, id):
    while True:
        trade_result, taxa = iq.check_win_digital_v2(id)
        if trade_result:
            return trade_result, taxa
        time.sleep(1)  # Sleep for a second before checking again

def get_candles(iq, Actives):
    login(iq=iq, checkConnection=True)
    return iq.get_candles(Actives, 60, 1000, time.time())

def get_all_candles(iq, Actives, start_candle):
    final_data = []
    for x in range(1):
        login(iq=iq, checkConnection=True)
        data = iq.get_candles(Actives, 60, 1000, start_candle)
        start_candle = data[0]['to'] - 1
        final_data.extend(data)
    return final_data

def get_data_needed(iq):
    start_candle = time.time()
    actives = ['EURUSD', 'GBPUSD', 'EURJPY', 'AUDUSD']
    final_data = pd.DataFrame()
    
    for active in actives:
        current = get_all_candles(iq, active, start_candle)
        main = pd.DataFrame()
        
        for candle in current:
            useful_frame = pd.DataFrame(list(candle.values()), index=list(candle.keys())).T.drop(columns=['at'])
            useful_frame = useful_frame.set_index(useful_frame['id']).drop(columns=['id'])
            main = pd.concat([main, useful_frame], ignore_index=True).drop_duplicates()
        
        if active == 'EURUSD':
            final_data = main.drop(columns=['from', 'to'])
        else:
            main = main.drop(columns=['from', 'to', 'open', 'min', 'max'])
            main.columns = [f'close_{active}', f'volume_{active}']
            final_data = final_data.join(main)
    
    final_data = final_data.loc[~final_data.index.duplicated(keep='first')]
    return final_data

def fast_data(iq, ratio):
    login(iq=iq, checkConnection=True)
    candles = iq.get_candles(ratio, 60, 300, time.time())
    main = pd.DataFrame()
    
    for candle in candles:
        useful_frame = pd.DataFrame(list(candle.values()), index=list(candle.keys())).T.drop(columns=['at'])
        useful_frame = useful_frame.set_index(useful_frame['id']).drop(columns=['id'])
        main = pd.concat([main, useful_frame], ignore_index=True)
    
    return main

def get_balance(iq):
    return iq.get_balance()

def get_profit(iq):
    return iq.get_all_profit()['EURUSD']['turbo']
