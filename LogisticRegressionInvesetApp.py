import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import date

class DataHandler:
    '''
    Class for downloading financial data and generating technical indicators && Processing the data.
    '''
    def __init__(self, ticker, start, end, target_pct, horizon):
        '''
        Data parameters.
        Args. {}
            - ticker: The asset's ticker symbol.
            - start && end: The start and end date for the data in "YYYY-MM-DD".
            - target_pct (float):target profit percentage for defining the 'target'.
            - horizon: The number of days to look for the target to be hit.
        '''
        self.ticker = ticker
        self.start = start
        self.end = end
        self.target_pct = target_pct
        self.horizon = horizon
        self.df = None

    def download_data(self):
        '''
        Downloads historical asset price data - yfinance to get daily Open, High, Low, Close, and Volume data.
        Don't have missing values are removed.
        '''
        print("Downloading data ...")
        self.df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
        self.df = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    def create_features(self):
        '''
        Generates a set of technical indicators as 'features'.
        calculated based on data we have from yfinance.
        '''
        print("Generating features...")
        #Return-based 
        self.df['return_1'] = self.df['Close'].pct_change()
        self.df['r1'] = self.df['return_1']
        self.df['r2'] = self.df['return_1'].shift(1)
        self.df['r3'] = self.df['return_1'].shift(2)

        #Moving Averages
        self.df['ma5'] = self.df['Close'].rolling(5).mean()
        self.df['ma20'] = self.df['Close'].rolling(20).mean()
        self.df['ma5_20'] = self.df['ma5'] - self.df['ma20']

        #Volatility and momentum 
        self.df['vol10'] = self.df['return_1'].rolling(10).std()
        self.df['momentum10'] = self.df['Close'] / self.df['Close'].shift(10) - 1
        self.df['vol_med20'] = self.df['Volume'].rolling(20).median()

        #Simplified RSI
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        self.df['rsi14'] = 100 - (100 / (1 + rs))

        self.df.dropna(inplace=True) #remove the values with NAN VALUE

    def create_target(self):
        print("Creating target variable...")
        closes = self.df['Close'].values
        highs = self.df['High'].values
        labels = []
        n = len(self.df)
        for i in range(n):
            target_hit = False
            for d in range(1, self.horizon + 1):
                j = i + d
                if j >= n:
                    break
                if highs[j] >= closes[i] * (1 + self.target_pct):
                    target_hit = True
                    break
            labels.append(1 if target_hit else 0)
        self.df['target'] = labels

    def get_processed_data(self):
        self.download_data()
        self.create_features()
        self.create_target()
        #remove NaNs that might have appeared from shift functions after feat. creat.
        self.df.dropna(inplace=True)
        return self.df
    
class LogisticRegressionModel:
    '''
    Training and testing a model with an expanding window (evaluates its performance on a subsequent, fixed-size window)
    Uses past data to retrain it!!!
    '''

    def __init__(self, data_df, features_list, target_col, train_window, predict_window):
        '''
        Set up for model
            data_df        - All the data we've prepared with our indicators
            features_list  - The list of column names the model will look at to learn
            target_col     - The name of the column the model is trying to predict
            train_window   - How many days of past data the model will use to train
            predict_window - How many days into the future it will try to predict
        '''
        self.data_df = data_df.copy()
        self.features_list = features_list
        self.target_col = target_col
        self.train_window = train_window
        self.predict_window = predict_window
        self.predictions = pd.DataFrame()
        #avoid re-creation
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='saga', max_iter=2000, C=1))
        ])

    def run_training_and_prediction(self):
        '''
        Trains the model on past data and predicts the future - uses a "rolling window"
        '''
        print("Starting exp.window training...")
        
        n = len(self.data_df)
        start_idx = self.train_window
        
        while start_idx < n:
            end_idx = min(start_idx + self.predict_window, n)
            
            train_df = self.data_df.iloc[:start_idx]
            test_df = self.data_df.iloc[start_idx:end_idx]
            
            X_train = train_df[self.features_list]
            y_train = train_df[self.target_col]
            X_test = test_df[self.features_list]
            y_test = test_df[self.target_col]
            
            self.model.fit(X_train, y_train)
            
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            temp_results = pd.DataFrame({
                'prob': y_prob,
                'actual': y_test.values
            }, index=test_df.index)
            self.predictions = pd.concat([self.predictions, temp_results])
            
            #print(f"Processed data up to {test_df.index[-1].strftime('%Y-%m-%d')}")
            
            start_idx = end_idx
            
if __name__ == '__main__':
    # 1. Parameters
    TICKER = "META" # APPL, MSFT, ...
    START = "2020-01-01"
    END = date.today().strftime("%Y-%m-%d") 
    TARGET_PCT = 0.02
    HORIZON = 30
    FEATURES = ['r1', 'r2', 'r3', 'ma5_20', 'vol10', 'momentum10', 'rsi14']

    #params for roll training
    TRAIN_WINDOW_SIZE = 500
    PREDICT_WINDOW_SIZE = 20 # 20 to 60?

    try:
        #Data processing
        data_processor = DataHandler(
            ticker = TICKER,
            start = START,
            end = END,
            target_pct = TARGET_PCT,
            horizon = HORIZON
        )
        df = data_processor.get_processed_data()
        
        print(df.sample(5))
        #Expanding window training and prediction
        roller = LogisticRegressionModel(
            data_df = df,
            features_list = FEATURES,
            target_col = 'target',
            train_window = TRAIN_WINDOW_SIZE,
            predict_window = PREDICT_WINDOW_SIZE
        )
        roller.run_training_and_prediction()
        
    except Exception as e:
        print(f"An error occurred: {e}")    