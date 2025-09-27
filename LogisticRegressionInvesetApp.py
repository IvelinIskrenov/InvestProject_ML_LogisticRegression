import yfinance as yf
import pandas as pd
import numpy as np

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