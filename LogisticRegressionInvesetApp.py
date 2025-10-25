import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import date
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

#STANDERTIZE ?
#check samples !


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
        self.data_df = data_df
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

    def load_data(self):
        data = DataHandler() 
        self.data_df = data.get_processed_data()
        
    def data_analysis(self):
        '''Describe data, info and see the correlations between the data'''
        print(f"Data analysis started ...")
        print("Describe data: ")
        print(self.data_df.describe())
        print("Data info: ")
        print(self.data_df.info())
        
        corr = self.data_df.corr()
        
        #to see the correclation between data
        correlation_values = self.data_df.corr()['target'].drop('target')
        correlation_values.plot(kind="barh",figsize=(10, 6))
        plt.show()
        
        #abs(correlation_values).sort_values(ascending=False)[:10] #!!!
    
    def run_training_and_prediction(self):
        '''
        Trains the model on past data and predicts the future - uses a "rolling window"
        '''
        #Calculate prediction error (cost function)
        #Update teta to reduce prediction error
        #Repeat until - reach samll log-loss value or target number of iterations  / Can try Gradient D.
        
        
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
            
    def evaluate(self):
        '''
        Evaluates the collected predictions.
        '''
        # Precision - FP are expensive
        if self.predictions.empty:
            print("No predictions collected. Please run run_training_and_prediction() first.")
            return

        print("\nEvaluating collected predictions...")
        y_prob = self.predictions['prob']
        y_test = self.predictions['actual']
        y_pred = (y_prob > 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:\n", conf_matrix)
        
        return y_prob, y_test
    
class BuySellSimulator:
    '''
    Class for simulating a trading strategy and evaluating its performance
    Executes a buy simulation based on the model's predictions
    '''

    def __init__(self, data_df, y_prob, horizon, target_pct, target_col):
        '''
        set up for Simulator
            data_df    - all of our processed data
            y_prob     - the model's prediction probabilities
            horizon    - how many days to check for a winning trade
            target_pct - the profit goal we're testing
            target_col - the name of our target column
        '''
        #chech that df and y_prob have the same indices
        self.data_df = data_df.loc[y_prob.index]
        self.y_prob = y_prob
        self.horizon = horizon
        self.target_pct = target_pct
        
    def run_backtest(self, threshold=0.5, transaction_cost=0.001):
        '''
        Executes the strategy simulation - threshold && transaction_cost
        '''
        print("\nRunning backtest...")
        rets = []
        wins = 0
        trades = 0

        #looping through each day in the test set
        for idx in range(len(self.data_df)):
            prob = self.y_prob.iloc[idx]
            
            #Enter a trade only if the probability is above the threshold
            if prob <= threshold:
                continue
            
            #Enter on the Open of the next day
            entry_idx = idx + 1
            if entry_idx >= len(self.data_df):
                continue
            
            entry_price = float(self.data_df.iloc[entry_idx]['Open'])
            if np.isnan(entry_price) or np.isinf(entry_price):
                continue
            
            #Look for an exit price (profit or stop-loss)
            hit = False
            hit_price = None
            for d in range(1, self.horizon + 1):
                check_idx = entry_idx + d - 1
                if check_idx >= len(self.data_df):
                    break
                    
                high = float(self.data_df.iloc[check_idx]['High'])
                if high >= entry_price * (1 + self.target_pct):
                    hit_price = entry_price * (1 + self.target_pct)
                    hit = True
                    break
            
            #Calculating   the return
            if hit:
                realized_return = (hit_price / entry_price - 1) - 2 * transaction_cost
                wins += 1
            else:
                exit_idx = min(entry_idx + self.horizon - 1, len(self.data_df) - 1)
                exit_price = float(self.data_df.iloc[exit_idx]['Close'])
                realized_return = (exit_price / entry_price - 1) - 2 * transaction_cost
                
            trades += 1
            rets.append(realized_return)

        if trades == 0:
            return {"trades": 0, "message": "No trades were made at this threshold."}

        #Collect the results
        rets = np.array(rets)
        avg_return = rets.mean()
        cumulative_return = (1 + rets).prod()
        sharpe = avg_return / (rets.std() + 1e-9) * np.sqrt(252) if rets.std() > 0 else 0
        
        return {
            "trades": trades,
            "win_rate": wins / trades,
            "avg_return_per_trade": avg_return * 100,
            "cumulative_return": cumulative_return,
            "sharpe_ratio": sharpe
        }
 
class InteractivePredictor:
    '''
    Class for interactive prediction based on a trained model - (Allows the user to input horizon and percentage)
    '''

    def __init__(self, trained_model, full_data, features_list, horizon):
        '''
        set up :
            trained_model: The model ready to make predictions.
            full_data: All of our processed data.
            features_list: The names of the columns the model uses.
            horizon: How many days the model was trained to predict.
        '''
        self.model = trained_model
        self.data = full_data.copy()
        self.features_list = features_list
        self.horizon = horizon 
        
    def predict_for_today(self, target_pct_final):
        '''
        Forecasts the next 5 days.
            arg: target_pct_final (float): The percentage gain we're trying to predict.
        '''
        #Take - today's date (last row)
        X_predict = self.data.iloc[-1][self.features_list].to_frame().T

        y_prob = self.model.predict_proba(X_predict)[:, 1][0]

        prediction_result = "Yes" if y_prob >= 0.5 else "No"
        
        return y_prob, prediction_result
            
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
        roller.data_analysis()
        
        roller.run_training_and_prediction()
        roller.evaluate()
        
        print(df.sample(100))
        
        #Strategy testing
        backtester = BuySellSimulator(
            data_df = df,
            y_prob = roller.predictions['prob'],
            horizon = HORIZON,
            target_pct = TARGET_PCT,
            target_col = 'target'
        )
        backtest_results = backtester.run_backtest(threshold=0.5)

        print("\n--- Backtest Results ---")
        if backtest_results.get("trades", 0) > 0:
            print(f"Total trades: {backtest_results['trades']}")
            print(f"Win Rate: {backtest_results['win_rate']:.4f}")
            print(f"Average return per trade: {backtest_results['avg_return_per_trade']:.4f}")
            print(f"Total cumulative return: {backtest_results['cumulative_return']:.4f}")
            print(f"Annualized Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
        else:
            print(backtest_results["message"])

        print("\n--- Forecast for the next 5 days ---")
        last_trained_model = roller.model
        
        # Use the last trained model for prediction
        predictor = InteractivePredictor(
            trained_model = last_trained_model,
            full_data = df,
            features_list = FEATURES,
            horizon = HORIZON
        )

        #Predict for today's date with the parameters the model was trained with (5 days and 2%)
        prob, result = predictor.predict_for_today(TARGET_PCT)

        print(f"The model is trained to predict for {HORIZON} days and a {TARGET_PCT*100:.2f}% gain.")
        print(f"The forecast for the next {HORIZON} days is:")
        print(f"-> Will a {TARGET_PCT*100:.2f}% gain be achieved? {result}")
        print(f"-> Prediction value (probability): {prob:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")    
        
