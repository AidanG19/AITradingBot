import ccxt  # for accessing market data and executing trades on supported exchanges
import pandas as pd
import ta  # for technical analysis indicators
import numpy as np

class TradingBot:
    def __init__(self, api_key, secret_key, symbol, timeframe='1h'):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        # Set sandbox mode to True for testnet
        self.exchange.set_sandbox_mode(True)
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = None
        self.trades = []  # To store trade positions

    # Fetch OHLCV data
    def fetch_data(self):
        try:
            bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=100)
            self.df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            print("Market Data fetched successfully.")
            print(self.df.tail())  # Display the last few rows of the DataFrame
        except Exception as e:
            print("Error fetching market data:", e)

    # Identify trend direction (use multiple timeframes)
    def identify_trend(self):
        self.df['EMA_200'] = ta.trend.ema_indicator(self.df['close'], window=200)  # Long-term trend
        self.df['EMA_50'] = ta.trend.ema_indicator(self.df['close'], window=50)    # Medium-term trend
        self.df['EMA_20'] = ta.trend.ema_indicator(self.df['close'], window=20)    # Short-term trend

        # Bullish or bearish trend based on EMA crossovers
        self.df['trend'] = np.where(self.df['EMA_50'] > self.df['EMA_200'], 'bullish', 'bearish')
        self.df['short_trend'] = np.where(self.df['EMA_20'] > self.df['EMA_50'], 'bullish', 'bearish')

    # Support and resistance levels (Supply and Demand Zones)
    def calculate_support_resistance(self):
        high_price = self.df['high'].rolling(window=14).max()
        low_price = self.df['low'].rolling(window=14).min()

        self.df['resistance'] = high_price
        self.df['support'] = low_price

    # Candlestick patterns
    def detect_candlestick_patterns(self):
        # Example: Detecting bullish engulfing pattern
        self.df['bullish_engulfing'] = np.where(
            (self.df['close'].shift(1) < self.df['open'].shift(1)) &
            (self.df['close'] > self.df['open']) &
            (self.df['close'] > self.df['high'].shift(1)) &
            (self.df['open'] < self.df['low'].shift(1)),
            True, False
        )

    # Fair value gaps (FVG) and liquidity sweeps
    def detect_fvg_liquidity_sweeps(self):
        # Identify liquidity sweeps based on relative highs and lows
        self.df['liquidity_sweep_high'] = np.where(self.df['high'] > self.df['high'].shift(1), True, False)
        self.df['liquidity_sweep_low'] = np.where(self.df['low'] < self.df['low'].shift(1), True, False)

        # Mark fair value gaps (FVG)
        self.df['fvg_up'] = np.where(
            (self.df['low'] > self.df['high'].shift(2)) &
            (self.df['close'] < self.df['open']), True, False
        )
        self.df['fvg_down'] = np.where(
            (self.df['high'] < self.df['low'].shift(2)) &
            (self.df['close'] > self.df['open']), True, False
        )

    # Execute trade signals based on confluence of signals
    def generate_trade_signal(self):
        self.df['buy_signal'] = (
            (self.df['trend'] == 'bullish') &
            (self.df['short_trend'] == 'bullish') &
            (self.df['bullish_engulfing']) &
            (self.df['liquidity_sweep_low']) &
            (self.df['support'])
        )

        self.df['sell_signal'] = (
            (self.df['trend'] == 'bearish') &
            (self.df['short_trend'] == 'bearish') &
            (self.df['liquidity_sweep_high']) &
            (self.df['fvg_down']) &
            (self.df['resistance'])
        )

    # Define risk management with stop loss and take profit
    def risk_management(self, entry_price, signal_type):
        if signal_type == 'buy':
            # Set stop loss below recent support
            stop_loss = self.df['low'].rolling(window=14).min().iloc[-1]
            take_profit = entry_price + 2 * (entry_price - stop_loss)  # Risk/reward ratio 1:2
        elif signal_type == 'sell':
            # Set stop loss above recent resistance
            stop_loss = self.df['high'].rolling(window=14).max().iloc[-1]
            take_profit = entry_price - 2 * (stop_loss - entry_price)  # Risk/reward ratio 1:2
        
        return stop_loss, take_profit

    # Execute trade (for live trading)
    def execute_trade(self, signal_type):
        entry_price = self.df['close'].iloc[-1]
        stop_loss, take_profit = self.risk_management(entry_price, signal_type)
        
        if signal_type == 'buy':
            trade_info = {
                'type': 'buy',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': self.df['timestamp'].iloc[-1]
            }
            self.trades.append(trade_info)
            print(f"Signal: BUY at {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
            # Place a market order to BUY on testnet
            # Uncomment the following line to enable actual trading on testnet
            # self.exchange.create_market_buy_order(self.symbol, amount)
        elif signal_type == 'sell':
            trade_info = {
                'type': 'sell',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': self.df['timestamp'].iloc[-1]
            }
            self.trades.append(trade_info)
            print(f"Signal: SELL at {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
            # Place a market order to SELL on testnet
            # Uncomment the following line to enable actual trading on testnet
            # self.exchange.create_market_sell_order(self.symbol, amount)

    # Backtesting method to simulate trading over historical data
    def backtest(self):
        self.trades.clear()  # Clear previous trades
        for index, row in self.df.iterrows():
            if row['buy_signal']:
                entry_price = row['close']
                stop_loss, take_profit = self.risk_management(entry_price, 'buy')
                self.trades.append({
                    'type': 'buy',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': row['timestamp']
                })
            elif row['sell_signal']:
                entry_price = row['close']
                stop_loss, take_profit = self.risk_management(entry_price, 'sell')
                self.trades.append({
                    'type': 'sell',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': row['timestamp']
                })

    # Print out trades executed during backtesting
    def display_trades(self):
        if not self.trades:
            print("No trades executed.")
            return
        
        for trade in self.trades:
            print(f"{trade['timestamp']} - {trade['type'].upper()} at {trade['entry_price']}, "
                  f"Stop Loss: {trade['stop_loss']}, Take Profit: {trade['take_profit']}")

    # Main function to run the bot
    def run(self):
        self.fetch_data()
        self.identify_trend()
        self.calculate_support_resistance()
        self.detect_candlestick_patterns()
        self.detect_fvg_liquidity_sweeps()
        self.generate_trade_signal()
        
        # Check for trade signals
        if self.df['buy_signal'].iloc[-1]:
            self.execute_trade('buy')
        elif self.df['sell_signal'].iloc[-1]:
            self.execute_trade('sell')

        # Perform backtesting and display results
        self.backtest()
        self.display_trades()


# Example usage
api_key = 'iHHTG1vv9GtlbhdWGJ81AsoGw7atjqJcLUmPq0VJQxUygnRyic0ZOiqMt0iucVhm'
secret_key = 'wa5HuO1gDg1ZzpKMNvV7eDn2A1U1wO5a5gdpdlOmPz9f7untJsZA7D1FLy7q5cYo'
bot = TradingBot(api_key, secret_key, symbol='XAU/USDT', timeframe='1h')
bot.run()
