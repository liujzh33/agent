import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
import os
import glob
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class MultiTimeframeTechnicalIndicators:
    
    @staticmethod
    def ema(close, period):
        return close.ewm(span=period).mean()
    
    @staticmethod
    def sma(close, period):
        return close.rolling(window=period).mean()
    
    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high, low, close, period=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def bollinger_bands(close, period=20, std_dev=2):
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def volume_sma(volume, period=20):
        return volume.rolling(window=period).mean()

class MultiTimeframeDataProcessor:
    
    def __init__(self, base_data_path):
        self.base_data_path = base_data_path
        self.timeframes = {
            '1h': '1h',
            '15m': '15m', 
            '5m': '5m',
            '3m': '3m'
        }
        
    def load_all_timeframes(self):
        all_data = {}
        
        for tf_name, tf_path in self.timeframes.items():
            file_path = f"{self.base_data_path}/futures/klines/data/futures/um/monthly/klines/BTCUSDT/{tf_path}/2025-06-01_2025-07-31/BTCUSDT-{tf_path}-2025-06.csv"
            
            print(f"Loading {tf_name} data: {file_path}")
            data = self.load_single_timeframe(file_path)
            if data is not None:
                all_data[tf_name] = data
                print(f"{tf_name} data loaded: {len(data)} records")
            else:
                print(f"Warning: {tf_name} data loading failed")
                
        return all_data
    
    def load_single_timeframe(self, file_path):
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            print(f"Loading data file: {file_path}")
            data = pd.read_csv(file_path)
            
            print(f"Column names: {list(data.columns)}")
            print(f"Original data shape: {data.shape}")
            print("Data sample:")
            print(data.head(3))
            
            required_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return None
            
            print(f"Timestamp sample: {data['open_time'].head(3).tolist()}")
            data['timestamp'] = pd.to_datetime(data['open_time'], unit='ms')
            
            column_mapping = {
                'open_time': 'open_time_original',
                'taker_buy_volume': 'taker_buy_base',
                'taker_buy_quote_volume': 'taker_buy_quote'
            }
            data = data.rename(columns=column_mapping)
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            print(f"Original data rows: {len(data)}")
            data = data.dropna(subset=numeric_cols + ['timestamp']).reset_index(drop=True)
            print(f"Cleaned data rows: {len(data)}")
            
            if len(data) == 0:
                print(f"Warning: Data is empty after cleaning")
                return None
            
            print(f"Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
            
            print("Starting technical indicator calculation...")
            data = self.calculate_indicators(data)
            print(f"Technical indicators calculation completed, final data: {len(data)} rows")
            
            return data
            
        except Exception as e:
            print(f"Data loading failed {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_indicators(self, data):
        try:
            print("Calculating technical indicators...")
            original_length = len(data)
            
            data['ema_12'] = MultiTimeframeTechnicalIndicators.ema(data['close'], 12)
            data['ema_26'] = MultiTimeframeTechnicalIndicators.ema(data['close'], 26)
            data['ema_50'] = MultiTimeframeTechnicalIndicators.ema(data['close'], 50)
            data['sma_20'] = MultiTimeframeTechnicalIndicators.sma(data['close'], 20)
            data['sma_50'] = MultiTimeframeTechnicalIndicators.sma(data['close'], 50)
            
            data['rsi'] = MultiTimeframeTechnicalIndicators.rsi(data['close'], 14)
            
            data['macd'], data['macd_signal'], data['macd_hist'] = MultiTimeframeTechnicalIndicators.macd(data['close'])
            
            data['atr'] = MultiTimeframeTechnicalIndicators.atr(data['high'], data['low'], data['close'], 14)
            
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = MultiTimeframeTechnicalIndicators.bollinger_bands(data['close'])
            
            if 'volume' in data.columns:
                data['volume_sma'] = MultiTimeframeTechnicalIndicators.volume_sma(data['volume'], 20)
                data['volume_ratio'] = data['volume'] / data['volume_sma'].replace(0, 1)
            else:
                data['volume_sma'] = 1
                data['volume_ratio'] = 1
            
            bb_range = data['bb_upper'] - data['bb_lower']
            bb_range = bb_range.replace(0, 1)
            data['price_position'] = (data['close'] - data['bb_lower']) / bb_range
            data['price_position'] = data['price_position'].clip(0, 1)
            
            data['trend_strength'] = (data['ema_12'] - data['ema_26']) / data['close']
            
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna().reset_index(drop=True)
            
            print(f"Technical indicators calculation completed: {original_length} -> {len(data)} rows")
            
            key_indicators = ['ema_12', 'ema_26', 'rsi', 'macd', 'atr', 'price_position']
            for indicator in key_indicators:
                if indicator in data.columns:
                    valid_count = data[indicator].notna().sum()
                    print(f"{indicator}: {valid_count}/{len(data)} valid values")
            
            return data
            
        except Exception as e:
            print(f"Technical indicator calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return data

class EnhancedTradingSignalGenerator:
    """增强版信号生成器 - 采用打分制，降低触发门槛"""
    
    def __init__(self):
        self.signal_threshold = 2  # 从3降到2
        self.cooldown_period = 3  # 缩短冷却期
        self.last_trade_step = -999
        
    def calculate_signal_scores(self, data_1h, data_15m, data_5m, current_idx, current_step):
        
        # 冷却期检查放宽
        if current_step - self.last_trade_step < self.cooldown_period:
            return {
                'long_score': 0,
                'short_score': 0, 
                'trend_direction': 'cooldown',
                'signal_strength': 0
            }
        
        scores = {
            'long_score': 0,
            'short_score': 0,
            'trend_direction': 'neutral',
            'signal_strength': 0
        }
        
        # 1小时趋势评分（权重更高）
        h1_trend_score = self.get_enhanced_1h_trend_score(data_1h, current_idx)
        scores['trend_direction'] = h1_trend_score['direction']
        scores['long_score'] += h1_trend_score['long_strength']
        scores['short_score'] += h1_trend_score['short_strength']
        
        # 15分钟信号评分
        m15_scores = self.get_enhanced_15m_signals(data_15m, current_idx)
        scores['long_score'] += m15_scores['long_signals']
        scores['short_score'] += m15_scores['short_signals']
        
        # 5分钟信号评分
        m5_scores = self.get_enhanced_5m_signals(data_5m, current_idx)
        scores['long_score'] += m5_scores['long_signals']
        scores['short_score'] += m5_scores['short_signals']
        
        # 计算总信号强度
        scores['signal_strength'] = max(scores['long_score'], scores['short_score'])
        
        return scores
    
    def get_enhanced_1h_trend_score(self, data_1h, current_idx):
        if data_1h is None or current_idx >= len(data_1h):
            return {'direction': 'neutral', 'long_strength': 0, 'short_strength': 0}
            
        try:
            current = data_1h.iloc[current_idx]
            prev = data_1h.iloc[max(0, current_idx-1)]
            
            long_strength = 0
            short_strength = 0
            
            # EMA趋势强度评分
            if current['ema_12'] > current['ema_26']:
                ema_strength = abs(current['ema_12'] - current['ema_26']) / current['ema_26']
                if ema_strength > 0.003:
                    long_strength += 2
                elif ema_strength > 0.001:
                    long_strength += 1
            else:
                ema_strength = abs(current['ema_12'] - current['ema_26']) / current['ema_26']
                if ema_strength > 0.003:
                    short_strength += 2
                elif ema_strength > 0.001:
                    short_strength += 1
            
            # MACD动量评分
            if current['macd'] > current['macd_signal'] and current['macd_hist'] > prev['macd_hist']:
                long_strength += 1.5
            elif current['macd'] < current['macd_signal'] and current['macd_hist'] < prev['macd_hist']:
                short_strength += 1.5
            
            # RSI确认评分（更宽松的范围）
            if 40 < current['rsi'] < 70:
                if current['rsi'] > 55:
                    long_strength += 0.5
                elif current['rsi'] < 45:
                    short_strength += 0.5
            
            # 价格相对SMA位置
            if current['close'] > current['sma_50']:
                long_strength += 1
            else:
                short_strength += 1
            
            # 确定主趋势方向
            if long_strength >= short_strength + 1:
                direction = 'bullish' if long_strength >= 3 else 'weak_bullish'
            elif short_strength >= long_strength + 1:
                direction = 'bearish' if short_strength >= 3 else 'weak_bearish'
            else:
                direction = 'neutral'
                
            return {
                'direction': direction, 
                'long_strength': long_strength, 
                'short_strength': short_strength
            }
                
        except Exception as e:
            return {'direction': 'neutral', 'long_strength': 0, 'short_strength': 0}
    
    def get_enhanced_15m_signals(self, data_15m, current_idx):
        signals = {'long_signals': 0, 'short_signals': 0}
        
        if data_15m is None or current_idx >= len(data_15m):
            return signals
            
        try:
            current = data_15m.iloc[current_idx]
            prev = data_15m.iloc[max(0, current_idx-1)]
            
            # RSI反转信号（范围放宽）
            if 25 < current['rsi'] < 45 and current['rsi'] > prev['rsi']:
                signals['long_signals'] += 1.5
            elif 55 < current['rsi'] < 75 and current['rsi'] < prev['rsi']:
                signals['short_signals'] += 1.5
                
            # MACD动量信号
            if (current['macd'] > current['macd_signal'] and 
                current['macd_hist'] > 0 and 
                current['macd_hist'] > prev['macd_hist']):
                signals['long_signals'] += 1
            elif (current['macd'] < current['macd_signal'] and 
                  current['macd_hist'] < 0 and 
                  current['macd_hist'] < prev['macd_hist']):
                signals['short_signals'] += 1
                
            # EMA排列信号
            if (current['close'] > current['ema_12'] > current['ema_26'] and
                current['ema_12'] > prev['ema_12']):
                signals['long_signals'] += 1
            elif (current['close'] < current['ema_12'] < current['ema_26'] and
                  current['ema_12'] < prev['ema_12']):
                signals['short_signals'] += 1
                
            # 成交量确认（门槛降低）
            if current['volume_ratio'] > 1.2:  # 从1.5降到1.2
                price_change = (current['close'] - current['open']) / current['open']
                if price_change > 0.001:  # 从0.002降到0.001
                    signals['long_signals'] += 0.5
                elif price_change < -0.001:
                    signals['short_signals'] += 0.5
                    
        except:
            pass
                    
        return signals
    
    def get_enhanced_5m_signals(self, data_5m, current_idx):
        signals = {'long_signals': 0, 'short_signals': 0}
        
        if data_5m is None or current_idx >= len(data_5m):
            return signals
            
        try:
            current = data_5m.iloc[current_idx]
            prev = data_5m.iloc[max(0, current_idx-1)]
            
            # 布林带位置信号（更敏感）
            if current['price_position'] < 0.25 and current['close'] > prev['close']:  # 从0.15提高到0.25
                signals['long_signals'] += 1
            elif current['price_position'] > 0.75 and current['close'] < prev['close']:  # 从0.85降到0.75
                signals['short_signals'] += 1
                
            # 短期趋势确认
            if (current['close'] > current['ema_12'] and 
                prev['close'] > prev['ema_12'] and
                current['ema_12'] > prev['ema_12']):
                signals['long_signals'] += 0.5
            elif (current['close'] < current['ema_12'] and 
                  prev['close'] < prev['ema_12'] and
                  current['ema_12'] < prev['ema_12']):
                signals['short_signals'] += 0.5
                
        except:
            pass
                
        return signals
    
    def update_last_trade_step(self, step):
        self.last_trade_step = step

class EnhancedMultiTimeframeBTCEnv(gym.Env):
    """增强版多时间框架环境 - 更灵活的仓位管理"""
    
    def __init__(self, base_data_path, initial_balance=10000, lookback_window=50, verbose=False):
        super(EnhancedMultiTimeframeBTCEnv, self).__init__()
        
        print("=== Enhanced Multi-Timeframe BTC Futures Trading Environment ===")
        print("更灵活仓位 + 增强收益 + 风控平衡")
        
        self.base_data_path = base_data_path
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.verbose = verbose
        
        self.data_processor = MultiTimeframeDataProcessor(base_data_path)
        self.all_data = self.data_processor.load_all_timeframes()
        
        loaded_timeframes = list(self.all_data.keys())
        print(f"Successfully loaded timeframes: {loaded_timeframes}")
        
        if not self.all_data:
            raise ValueError("No timeframe data loaded successfully! Please check data path and format.")
        
        if '15m' in self.all_data:
            self.main_data = self.all_data['15m']
            print("Using 15m as main data timeframe")
        elif '5m' in self.all_data:
            self.main_data = self.all_data['5m']
            print("15m data unavailable, using 5m as main data timeframe")
        elif '1h' in self.all_data:
            self.main_data = self.all_data['1h']
            print("15m and 5m data unavailable, using 1h as main data timeframe")
        else:
            first_available = list(self.all_data.keys())[0]
            self.main_data = self.all_data[first_available]
            print(f"Using {first_available} as main data timeframe")
        
        print(f"Main data contains {len(self.main_data)} records")
        self.current_step = lookback_window
        
        self.signal_generator = EnhancedTradingSignalGenerator()
        
        # 新的动作空间：7个动作，更灵活的仓位管理
        self.action_space = spaces.Discrete(7)
        
        self.action_mapping = {
            0: ('hold', 0),          # 持有
            1: ('buy', 0.20),        # 买入20%
            2: ('buy', 0.50),        # 买入50%
            3: ('buy', 0.80),        # 买入80%（激进）
            4: ('sell', 0.20),       # 减仓20%
            5: ('sell', 0.50),       # 减仓50%
            6: ('sell', 1.00)        # 全平
        }
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(25,), dtype=np.float32
        )
        
        # 风险参数调整
        self.max_position_ratio = 0.80  # 最大仓位提高到80%
        self.atr_stop_multiplier = 1.8  # ATR止损系数略微收紧
        self.min_trade_interval = 2     # 最小交易间隔缩短
        self.last_trade_step = -999
        
        # 新增参数
        self.high_quality_profit_threshold = 0.5  # 高质量止盈阈值（R倍数）
        self.drawdown_penalty_threshold = 0.012   # 1.2% ATR回撤惩罚阈值
        self.max_acceptable_drawdown = 0.05       # 5%最大可接受回撤
        self.oscillation_position_threshold = 0.6 # 震荡期高仓位惩罚阈值
        
        self.reset_statistics()
        
        obs, _ = self.reset()
        print("=== Enhanced multi-timeframe trading environment initialization completed ===")
    
    def reset_statistics(self):
        self.trades_history = []
        self.positions_history = []
        self.returns_history = []
        self.pnl_history = []
        self.max_drawdown = 0
        self.peak_net_worth = 0
        self.long_trades = 0
        self.short_trades = 0
        self.holding_periods = []
        self.current_holding_start = None
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.trades_count = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        
        self.entry_price = 0
        self.stop_loss_price = 0
        self.position_size = 0
        self.last_trade_step = -999
        self.consecutive_trades = 0  # 连续交易计数
        self.last_action_type = 'hold'
        
        self.signal_generator.last_trade_step = -999
        
        self.reset_statistics()
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self):
        current_time = self.main_data.iloc[self.current_step]['timestamp']
        
        h1_features = self._get_1h_features(current_time)
        m15_features = self._get_15m_features()
        m5_features = self._get_5m_features(current_time)
        account_features = self._get_account_features()
        risk_features = self._get_risk_features()
        
        observation = np.concatenate([
            h1_features, m15_features, m5_features, 
            account_features, risk_features
        ])
        
        observation = np.nan_to_num(observation, nan=0.0, posinf=3.0, neginf=-3.0)
        observation = np.clip(observation, -5, 5)
        
        return observation.astype(np.float32)
    
    def _get_1h_features(self, current_time):
        if '1h' not in self.all_data:
            return np.zeros(5)
            
        data_1h = self.all_data['1h']
        idx = self._find_closest_index(data_1h, current_time)
        
        if idx >= len(data_1h):
            return np.zeros(5)
            
        current = data_1h.iloc[idx]
        
        features = [
            1.0 if current['ema_12'] > current['ema_26'] else -1.0,
            np.tanh(current['macd'] / current['close'] * 1000),
            (current['rsi'] - 50) / 50,
            (current['close'] - current['sma_50']) / current['sma_50'],
            np.log(current['volume_ratio']) if current['volume_ratio'] > 0 else 0
        ]
        
        return np.array(features)
    
    def _get_15m_features(self):
        if self.current_step >= len(self.main_data):
            return np.zeros(10)
            
        current = self.main_data.iloc[self.current_step]
        
        features = [
            current['trend_strength'],
            (current['rsi'] - 50) / 50,
            np.tanh(current['macd'] / current['close'] * 1000),
            np.tanh(current['macd_hist'] / current['close'] * 1000),
            (current['close'] - current['ema_12']) / current['ema_12'],
            (current['ema_12'] - current['ema_26']) / current['ema_26'],
            current['price_position'],
            np.log(current['volume_ratio']) if current['volume_ratio'] > 0 else 0,
            current['atr'] / current['close'],
            (current['close'] - current['open']) / current['open']
        ]
        
        return np.array(features)
    
    def _get_5m_features(self, current_time):
        if '5m' not in self.all_data:
            return np.zeros(5)
            
        data_5m = self.all_data['5m']
        idx = self._find_closest_index(data_5m, current_time)
        
        if idx >= len(data_5m):
            return np.zeros(5)
            
        current = data_5m.iloc[idx]
        
        features = [
            (current['close'] - current['ema_12']) / current['ema_12'],
            current['price_position'],
            (current['rsi'] - 50) / 50,
            (current['close'] - current['open']) / current['open'],
            np.log(current['volume_ratio']) if current['volume_ratio'] > 0 else 0
        ]
        
        return np.array(features)
    
    def _get_account_features(self):
        current_price = self.main_data.iloc[self.current_step]['close']
        position_value = self.shares_held * current_price
        
        features = [
            position_value / self.net_worth if self.net_worth > 0 else 0,
            self.balance / self.net_worth if self.net_worth > 0 else 0,
            (self.net_worth - self.initial_balance) / self.initial_balance
        ]
        
        return np.array(features)
    
    def _get_risk_features(self):
        current_price = self.main_data.iloc[self.current_step]['close']
        
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        
        stop_distance = 0
        if self.shares_held > 0 and self.stop_loss_price > 0:
            stop_distance = (current_price - self.stop_loss_price) / current_price
        
        features = [drawdown, stop_distance]
        
        return np.array(features)
    
    def _find_closest_index(self, data, target_time):
        time_diffs = np.abs(data['timestamp'] - target_time)
        return time_diffs.argmin()
    
    def step(self, action):
        current_price = self.main_data.iloc[self.current_step]['close']
        
        try:
            current_atr = self.main_data.iloc[self.current_step]['atr']
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.01
        except:
            current_atr = current_price * 0.01
            
        prev_net_worth = self.net_worth
        prev_position_value = self.shares_held * current_price
        
        action_type, ratio = self.action_mapping[action]
        trade_executed = False
        
        # 执行交易
        if action_type == 'buy':
            trade_executed = self._execute_buy(current_price, current_atr, ratio)
            if trade_executed:
                self.long_trades += 1
        elif action_type == 'sell':
            trade_executed = self._execute_sell(current_price, ratio)
        
        # 检查止损
        self._check_stop_loss(current_price)
        
        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        
        # 计算增强奖励
        reward = self._calculate_enhanced_reward(
            prev_net_worth, prev_position_value, action, trade_executed, current_price, current_atr
        )
        
        # 更新统计信息
        self._update_enhanced_statistics(current_price, action_type)
        
        self.current_step += 1
        done = self.current_step >= len(self.main_data) - 1
        
        info = self._get_enhanced_info(current_price, trade_executed, action_type, ratio)
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_buy(self, current_price, current_atr, ratio):
        
        # 冷却期检查
        if self.current_step - self.last_trade_step < self.min_trade_interval:
            return False
        
        # 计算当前仓位比例
        current_position_value = self.shares_held * current_price
        current_position_ratio = current_position_value / self.net_worth if self.net_worth > 0 else 0
        
        # 检查是否超过最大仓位限制
        if current_position_ratio >= self.max_position_ratio:
            if self.verbose:
                print(f"Position limit reached {current_position_ratio:.1%}, skipping buy")
            return False
        
        # 计算可以增加的仓位
        max_additional_ratio = self.max_position_ratio - current_position_ratio
        actual_ratio = min(ratio, max_additional_ratio)
        
        if actual_ratio < 0.05:  # 最小交易门槛
            return False
            
        available_balance = self.balance * actual_ratio
        
        if available_balance < 100:  # 最小交易金额
            return False
            
        shares_to_buy = available_balance / current_price
        
        if shares_to_buy > 0:
            self.balance -= available_balance
            self.shares_held += shares_to_buy
            self.trades_count += 1
            self.last_trade_step = self.current_step
            self.signal_generator.update_last_trade_step(self.current_step)
            
            # 更新入场价格（加权平均）
            if self.entry_price > 0 and self.position_size > 0:
                total_value = self.position_size * self.entry_price + available_balance
                total_shares = self.position_size / self.entry_price + shares_to_buy
                self.entry_price = total_value / (total_shares * current_price) * current_price
            else:
                self.entry_price = current_price
            
            self.position_size += available_balance
            
            # 设置动态止损
            if pd.notna(current_atr) and current_atr > 0:
                self.stop_loss_price = current_price - (current_atr * self.atr_stop_multiplier)
                if self.stop_loss_price <= 0:
                    self.stop_loss_price = current_price * 0.95
            else:
                self.stop_loss_price = current_price * 0.95
                
            # 记录持仓开始时间
            if self.current_holding_start is None:
                self.current_holding_start = self.current_step
                
            if self.verbose:
                print(f"Buy: {shares_to_buy:.4f} @ ${current_price:.2f}, Stop loss: ${self.stop_loss_price:.2f}")
            
            return True
            
        return False
    
    def _execute_sell(self, current_price, ratio):
        if self.shares_held <= 0:
            return False
            
        shares_to_sell = self.shares_held * ratio
        sell_amount = shares_to_sell * current_price
        
        if shares_to_sell > 0:
            # 计算该部分的盈亏
            if self.entry_price > 0:
                pnl = (current_price - self.entry_price) * shares_to_sell
                self.pnl_history.append(pnl)
                
                if pnl > 0:
                    self.profitable_trades += 1
                    self.total_profit += pnl
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(pnl)
            
            self.balance += sell_amount
            self.shares_held -= shares_to_sell
            self.position_size -= (self.position_size * ratio) if self.position_size > 0 else 0
            self.trades_count += 1
            self.last_trade_step = self.current_step
            self.signal_generator.update_last_trade_step(self.current_step)
            
            if self.verbose:
                print(f"Sell: {shares_to_sell:.4f} @ ${current_price:.2f}, Remaining position: {self.shares_held:.4f}")
            
            # 如果完全平仓，重置相关参数
            if self.shares_held <= 0.001:  # 接近0
                self.shares_held = 0
                self.stop_loss_price = 0
                self.entry_price = 0
                self.position_size = 0
                
                # 记录持仓时间
                if self.current_holding_start is not None:
                    holding_period = self.current_step - self.current_holding_start
                    self.holding_periods.append(holding_period)
                    self.current_holding_start = None
            
            return True
            
        return False
    
    def _check_stop_loss(self, current_price):
        if self.shares_held > 0 and self.stop_loss_price > 0:
            if current_price <= self.stop_loss_price:
                if self.verbose:
                    print(f"ATR stop loss triggered: Price {current_price:.2f} <= Stop {self.stop_loss_price:.2f}")
                self._execute_sell(current_price, 1.0)
            else:
                # 动态追踪止损
                try:
                    current_atr = self.main_data.iloc[self.current_step]['atr']
                    if pd.notna(current_atr) and current_atr > 0:
                        new_stop_loss = current_price - (current_atr * self.atr_stop_multiplier)
                        if new_stop_loss > self.stop_loss_price:
                            old_stop = self.stop_loss_price
                            self.stop_loss_price = new_stop_loss
                            if self.verbose and abs(new_stop_loss - old_stop) > 50:
                                print(f"Trailing stop updated: ${old_stop:.2f} -> ${new_stop_loss:.2f}")
                except:
                    pass
    
    def _update_enhanced_statistics(self, current_price, action_type):
        # 更新净值峰值
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        # 更新最大回撤
        if self.max_net_worth > 0:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # 记录收益率历史
        self.returns_history.append(self.net_worth / self.initial_balance - 1)
        
        # 记录仓位历史
        position_ratio = (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        self.positions_history.append(position_ratio)
        
        # 连续交易计数
        if action_type != 'hold':
            if self.last_action_type != 'hold' and self.last_action_type != action_type:
                self.consecutive_trades += 1
            else:
                self.consecutive_trades = 0
        else:
            self.consecutive_trades = 0
            
        self.last_action_type = action_type
    
    def _calculate_enhanced_reward(self, prev_net_worth, prev_position_value, action, trade_executed, current_price, current_atr):
        """增强版奖励函数 - 平衡收益、稳定性和风险控制"""
        
        # 1. 基础盈亏奖励（核心）
        net_worth_change = self.net_worth - prev_net_worth
        base_reward = net_worth_change / self.initial_balance * 10  # 放大基础奖励
        
        # 2. 信号质量奖励
        signal_reward = 0
        if trade_executed:
            current_time = self.main_data.iloc[self.current_step-1]['timestamp']
            data_1h = self.all_data.get('1h')
            data_5m = self.all_data.get('5m')
            
            if data_1h is not None and data_5m is not None:
                h1_idx = self._find_closest_index(data_1h, current_time)
                m5_idx = self._find_closest_index(data_5m, current_time)
                
                scores = self.signal_generator.calculate_signal_scores(
                    data_1h, self.main_data, data_5m, 
                    min(h1_idx, self.current_step-1),
                    self.current_step
                )
                
                action_type, ratio = self.action_mapping[action]
                
                # 按信号强度给予奖励
                if action_type == 'buy' and scores['long_score'] >= 2:
                    signal_reward = 0.008 * scores['long_score']  # 提高信号奖励
                elif action_type == 'sell' and scores['short_score'] >= 2:
                    signal_reward = 0.008 * scores['short_score']
        
        # 3. 趋势持仓奖励
        trend_reward = 0
        if self.shares_held > 0:
            try:
                current_time = self.main_data.iloc[self.current_step-1]['timestamp']
                data_1h = self.all_data.get('1h')
                if data_1h is not None:
                    h1_idx = self._find_closest_index(data_1h, current_time)
                    h1_trend = self.signal_generator.get_enhanced_1h_trend_score(data_1h, h1_idx)
                    
                    # 顺势持仓奖励
                    if h1_trend['direction'] in ['bullish', 'weak_bullish']:
                        trend_reward = 0.005
                    elif h1_trend['direction'] in ['bearish', 'weak_bearish']:
                        trend_reward = -0.003  # 逆势持仓轻微惩罚
            except:
                pass
        
        # 4. 高质量止盈奖励
        profit_taking_reward = 0
        if trade_executed and action in [4, 5, 6]:  # 卖出动作
            if self.entry_price > 0 and current_price > self.entry_price:
                profit_ratio = (current_price - self.entry_price) / self.entry_price
                if profit_ratio > self.high_quality_profit_threshold * current_atr / current_price:
                    profit_taking_reward = 0.015  # 高质量止盈奖励
        
        # 5. 仓位管理奖励
        position_reward = 0
        position_ratio = (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        cash_ratio = self.balance / self.net_worth if self.net_worth > 0 else 0
        
        # 合理仓位奖励
        if 0.2 <= position_ratio <= 0.6:
            position_reward += 0.004
        elif 0.6 < position_ratio <= 0.8:
            position_reward += 0.002
        
        # 现金储备奖励
        if 0.2 <= cash_ratio <= 0.6:
            position_reward += 0.003
        
        # 6. 风险惩罚
        risk_penalty = 0
        
        # 交易成本惩罚（降低）
        if trade_executed:
            risk_penalty += 0.0005  # 降低交易成本惩罚
        
        # 过度仓位惩罚
        if position_ratio > 0.8:
            risk_penalty += (position_ratio - 0.8) * 0.02
        
        # 频繁交易惩罚
        if self.consecutive_trades > 2:
            risk_penalty += 0.003
        
        # ATR回撤惩罚
        if self.shares_held > 0 and self.entry_price > 0:
            drawdown_from_entry = (self.entry_price - current_price) / self.entry_price
            atr_ratio = current_atr / current_price
            if drawdown_from_entry > self.drawdown_penalty_threshold * atr_ratio:
                risk_penalty += 0.008
        
        # 整体回撤惩罚
        if self.max_drawdown > self.max_acceptable_drawdown:
            risk_penalty += (self.max_drawdown - self.max_acceptable_drawdown) * 0.5
        
        # 震荡期高仓位惩罚
        try:
            current_volatility = current_atr / current_price
            if current_volatility > 0.02 and position_ratio > self.oscillation_position_threshold:
                risk_penalty += 0.005
        except:
            pass
        
        # 7. 逆势交易惩罚
        trend_penalty = 0
        if trade_executed:
            try:
                current_time = self.main_data.iloc[self.current_step-1]['timestamp']
                data_1h = self.all_data.get('1h')
                if data_1h is not None:
                    h1_idx = self._find_closest_index(data_1h, current_time)
                    h1_trend = self.signal_generator.get_enhanced_1h_trend_score(data_1h, h1_idx)
                    action_type, _ = self.action_mapping[action]
                    
                    if action_type == 'buy' and h1_trend['direction'] in ['bearish', 'weak_bearish']:
                        trend_penalty = 0.005
                    elif action_type == 'sell' and h1_trend['direction'] in ['bullish', 'weak_bullish']:
                        trend_penalty = 0.003
            except:
                pass
        
        # 总奖励计算
        total_reward = (base_reward + signal_reward + trend_reward + 
                       profit_taking_reward + position_reward - 
                       risk_penalty - trend_penalty)
        
        # 限制奖励范围
        total_reward = np.clip(total_reward, -0.1, 0.1)
        
        return total_reward
    
    def _get_enhanced_info(self, current_price, trade_executed, action_type, ratio):
        total_trades = self.profitable_trades + self.losing_trades
        win_rate = self.profitable_trades / max(total_trades, 1) * 100
        
        profit_loss_ratio = 0
        if self.total_loss > 0:
            profit_loss_ratio = self.total_profit / self.total_loss
        
        # 计算年化收益率（假设250个交易日）
        total_return = (self.net_worth / self.initial_balance - 1)
        periods_per_year = 250 * 24 * 4  # 15分钟周期
        current_periods = len(self.returns_history)
        annualized_return = 0
        if current_periods > 0:
            annualized_return = ((1 + total_return) ** (periods_per_year / current_periods) - 1) * 100
        
        # 计算夏普比率
        sharpe_ratio = 0
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            if np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(periods_per_year)
        
        # 计算平均持仓时间
        avg_holding_time = np.mean(self.holding_periods) if self.holding_periods else 0
        
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'action_type': action_type,
            'ratio': ratio,
            'trades_count': self.trades_count,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'position_ratio': (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0,
            'stop_loss_price': self.stop_loss_price,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'avg_holding_time': avg_holding_time,
            'consecutive_trades': self.consecutive_trades
        }
    
    def get_enhanced_performance_summary(self):
        """增强版性能统计"""
        total_trades = self.profitable_trades + self.losing_trades
        win_rate = self.profitable_trades / max(total_trades, 1) * 100
        
        profit_loss_ratio = 0
        if self.total_loss > 0:
            profit_loss_ratio = self.total_profit / self.total_loss
        
        # 年化收益率
        total_return = (self.net_worth / self.initial_balance - 1)
        periods_per_year = 250 * 24 * 4  # 15分钟周期
        current_periods = len(self.returns_history)
        annualized_return = 0
        if current_periods > 0:
            annualized_return = ((1 + total_return) ** (periods_per_year / current_periods) - 1) * 100
        
        # 夏普比率
        sharpe_ratio = 0
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            if np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(periods_per_year)
        
        # 平均持仓时间
        avg_holding_time = np.mean(self.holding_periods) if self.holding_periods else 0
        
        # 多空分布
        total_directional_trades = self.long_trades + self.short_trades
        long_ratio = self.long_trades / max(total_directional_trades, 1) * 100
        short_ratio = self.short_trades / max(total_directional_trades, 1) * 100
        
        return {
            'total_return': total_return * 100,
            'annualized_return': annualized_return,
            'total_trades': self.trades_count,
            'profitable_trades': self.profitable_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'avg_holding_time': avg_holding_time,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'final_net_worth': self.net_worth
        }

def train_enhanced_multi_timeframe_agent(base_data_path, total_timesteps=150000, save_path="enhanced_multi_timeframe_btc_agent"):
    """训练增强版多时间框架交易智能体"""
    print("=== Creating Enhanced Multi-Timeframe BTC Futures Trading Environment ===")
    
    try:
        env = EnhancedMultiTimeframeBTCEnv(base_data_path, verbose=False)
        env = DummyVecEnv([lambda: env])
        
        print("=== Initializing Enhanced PPO Model ===")
        
        import torch.nn as nn
        
        # 优化超参数
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,      # 提高学习率
            n_steps=2048,            # 减少步数，增加更新频率
            batch_size=64,           # 减少批量大小
            n_epochs=10,             # 减少epochs
            gamma=0.99,
            ent_coef=0.05,           # 增加探索
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": [512, 256, 128, 64],
                "activation_fn": nn.ReLU
            },
            tensorboard_log="./enhanced_ppo_tensorboard/"
        )
        
        print(f"=== Starting Enhanced Training, Total Timesteps: {total_timesteps} ===")
        model.learn(total_timesteps=total_timesteps)
        
        print(f"=== Saving Enhanced Model: {save_path} ===")
        model.save(save_path)
        
        return model
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_enhanced_multi_timeframe_agent(model_path, base_data_path, episodes=3):
    """评估增强版多时间框架智能体"""
    print("=== Evaluating Enhanced Multi-Timeframe Model ===")
    
    try:
        env = EnhancedMultiTimeframeBTCEnv(base_data_path, verbose=False)
        model = PPO.load(model_path)
        
        all_results = []
        action_distributions = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            action_counts = {i: 0 for i in range(7)}
            
            print(f"\n=== Episode {episode+1} in progress... ===")
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action) if isinstance(action, np.ndarray) else action
                action_counts[action] += 1
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
            
            perf_summary = env.get_enhanced_performance_summary()
            all_results.append(perf_summary)
            
            # 动作分布
            action_names = ['Hold', 'Buy20%', 'Buy50%', 'Buy80%', 'Sell20%', 'Sell50%', 'SellAll']
            episode_distribution = {}
            for i, name in enumerate(action_names):
                percentage = action_counts[i] / step_count * 100
                episode_distribution[name] = percentage
            action_distributions.append(episode_distribution)
            
            print(f"\nEpisode {episode+1} Enhanced Results:")
            print(f"  Episode reward: {episode_reward:.4f}")
            print(f"  Total return: {perf_summary['total_return']:.2f}%")
            print(f"  Annualized return: {perf_summary['annualized_return']:.2f}%")
            print(f"  Max drawdown: {perf_summary['max_drawdown']:.2f}%")
            print(f"  Total trades: {perf_summary['total_trades']}")
            print(f"  Win rate: {perf_summary['win_rate']:.1f}%")
            print(f"  Profit/Loss ratio: {perf_summary['profit_loss_ratio']:.2f}")
            print(f"  Sharpe ratio: {perf_summary['sharpe_ratio']:.2f}")
            print(f"  Average holding time: {perf_summary['avg_holding_time']:.1f} periods")
            print(f"  Long trades: {perf_summary['long_trades']} ({perf_summary['long_ratio']:.1f}%)")
            print(f"  Short trades: {perf_summary['short_trades']} ({perf_summary['short_ratio']:.1f}%)")
            
            print(f"  Enhanced Action Distribution:")
            for name, percentage in episode_distribution.items():
                if percentage > 0:
                    print(f"    {name}: {action_counts[action_names.index(name)]} ({percentage:.1f}%)")
            
            print("-" * 80)
        
        if all_results:
            # 计算平均性能指标
            avg_return = np.mean([r['total_return'] for r in all_results])
            avg_annualized = np.mean([r['annualized_return'] for r in all_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in all_results])
            avg_win_rate = np.mean([r['win_rate'] for r in all_results])
            avg_trades = np.mean([r['total_trades'] for r in all_results])
            avg_pnl_ratio = np.mean([r['profit_loss_ratio'] for r in all_results])
            avg_holding_time = np.mean([r['avg_holding_time'] for r in all_results])
            avg_long_ratio = np.mean([r['long_ratio'] for r in all_results])
            
            # 计算平均动作分布
            avg_action_dist = {}
            action_names = ['Hold', 'Buy20%', 'Buy50%', 'Buy80%', 'Sell20%', 'Sell50%', 'SellAll']
            for name in action_names:
                avg_action_dist[name] = np.mean([dist[name] for dist in action_distributions])
            
            print(f"\n=== Enhanced Average Performance Metrics ===")
            print(f"Average total return: {avg_return:.2f}%")
            print(f"Average annualized return: {avg_annualized:.2f}%")
            print(f"Average max drawdown: {avg_drawdown:.2f}%")
            print(f"Average Sharpe ratio: {avg_sharpe:.2f}")
            print(f"Average win rate: {avg_win_rate:.1f}%")
            print(f"Average trades: {avg_trades:.0f}")
            # print(f"Average profit/loss ratio: {avg_pnl_ratio:.2f}")
            print(f"Average holding time: {avg_holding_time:.1f} periods")
            print(f"Average long trade ratio: {avg_long_ratio:.1f}%")
            
            print(f"\n=== Average Action Distribution ===")
            for name, percentage in avg_action_dist.items():
                if percentage > 0.1:
                    print(f"{name}: {percentage:.1f}%")
        
        return all_results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    base_data_path = "/data1/linbingqian/liujzh33/Agent/data/btc_2025_06_07"
    
    try:
        print("\n=== 1. Starting Enhanced Multi-Timeframe Training ===")
        model = train_enhanced_multi_timeframe_agent(
            base_data_path, 
            total_timesteps=150000,
            save_path="enhanced_multi_timeframe_btc_futures_agent"
        )
        
        if model is not None:
            print("Enhanced multi-timeframe training completed!")
            
            print("\n=== 2. Starting Enhanced Multi-Timeframe Evaluation ===")
            results = evaluate_enhanced_multi_timeframe_agent(
                "enhanced_multi_timeframe_btc_futures_agent", 
                base_data_path,
                episodes=3
            )
            
            if results:
                print("Enhanced evaluation completed!")
            else:
                print("Evaluation failed")
        else:
            print("Training failed")
            
        print(f"\nEnhanced model save location: enhanced_multi_timeframe_btc_futures_agent.zip")
        
    except Exception as e:
        print(f"System execution failed: {e}")
        import traceback
        traceback.print_exc()