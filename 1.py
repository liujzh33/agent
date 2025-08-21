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
    """增强版信号生成器 - 支持双向交易信号"""
    
    def __init__(self):
        self.signal_threshold = 3  # 恢复更严格的信号阈值
        self.short_signal_threshold = 4  # 做空需要更强的信号
        self.cooldown_period = 5  # 延长冷却期
        self.last_trade_step = -999
        
    def calculate_signal_scores(self, data_1h, data_15m, data_5m, current_idx, current_step):
        
        # 冷却期检查
        if current_step - self.last_trade_step < self.cooldown_period:
            return {
                'long_score': 0,
                'short_score': 0, 
                'trend_direction': 'cooldown',
                'signal_strength': 0,
                'safe_to_short': False
            }
        
        scores = {
            'long_score': 0,
            'short_score': 0,
            'trend_direction': 'neutral',
            'signal_strength': 0,
            'safe_to_short': False
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
        
        # 做空安全性检查
        scores['safe_to_short'] = self._check_short_safety(data_1h, data_15m, current_idx, scores['short_score'])
        
        return scores
    
    def _check_short_safety(self, data_1h, data_15m, current_idx, short_score):
        """检查做空的安全性 - 多重条件验证"""
        
        if short_score < self.short_signal_threshold:
            return False
            
        safety_checks = []
        
        try:
            # 1. 1小时趋势必须明确下跌
            if data_1h is not None and current_idx < len(data_1h):
                current_1h = data_1h.iloc[current_idx]
                prev_1h = data_1h.iloc[max(0, current_idx-1)]
                
                # EMA趋势检查
                ema_trend_down = (current_1h['ema_12'] < current_1h['ema_26'] and 
                                current_1h['ema_12'] < prev_1h['ema_12'])
                safety_checks.append(ema_trend_down)
                
                # MACD确认
                macd_bearish = (current_1h['macd'] < current_1h['macd_signal'] and 
                              current_1h['macd_hist'] < 0)
                safety_checks.append(macd_bearish)
                
                # RSI不能过度超卖
                rsi_safe = current_1h['rsi'] > 25  # 避免在极度超卖时开空
                safety_checks.append(rsi_safe)
                
            # 2. 15分钟确认信号
            if data_15m is not None and current_idx < len(data_15m):
                current_15m = data_15m.iloc[current_idx]
                prev_15m = data_15m.iloc[max(0, current_idx-1)]
                
                # 价格在EMA下方
                price_below_ema = current_15m['close'] < current_15m['ema_12']
                safety_checks.append(price_below_ema)
                
                # 布林带位置
                bb_position_high = current_15m['price_position'] > 0.7
                safety_checks.append(bb_position_high)
                
        except Exception as e:
            print(f"Short safety check error: {e}")
            return False
        
        # 至少需要通过80%的安全检查
        safety_ratio = sum(safety_checks) / max(len(safety_checks), 1)
        return safety_ratio >= 0.8
    
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
                if ema_strength > 0.005:  # 提高阈值
                    long_strength += 3
                elif ema_strength > 0.002:
                    long_strength += 2
            else:
                ema_strength = abs(current['ema_12'] - current['ema_26']) / current['ema_26']
                if ema_strength > 0.005:
                    short_strength += 3
                elif ema_strength > 0.002:
                    short_strength += 2
            
            # MACD动量评分
            if current['macd'] > current['macd_signal'] and current['macd_hist'] > prev['macd_hist']:
                long_strength += 2
            elif current['macd'] < current['macd_signal'] and current['macd_hist'] < prev['macd_hist']:
                short_strength += 2
            
            # RSI确认评分（更严格的范围）
            if 45 < current['rsi'] < 65:
                if current['rsi'] > 55:
                    long_strength += 1
                elif current['rsi'] < 50:
                    short_strength += 1
            
            # 价格相对SMA位置
            if current['close'] > current['sma_50'] * 1.01:  # 至少高于1%
                long_strength += 1
            elif current['close'] < current['sma_50'] * 0.99:  # 至少低于1%
                short_strength += 1
            
            # 确定主趋势方向
            if long_strength >= short_strength + 2:
                direction = 'bullish' if long_strength >= 4 else 'weak_bullish'
            elif short_strength >= long_strength + 2:
                direction = 'bearish' if short_strength >= 4 else 'weak_bearish'
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
            
            # RSI反转信号（更严格的范围）
            if 20 < current['rsi'] < 40 and current['rsi'] > prev['rsi']:
                signals['long_signals'] += 2
            elif 60 < current['rsi'] < 80 and current['rsi'] < prev['rsi']:
                signals['short_signals'] += 2
                
            # MACD动量信号
            if (current['macd'] > current['macd_signal'] and 
                current['macd_hist'] > 0 and 
                current['macd_hist'] > prev['macd_hist']):
                signals['long_signals'] += 1.5
            elif (current['macd'] < current['macd_signal'] and 
                  current['macd_hist'] < 0 and 
                  current['macd_hist'] < prev['macd_hist']):
                signals['short_signals'] += 1.5
                
            # EMA排列信号
            if (current['close'] > current['ema_12'] > current['ema_26'] and
                current['ema_12'] > prev['ema_12']):
                signals['long_signals'] += 1
            elif (current['close'] < current['ema_12'] < current['ema_26'] and
                  current['ema_12'] < prev['ema_12']):
                signals['short_signals'] += 1
                
            # 成交量确认（门槛恢复）
            if current['volume_ratio'] > 1.5:
                price_change = (current['close'] - current['open']) / current['open']
                if price_change > 0.003:  # 恢复0.3%阈值
                    signals['long_signals'] += 1
                elif price_change < -0.003:
                    signals['short_signals'] += 1
                    
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
            
            # 布林带位置信号（更严格）
            if current['price_position'] < 0.15 and current['close'] > prev['close']:
                signals['long_signals'] += 1.5
            elif current['price_position'] > 0.85 and current['close'] < prev['close']:
                signals['short_signals'] += 1.5
                
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

class SafeShortMultiTimeframeBTCEnv(gym.Env):
    """安全做空版多时间框架环境 - 分离多空持仓跟踪"""
    
    def __init__(self, base_data_path, initial_balance=10000, lookback_window=50, verbose=False):
        super(SafeShortMultiTimeframeBTCEnv, self).__init__()
        
        print("=== Safe Short-Selling Multi-Timeframe BTC Futures Environment ===")
        print("分离多空持仓 + 安全做空机制 + 增强风控")
        
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
        
        # 新的动作空间：11个动作，包含安全做空
        self.action_space = spaces.Discrete(11)
        
        self.action_mapping = {
            0: ('hold', 0),           # 持有
            1: ('buy', 0.20),         # 买入20%
            2: ('buy', 0.50),         # 买入50%
            3: ('buy', 0.80),         # 买入80%（激进）
            4: ('sell', 0.20),        # 减仓20%
            5: ('sell', 0.50),        # 减仓50%
            6: ('sell', 1.00),        # 全平
            7: ('short', 0.20),       # 做空20%（保守）
            8: ('short', 0.40),       # 做空40%（标准）
            9: ('cover', 0.50),       # 部分平空
            10: ('cover', 1.00)       # 全部平空
        }
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(31,), dtype=np.float32  # 修正为31维: 5+10+5+3+3+5
        )
        
        # 增强风险参数
        self.max_long_position_ratio = 0.70   # 多头最大仓位
        self.max_short_position_ratio = 0.40  # 空头最大仓位（更保守）
        self.short_margin_requirement = 1.5   # 做空保证金倍数
        self.atr_stop_multiplier = 1.5        # ATR止损系数
        self.short_atr_stop_multiplier = 1.2  # 做空ATR止损系数（更严格）
        self.min_trade_interval = 3           # 最小交易间隔
        self.last_trade_step = -999
        
        # 做空风控参数
        self.max_short_drawdown = 0.08        # 最大空头回撤8%
        self.short_profit_target = 0.03       # 空头获利目标3%
        self.emergency_cover_threshold = 0.15 # 紧急平空阈值15%
        self.trend_reversal_sensitivity = 0.005 # 趋势反转敏感度
        
        # 新增参数
        self.drawdown_penalty_threshold = 0.02
        self.max_acceptable_drawdown = 0.12
        self.high_quality_profit_threshold = 0.8
        
        self.reset_statistics()
        
        obs, _ = self.reset()
        print("=== Safe short-selling environment initialization completed ===")
    
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
        
        # 分离多空统计
        self.long_pnl_history = []
        self.short_pnl_history = []
        self.forced_covers = 0
        self.emergency_exits = 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        
        # 分离多空持仓跟踪
        self.long_shares = 0      # 多头持仓
        self.short_shares = 0     # 空头持仓
        self.short_entry_value = 0 # 空头开仓价值（用于计算盈亏）
        
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.trades_count = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # 分离多空入场价格和止损
        self.long_entry_price = 0
        self.short_entry_price = 0
        self.long_stop_loss = 0
        self.short_stop_loss = 0
        self.long_position_size = 0
        self.short_position_size = 0
        
        self.last_trade_step = -999
        self.consecutive_trades = 0
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
        position_features = self._get_position_features()  # 新增持仓特征
        
        observation = np.concatenate([
            h1_features, m15_features, m5_features, 
            account_features, risk_features, position_features
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
        
        long_value = self.long_shares * current_price
        short_value = self._calculate_short_value(current_price)
        total_position_value = long_value + short_value
        
        features = [
            total_position_value / self.net_worth if self.net_worth > 0 else 0,
            self.balance / self.net_worth if self.net_worth > 0 else 0,
            (self.net_worth - self.initial_balance) / self.initial_balance
        ]
        
        return np.array(features)
    
    def _get_risk_features(self):
        current_price = self.main_data.iloc[self.current_step]['close']
        
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        
        # 多头止损距离
        long_stop_distance = 0
        if self.long_shares > 0 and self.long_stop_loss > 0:
            long_stop_distance = (current_price - self.long_stop_loss) / current_price
        
        # 空头止损距离
        short_stop_distance = 0
        if self.short_shares > 0 and self.short_stop_loss > 0:
            short_stop_distance = (self.short_stop_loss - current_price) / current_price
        
        features = [drawdown, long_stop_distance, short_stop_distance]
        
        return np.array(features)
    
    def _get_position_features(self):
        """新增持仓特征"""
        current_price = self.main_data.iloc[self.current_step]['close']
        
        # 多头持仓比例
        long_ratio = (self.long_shares * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        # 空头持仓比例  
        short_ratio = (self.short_shares * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        # 多头盈亏比例
        long_pnl_ratio = 0
        if self.long_shares > 0 and self.long_entry_price > 0:
            long_pnl_ratio = (current_price - self.long_entry_price) / self.long_entry_price
        
        # 空头盈亏比例
        short_pnl_ratio = 0
        if self.short_shares > 0 and self.short_entry_price > 0:
            short_pnl_ratio = (self.short_entry_price - current_price) / self.short_entry_price
        
        # 持仓时间
        holding_time = (self.current_step - self.current_holding_start) if self.current_holding_start else 0
        
        features = [long_ratio, short_ratio, long_pnl_ratio, short_pnl_ratio, holding_time / 100]
        
        return np.array(features)
    
    def _find_closest_index(self, data, target_time):
        time_diffs = np.abs(data['timestamp'] - target_time)
        return time_diffs.argmin()
    
    def _calculate_short_value(self, current_price):
        """计算空头持仓价值（正数表示价值）"""
        if self.short_shares <= 0:
            return 0
        
        # 空头盈亏 = 开仓价值 - 当前价值
        current_short_value = self.short_shares * current_price
        short_pnl = self.short_entry_value - current_short_value
        
        # 返回等价的持仓价值（保证金 + 盈亏）
        return self.short_entry_value * self.short_margin_requirement + short_pnl
    
    def step(self, action):
        current_price = self.main_data.iloc[self.current_step]['close']
        
        try:
            current_atr = self.main_data.iloc[self.current_step]['atr']
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.015
        except:
            current_atr = current_price * 0.015
            
        prev_net_worth = self.net_worth
        prev_long_value = self.long_shares * current_price
        prev_short_value = self._calculate_short_value(current_price)
        
        action_type, ratio = self.action_mapping[action]
        trade_executed = False
        
        # 执行交易
        if action_type == 'buy':
            trade_executed = self._execute_buy(current_price, current_atr, ratio)
            if trade_executed:
                self.long_trades += 1
        elif action_type == 'sell':
            trade_executed = self._execute_sell(current_price, ratio)
        elif action_type == 'short':
            trade_executed = self._execute_short(current_price, current_atr, ratio)
            if trade_executed:
                self.short_trades += 1
        elif action_type == 'cover':
            trade_executed = self._execute_cover(current_price, ratio)
        
        # 检查止损和强制平仓
        self._check_stop_losses(current_price, current_atr)
        self._check_forced_liquidation(current_price)
        
        # 更新净值（使用安全的计算方法）
        self.net_worth = self._calculate_safe_net_worth(current_price)
        
        # 计算增强奖励
        reward = self._calculate_enhanced_reward_with_short(
            prev_net_worth, prev_long_value, prev_short_value, 
            action, trade_executed, current_price, current_atr
        )
        
        # 更新统计信息
        self._update_enhanced_statistics(current_price, action_type)
        
        self.current_step += 1
        done = self.current_step >= len(self.main_data) - 1
        
        info = self._get_enhanced_info(current_price, trade_executed, action_type, ratio)
        
        return self._get_observation(), reward, done, False, info
    
    def _calculate_safe_net_worth(self, current_price):
        """安全的净值计算方法 - 分离多空持仓"""
        
        # 现金
        cash = self.balance
        
        # 多头价值
        long_value = self.long_shares * current_price
        
        # 空头价值（包含保证金和盈亏）
        short_value = self._calculate_short_value(current_price)
        
        # 总净值
        net_worth = cash + long_value + short_value
        
        return max(net_worth, 0)  # 确保净值不为负
    
    def _execute_buy(self, current_price, current_atr, ratio):
        
        # 冷却期检查
        if self.current_step - self.last_trade_step < self.min_trade_interval:
            return False
        
        # 计算当前多头仓位比例
        current_long_value = self.long_shares * current_price
        current_long_ratio = current_long_value / self.net_worth if self.net_worth > 0 else 0
        
        # 检查是否超过最大多头仓位限制
        if current_long_ratio >= self.max_long_position_ratio:
            if self.verbose:
                print(f"Long position limit reached {current_long_ratio:.1%}, skipping buy")
            return False
        
        # 计算可以增加的仓位
        max_additional_ratio = self.max_long_position_ratio - current_long_ratio
        actual_ratio = min(ratio, max_additional_ratio)
        
        if actual_ratio < 0.05:  # 最小交易门槛
            return False
            
        available_balance = self.balance * actual_ratio
        
        if available_balance < 100:  # 最小交易金额
            return False
            
        shares_to_buy = available_balance / current_price
        
        if shares_to_buy > 0:
            self.balance -= available_balance
            self.long_shares += shares_to_buy
            self.trades_count += 1
            self.last_trade_step = self.current_step
            self.signal_generator.update_last_trade_step(self.current_step)
            
            # 更新多头入场价格（加权平均）
            if self.long_entry_price > 0 and self.long_position_size > 0:
                total_value = self.long_position_size * self.long_entry_price + available_balance
                total_shares = self.long_position_size / self.long_entry_price + shares_to_buy
                self.long_entry_price = total_value / (total_shares * current_price) * current_price
            else:
                self.long_entry_price = current_price
            
            self.long_position_size += available_balance
            
            # 设置多头止损
            self.long_stop_loss = current_price - (current_atr * self.atr_stop_multiplier)
            self.long_stop_loss = max(self.long_stop_loss, current_price * 0.92)
                
            # 记录持仓开始时间
            if self.current_holding_start is None:
                self.current_holding_start = self.current_step
                
            if self.verbose:
                print(f"Buy: {shares_to_buy:.4f} @ ${current_price:.2f}, Stop loss: ${self.long_stop_loss:.2f}")
            
            return True
            
        return False
    
    def _execute_sell(self, current_price, ratio):
        if self.long_shares <= 0:
            return False
            
        shares_to_sell = self.long_shares * ratio
        sell_amount = shares_to_sell * current_price
        
        if shares_to_sell > 0:
            # 计算该部分的盈亏
            if self.long_entry_price > 0:
                pnl = (current_price - self.long_entry_price) * shares_to_sell
                self.pnl_history.append(pnl)
                self.long_pnl_history.append(pnl)
                
                if pnl > 0:
                    self.profitable_trades += 1
                    self.total_profit += pnl
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(pnl)
            
            self.balance += sell_amount
            self.long_shares -= shares_to_sell
            self.long_position_size -= (self.long_position_size * ratio) if self.long_position_size > 0 else 0
            self.trades_count += 1
            self.last_trade_step = self.current_step
            self.signal_generator.update_last_trade_step(self.current_step)
            
            if self.verbose:
                print(f"Sell: {shares_to_sell:.4f} @ ${current_price:.2f}, Remaining long: {self.long_shares:.4f}")
            
            # 如果完全平多仓，重置相关参数
            if self.long_shares <= 0.001:
                self.long_shares = 0
                self.long_stop_loss = 0
                self.long_entry_price = 0
                self.long_position_size = 0
                
                # 如果也没有空头仓位，记录持仓时间
                if self.short_shares <= 0 and self.current_holding_start is not None:
                    holding_period = self.current_step - self.current_holding_start
                    self.holding_periods.append(holding_period)
                    self.current_holding_start = None
            
            return True
            
        return False
    
    def _execute_short(self, current_price, current_atr, ratio):
        """执行做空操作 - 增强安全检查"""
        
        # 1. 基础检查
        if self.current_step - self.last_trade_step < self.min_trade_interval:
            return False
        
        # 2. 信号安全性检查
        current_time = self.main_data.iloc[self.current_step]['timestamp']
        data_1h = self.all_data.get('1h')
        data_5m = self.all_data.get('5m')
        
        if data_1h is not None and data_5m is not None:
            h1_idx = self._find_closest_index(data_1h, current_time)
            m5_idx = self._find_closest_index(data_5m, current_time)
            
            scores = self.signal_generator.calculate_signal_scores(
                data_1h, self.main_data, data_5m, 
                min(h1_idx, self.current_step),
                self.current_step
            )
            
            # 检查做空安全性
            if not scores['safe_to_short']:
                if self.verbose:
                    print(f"Short signal not safe, skipping short order")
                return False
        
        # 3. 仓位检查
        current_short_value = self.short_shares * current_price
        current_short_ratio = current_short_value / self.net_worth if self.net_worth > 0 else 0
        
        if current_short_ratio >= self.max_short_position_ratio:
            if self.verbose:
                print(f"Short position limit reached {current_short_ratio:.1%}, skipping short")
            return False
        
        # 4. 计算可开空仓位
        max_additional_ratio = self.max_short_position_ratio - current_short_ratio
        actual_ratio = min(ratio, max_additional_ratio)
        
        if actual_ratio < 0.05:
            return False
        
        # 5. 保证金检查
        short_value = self.net_worth * actual_ratio
        required_margin = short_value * self.short_margin_requirement
        
        if required_margin > self.balance:
            if self.verbose:
                print(f"Insufficient margin for short: required {required_margin:.2f}, available {self.balance:.2f}")
            return False
        
        # 6. 执行做空
        shares_to_short = short_value / current_price
        
        if shares_to_short > 0:
            self.balance -= required_margin
            self.short_shares += shares_to_short
            self.trades_count += 1
            self.last_trade_step = self.current_step
            self.signal_generator.update_last_trade_step(self.current_step)
            
            # 更新空头入场价格和价值
            if self.short_entry_price > 0 and self.short_entry_value > 0:
                # 加权平均
                total_old_value = self.short_entry_value
                new_value = shares_to_short * current_price
                total_new_value = total_old_value + new_value
                total_shares = self.short_entry_value / self.short_entry_price + shares_to_short
                self.short_entry_price = total_new_value / total_shares
                self.short_entry_value = total_new_value
            else:
                self.short_entry_price = current_price
                self.short_entry_value = shares_to_short * current_price
            
            # 设置空头止损（更严格）
            self.short_stop_loss = current_price + (current_atr * self.short_atr_stop_multiplier)
            self.short_stop_loss = min(self.short_stop_loss, current_price * 1.08)
            
            # 记录持仓开始时间
            if self.current_holding_start is None:
                self.current_holding_start = self.current_step
                
            if self.verbose:
                print(f"Short: {shares_to_short:.4f} @ ${current_price:.2f}, Stop loss: ${self.short_stop_loss:.2f}")
                print(f"Margin used: ${required_margin:.2f}")
            
            return True
            
        return False
    
    def _execute_cover(self, current_price, ratio):
        """执行平空操作"""
        if self.short_shares <= 0:
            return False
            
        shares_to_cover = self.short_shares * ratio
        
        if shares_to_cover > 0:
            # 计算空头盈亏
            if self.short_entry_price > 0:
                pnl = (self.short_entry_price - current_price) * shares_to_cover
                self.pnl_history.append(pnl)
                self.short_pnl_history.append(pnl)
                
                if pnl > 0:
                    self.profitable_trades += 1
                    self.total_profit += pnl
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(pnl)
                    
                # 释放保证金并结算盈亏
                margin_to_release = (shares_to_cover / self.short_shares) * self.short_entry_value * self.short_margin_requirement
                self.balance += margin_to_release + pnl
            
            self.short_shares -= shares_to_cover
            self.short_entry_value -= (self.short_entry_value * ratio) if self.short_entry_value > 0 else 0
            self.trades_count += 1
            self.last_trade_step = self.current_step
            self.signal_generator.update_last_trade_step(self.current_step)
            
            if self.verbose:
                print(f"Cover: {shares_to_cover:.4f} @ ${current_price:.2f}, Remaining short: {self.short_shares:.4f}")
            
            # 如果完全平空仓，重置相关参数
            if self.short_shares <= 0.001:
                self.short_shares = 0
                self.short_stop_loss = 0
                self.short_entry_price = 0
                self.short_entry_value = 0
                
                # 如果也没有多头仓位，记录持仓时间
                if self.long_shares <= 0 and self.current_holding_start is not None:
                    holding_period = self.current_step - self.current_holding_start
                    self.holding_periods.append(holding_period)
                    self.current_holding_start = None
            
            return True
            
        return False
    
    def _check_stop_losses(self, current_price, current_atr):
        """检查多空止损"""
        
        # 多头止损
        if self.long_shares > 0 and self.long_stop_loss > 0:
            if current_price <= self.long_stop_loss:
                if self.verbose:
                    print(f"Long stop loss triggered: Price {current_price:.2f} <= Stop {self.long_stop_loss:.2f}")
                self._execute_sell(current_price, 1.0)
            else:
                # 动态追踪止损
                try:
                    new_stop_loss = current_price - (current_atr * self.atr_stop_multiplier)
                    if new_stop_loss > self.long_stop_loss:
                        self.long_stop_loss = new_stop_loss
                except:
                    pass
        
        # 空头止损
        if self.short_shares > 0 and self.short_stop_loss > 0:
            if current_price >= self.short_stop_loss:
                if self.verbose:
                    print(f"Short stop loss triggered: Price {current_price:.2f} >= Stop {self.short_stop_loss:.2f}")
                self._execute_cover(current_price, 1.0)
                self.forced_covers += 1
            else:
                # 空头动态追踪止损（向下调整）
                try:
                    new_stop_loss = current_price + (current_atr * self.short_atr_stop_multiplier)
                    if new_stop_loss < self.short_stop_loss:
                        self.short_stop_loss = new_stop_loss
                except:
                    pass
    
    def _check_forced_liquidation(self, current_price):
        """检查强制平仓条件"""
        
        # 检查空头风险
        if self.short_shares > 0 and self.short_entry_price > 0:
            # 计算空头浮亏比例
            short_loss_ratio = (current_price - self.short_entry_price) / self.short_entry_price
            
            # 紧急平空条件
            if short_loss_ratio > self.emergency_cover_threshold:
                if self.verbose:
                    print(f"Emergency short cover triggered: Loss {short_loss_ratio:.1%} > {self.emergency_cover_threshold:.1%}")
                self._execute_cover(current_price, 1.0)
                self.emergency_exits += 1
                return
            
            # 检查保证金充足性
            current_short_value = self.short_shares * current_price
            unrealized_loss = max(0, current_short_value - self.short_entry_value)
            required_margin = self.short_entry_value * self.short_margin_requirement
            
            if self.balance < unrealized_loss + required_margin * 0.5:  # 维持保证金不足
                if self.verbose:
                    print(f"Margin call: Forced short liquidation")
                self._execute_cover(current_price, 1.0)
                self.forced_covers += 1
        
        # 检查整体净值
        if self.net_worth < self.initial_balance * 0.3:  # 净值低于30%强制平仓
            if self.verbose:
                print(f"Total liquidation: Net worth too low")
            if self.long_shares > 0:
                self._execute_sell(current_price, 1.0)
            if self.short_shares > 0:
                self._execute_cover(current_price, 1.0)
            self.emergency_exits += 1
    
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
        long_ratio = (self.long_shares * current_price) / self.net_worth if self.net_worth > 0 else 0
        short_ratio = (self.short_shares * current_price) / self.net_worth if self.net_worth > 0 else 0
        self.positions_history.append((long_ratio, short_ratio))
        
        # 连续交易计数
        if action_type != 'hold':
            if self.last_action_type != 'hold' and self.last_action_type != action_type:
                self.consecutive_trades += 1
            else:
                self.consecutive_trades = 0
        else:
            self.consecutive_trades = 0
            
        self.last_action_type = action_type
    
    def _calculate_enhanced_reward_with_short(self, prev_net_worth, prev_long_value, prev_short_value, 
                                            action, trade_executed, current_price, current_atr):
        """支持做空的增强奖励函数"""
        
        # 1. 基础盈亏奖励
        net_worth_change = self.net_worth - prev_net_worth
        base_reward = net_worth_change / self.initial_balance * 15  # 稍微放大基础奖励
        
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
                if action_type == 'buy' and scores['long_score'] >= 3:
                    signal_reward = 0.012 * scores['long_score']
                elif action_type == 'short' and scores['short_score'] >= 4 and scores['safe_to_short']:
                    signal_reward = 0.015 * scores['short_score']  # 做空奖励稍高，因为更难
                elif action_type in ['sell', 'cover'] and net_worth_change > 0:
                    signal_reward = 0.008  # 获利了结奖励
        
        # 3. 趋势持仓奖励
        trend_reward = 0
        try:
            current_time = self.main_data.iloc[self.current_step-1]['timestamp']
            data_1h = self.all_data.get('1h')
            if data_1h is not None:
                h1_idx = self._find_closest_index(data_1h, current_time)
                h1_trend = self.signal_generator.get_enhanced_1h_trend_score(data_1h, h1_idx)
                
                # 顺势持仓奖励
                if self.long_shares > 0 and h1_trend['direction'] in ['bullish', 'weak_bullish']:
                    trend_reward = 0.006
                elif self.short_shares > 0 and h1_trend['direction'] in ['bearish', 'weak_bearish']:
                    trend_reward = 0.008  # 做空顺势奖励更高
                # 逆势持仓惩罚
                elif self.long_shares > 0 and h1_trend['direction'] in ['bearish', 'weak_bearish']:
                    trend_reward = -0.004
                elif self.short_shares > 0 and h1_trend['direction'] in ['bullish', 'weak_bullish']:
                    trend_reward = -0.008  # 做空逆势惩罚更重
        except:
            pass
        
        # 4. 高质量止盈奖励
        profit_taking_reward = 0
        if trade_executed and action in [4, 5, 6, 9, 10]:  # 减仓或平仓动作
            if action in [4, 5, 6] and self.long_entry_price > 0:  # 多头止盈
                profit_ratio = (current_price - self.long_entry_price) / self.long_entry_price
                if profit_ratio > self.high_quality_profit_threshold * current_atr / current_price:
                    profit_taking_reward = 0.020
            elif action in [9, 10] and self.short_entry_price > 0:  # 空头止盈
                profit_ratio = (self.short_entry_price - current_price) / self.short_entry_price
                if profit_ratio > self.high_quality_profit_threshold * current_atr / current_price:
                    profit_taking_reward = 0.025  # 空头止盈奖励更高
        
        # 5. 仓位管理奖励
        position_reward = 0
        long_ratio = (self.long_shares * current_price) / self.net_worth if self.net_worth > 0 else 0
        short_ratio = (self.short_shares * current_price) / self.net_worth if self.net_worth > 0 else 0
        total_position_ratio = long_ratio + short_ratio
        cash_ratio = self.balance / self.net_worth if self.net_worth > 0 else 0
        
        # 合理仓位奖励
        if 0.3 <= total_position_ratio <= 0.7:
            position_reward += 0.006
        elif 0.7 < total_position_ratio <= 0.9:
            position_reward += 0.003
        
        # 现金储备奖励
        if 0.1 <= cash_ratio <= 0.4:
            position_reward += 0.004
        
        # 6. 风险惩罚
        risk_penalty = 0
        
        # 交易成本惩罚
        if trade_executed:
            risk_penalty += 0.001
        
        # 过度仓位惩罚
        if total_position_ratio > 0.9:
            risk_penalty += (total_position_ratio - 0.9) * 0.05
        
        # 频繁交易惩罚
        if self.consecutive_trades > 3:
            risk_penalty += 0.005
        
        # 空头特别风险惩罚
        if self.short_shares > 0:
            # 空头仓位过大惩罚
            if short_ratio > 0.3:
                risk_penalty += (short_ratio - 0.3) * 0.1
            
            # 空头亏损惩罚
            if self.short_entry_price > 0:
                short_loss_ratio = max(0, (current_price - self.short_entry_price) / self.short_entry_price)
                if short_loss_ratio > 0.05:  # 亏损超过5%
                    risk_penalty += short_loss_ratio * 0.2
        
        # 整体回撤惩罚
        if self.max_drawdown > self.max_acceptable_drawdown:
            risk_penalty += (self.max_drawdown - self.max_acceptable_drawdown) * 0.8
        
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
                    
                    # 逆势开仓惩罚
                    if action_type == 'buy' and h1_trend['direction'] in ['bearish', 'weak_bearish']:
                        trend_penalty = 0.008
                    elif action_type == 'short' and h1_trend['direction'] in ['bullish', 'weak_bullish']:
                        trend_penalty = 0.015  # 上涨趋势做空惩罚更重
            except:
                pass
        
        # 8. 强制平仓惩罚
        forced_penalty = 0
        if hasattr(self, 'forced_covers') and hasattr(self, 'emergency_exits'):
            if self.forced_covers > 0 or self.emergency_exits > 0:
                forced_penalty = 0.02
        
        # 总奖励计算
        total_reward = (base_reward + signal_reward + trend_reward + 
                       profit_taking_reward + position_reward - 
                       risk_penalty - trend_penalty - forced_penalty)
        
        # 限制奖励范围
        total_reward = np.clip(total_reward, -0.15, 0.15)
        
        return total_reward
    
    def _get_enhanced_info(self, current_price, trade_executed, action_type, ratio):
        total_trades = self.profitable_trades + self.losing_trades
        win_rate = self.profitable_trades / max(total_trades, 1) * 100
        
        profit_loss_ratio = 0
        if self.total_loss > 0:
            profit_loss_ratio = self.total_profit / self.total_loss
        
        # 计算年化收益率
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
        
        # 多空分别统计
        long_pnl = sum(self.long_pnl_history) if hasattr(self, 'long_pnl_history') else 0
        short_pnl = sum(self.short_pnl_history) if hasattr(self, 'short_pnl_history') else 0
        
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'long_shares': self.long_shares,
            'short_shares': self.short_shares,
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
            'long_position_ratio': (self.long_shares * current_price) / self.net_worth if self.net_worth > 0 else 0,
            'short_position_ratio': (self.short_shares * current_price) / self.net_worth if self.net_worth > 0 else 0,
            'long_stop_loss': self.long_stop_loss,
            'short_stop_loss': self.short_stop_loss,
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'forced_covers': getattr(self, 'forced_covers', 0),
            'emergency_exits': getattr(self, 'emergency_exits', 0),
            'consecutive_trades': self.consecutive_trades
        }
    
    def get_enhanced_performance_summary(self):
        """增强版性能统计 - 包含多空分析"""
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
        
        # 多空分别统计
        long_pnl = sum(self.long_pnl_history) if hasattr(self, 'long_pnl_history') else 0
        short_pnl = sum(self.short_pnl_history) if hasattr(self, 'short_pnl_history') else 0
        total_directional_trades = self.long_trades + self.short_trades
        
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
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'long_ratio': self.long_trades / max(total_directional_trades, 1) * 100,
            'short_ratio': self.short_trades / max(total_directional_trades, 1) * 100,
            'forced_covers': getattr(self, 'forced_covers', 0),
            'emergency_exits': getattr(self, 'emergency_exits', 0),
            'final_net_worth': self.net_worth,
            'final_balance': self.balance,
            'final_long_position': self.long_shares,
            'final_short_position': self.short_shares
        }

def train_safe_short_agent(base_data_path, total_timesteps=200000, save_path="safe_short_btc_agent"):
    """训练安全做空智能体"""
    print("=== Creating Safe Short-Selling Multi-Timeframe Environment ===")
    
    try:
        env = SafeShortMultiTimeframeBTCEnv(base_data_path, verbose=False)
        env = DummyVecEnv([lambda: env])
        
        print("=== Initializing Enhanced PPO Model for Short-Selling ===")
        
        import torch.nn as nn
        
        # 优化超参数 - 更保守的设置
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=2e-4,      # 降低学习率，更稳定
            n_steps=3072,            # 增加步数，更好的经验收集
            batch_size=128,          # 增加批量大小
            n_epochs=8,              # 减少epochs，防止过拟合
            gamma=0.995,             # 提高折扣因子，重视长期收益
            ent_coef=0.03,           # 降低探索，更保守
            clip_range=0.15,         # 降低裁剪范围
            max_grad_norm=0.3,       # 更严格的梯度裁剪
            policy_kwargs={
                "net_arch": [512, 384, 256, 128, 64],  # 更深的网络
                "activation_fn": nn.ReLU
            },
            tensorboard_log="./safe_short_ppo_tensorboard/"
        )
        
        print(f"=== Starting Safe Short Training, Total Timesteps: {total_timesteps} ===")
        model.learn(total_timesteps=total_timesteps)
        
        print(f"=== Saving Safe Short Model: {save_path} ===")
        model.save(save_path)
        
        return model
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_safe_short_agent(model_path, base_data_path, episodes=3):
    """评估安全做空智能体"""
    print("=== Evaluating Safe Short-Selling Model ===")
    
    try:
        env = SafeShortMultiTimeframeBTCEnv(base_data_path, verbose=False)
        model = PPO.load(model_path)
        
        all_results = []
        action_distributions = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            action_counts = {i: 0 for i in range(11)}
            
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
            action_names = ['Hold', 'Buy20%', 'Buy50%', 'Buy80%', 'Sell20%', 'Sell50%', 'SellAll', 
                          'Short20%', 'Short40%', 'Cover50%', 'CoverAll']
            episode_distribution = {}
            for i, name in enumerate(action_names):
                percentage = action_counts[i] / step_count * 100
                episode_distribution[name] = percentage
            action_distributions.append(episode_distribution)
            
            print(f"\nEpisode {episode+1} Safe Short Results:")
            print(f"  Episode reward: {episode_reward:.4f}")
            print(f"  Total return: {perf_summary['total_return']:.2f}%")
            print(f"  Annualized return: {perf_summary['annualized_return']:.2f}%")
            print(f"  Max drawdown: {perf_summary['max_drawdown']:.2f}%")
            print(f"  Total trades: {perf_summary['total_trades']}")
            print(f"  Win rate: {perf_summary['win_rate']:.1f}%")
            print(f"  Profit/Loss ratio: {perf_summary['profit_loss_ratio']:.2f}")
            print(f"  Sharpe ratio: {perf_summary['sharpe_ratio']:.2f}")
            print(f"  Long trades: {perf_summary['long_trades']} ({perf_summary['long_ratio']:.1f}%)")
            print(f"  Short trades: {perf_summary['short_trades']} ({perf_summary['short_ratio']:.1f}%)")
            print(f"  Long PnL: ${perf_summary['long_pnl']:.2f}")
            print(f"  Short PnL: ${perf_summary['short_pnl']:.2f}")
            print(f"  Forced covers: {perf_summary['forced_covers']}")
            print(f"  Emergency exits: {perf_summary['emergency_exits']}")
            
            print(f"  Action Distribution:")
            for name, percentage in episode_distribution.items():
                if percentage > 0.1:
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
            avg_long_ratio = np.mean([r['long_ratio'] for r in all_results])
            avg_short_ratio = np.mean([r['short_ratio'] for r in all_results])
            avg_long_pnl = np.mean([r['long_pnl'] for r in all_results])
            avg_short_pnl = np.mean([r['short_pnl'] for r in all_results])
            avg_forced_covers = np.mean([r['forced_covers'] for r in all_results])
            
            print(f"\n=== Safe Short Average Performance ===")
            print(f"Average total return: {avg_return:.2f}%")
            print(f"Average annualized return: {avg_annualized:.2f}%")
            print(f"Average max drawdown: {avg_drawdown:.2f}%")
            print(f"Average Sharpe ratio: {avg_sharpe:.2f}")
            print(f"Average win rate: {avg_win_rate:.1f}%")
            print(f"Average trades: {avg_trades:.0f}")
            print(f"Average profit/loss ratio: {avg_pnl_ratio:.2f}")
            print(f"Average long trade ratio: {avg_long_ratio:.1f}%")
            print(f"Average short trade ratio: {avg_short_ratio:.1f}%")
            print(f"Average long PnL: ${avg_long_pnl:.2f}")
            print(f"Average short PnL: ${avg_short_pnl:.2f}")
            print(f"Average forced covers: {avg_forced_covers:.1f}")
        
        return all_results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    base_data_path = "/data1/linbingqian/liujzh33/Agent/data/btc_2025_06_07"
    
    try:
        print("\n=== 1. Starting Safe Short-Selling Training ===")
        model = train_safe_short_agent(
            base_data_path, 
            total_timesteps=200000,
            save_path="safe_short_multi_timeframe_btc_agent"
        )
        
        if model is not None:
            print("Safe short-selling training completed!")
            
            print("\n=== 2. Starting Safe Short-Selling Evaluation ===")
            results = evaluate_safe_short_agent(
                "safe_short_multi_timeframe_btc_agent", 
                base_data_path,
                episodes=3
            )
            
            if results:
                print("Safe short-selling evaluation completed!")
            else:
                print("Evaluation failed")
        else:
            print("Training failed")
            
        print(f"\nSafe short model save location: safe_short_multi_timeframe_btc_agent.zip")
        
    except Exception as e:
        print(f"System execution failed: {e}")
        import traceback
        traceback.print_exc()