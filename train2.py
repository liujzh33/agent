"""
offline_ppo_btcusdt.py
======================

依赖:
  pip install pandas ta gymnasium stable-baselines3 tqdm

运行:
  python offline_ppo_btcusdt.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import ta  # 技术指标库
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

# ------------------------------------------------------------------
# 1. 路径与列名
# ------------------------------------------------------------------
ROOT = Path("data/btc_2025_06_07")
KLINE_CSV ="./data/btc_2025_06_07/spot/trades/data/spot/monthly/trades/BTCUSDT/2025-06-01_2025-07-31/BTCUSDT-trades-2025-06.csv"
TRADES_CSV =  "./data/btc_2025_06_07/spot/klines/data/spot/monthly/klines/BTCUSDT/15m/2025-06-01_2025-07-31/BTCUSDT-15m-2025-06.csv"

KLINE_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]
TRADE_COLS = ["id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]

# ------------------------------------------------------------------
# 2. 数据预处理
# ------------------------------------------------------------------
def load_kline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=KLINE_COLS)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df = df.astype(float)
    # 计算技术指标
    donchian_window = 20
    df["donchian_high"] = df["high"].rolling(donchian_window).max()
    df["donchian_low"] = df["low"].rolling(donchian_window).min()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df = df.dropna().copy()
    return df

def load_trades_agg(path: Path, bar_index) -> pd.DataFrame:
    """将 6 GB 成交明细分块读取，聚合到 15 分钟 K 线。
    返回的 DataFrame index 与 bar_index 一致。
    这里只聚合两条常用统计: 成交笔数 & 买方占比。
    """
    chunk_iter = pd.read_csv(
        path, header=None, names=TRADE_COLS,
        chunksize=5_000_000  # 可根据机器内存调整
    )
    agg = pd.Series(0, index=bar_index, dtype="int64")
    buy_qty = pd.Series(0.0, index=bar_index, dtype="float64")
    total_qty = pd.Series(0.0, index=bar_index, dtype="float64")

    for chunk in tqdm(chunk_iter, desc="Aggregating trades"):
        chunk["datetime"] = pd.to_datetime(chunk["time"], unit="ms", utc=True)
        chunk.set_index("datetime", inplace=True)
        # 先对每笔成交映射到 15 分钟桶
        resampled = chunk.resample("15T")
        agg = agg.add(resampled["id"].count(), fill_value=0)
        buy_mask = ~chunk["isBuyerMaker"].astype(bool)  # 买方主动成交
        buy_qty = buy_qty.add(resampled["qty"].sum().where(buy_mask), fill_value=0)
        total_qty = total_qty.add(resampled["qty"].sum(), fill_value=0)

    trades_df = pd.DataFrame({
        "trade_count": agg,
        "buy_ratio": np.where(total_qty > 0, buy_qty / total_qty, 0.0)
    }).fillna(0.0)
    return trades_df.loc[bar_index]  # 与 K 线对齐

# ------------------------------------------------------------------
# 3. 构建交易环境
# ------------------------------------------------------------------
class BTCTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_cash: float = 100_000, fee: float = 0.0004):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.fee = fee

        # 动作空间: 0-Hold, 1-Buy (100%), 2-Sell (平仓/做空100%)
        self.action_space = spaces.Discrete(3)

        # 状态向量: 价格/指标 + 持仓信息 (price, donchianH, donchianL, rsi, atr, pos, cash_perc)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0  # -1 做空, 0 空仓, 1 做多
        self.cash = self.initial_cash
        self.hold_price = 0.0
        self.net_worth = self.initial_cash
        return self._get_obs(), {}

    def step(self, action):
        done = False
        price = self.df.loc[self.t, "close"]

        # 计算手续费
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # 先平空
                self.cash -= (price - self.hold_price) * (self.initial_cash / self.hold_price)
            self.position = 1
            self.hold_price = price
            self.cash *= (1 - self.fee)

        elif action == 2 and self.position >= 0:  # Sell / Short
            if self.position == 1:  # 先平多
                self.cash += (price - self.hold_price) * (self.initial_cash / self.hold_price)
            self.position = -1
            self.hold_price = price
            self.cash *= (1 - self.fee)

        # 更新净值
        self.net_worth = (
            self.cash + (self.initial_cash / self.hold_price) * price * self.position
            if self.position != 0 else self.cash
        )
        reward = self.net_worth - self.initial_cash  # 主奖励【净值变化】
        # 过度交易惩罚
        reward -= 1 if action != 0 else 0

        self.t += 1
        if self.t >= len(self.df) - 1:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        row = self.df.iloc[self.t]
        cash_perc = self.cash / self.initial_cash
        obs = np.array([
            row["close"], row["donchian_high"], row["donchian_low"],
            row["rsi"], row["atr"],
            self.position, cash_perc
        ], dtype=np.float32)
        return obs

# ------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------
def main():
    # 读取 K 线
    kline_df = load_kline(KLINE_CSV)
    # 读取成交聚合(可选；若 K 线已含 volume/trades 可跳过)
    # trades_df = load_trades_agg(TRADES_CSV, kline_df.index)
    # kline_df = kline_df.join(trades_df)

    # 构建环境
    env = DummyVecEnv([lambda: BTCTradingEnv(kline_df)])

    # 训练 PPO
    model = PPO("MlpPolicy", env, verbose=1, batch_size=2048, n_steps=2048,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                tensorboard_log="./ppo_btc_tensorboard/")
    model.learn(total_timesteps=200_000)
    model.save("ppo_btcusdt_15m")

    print("✅ 训练完成，模型保存在 ppo_btcusdt_15m.zip")

if __name__ == "__main__":
    main()
