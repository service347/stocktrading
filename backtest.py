"""
backtest.py — hajime-f/stocktrading の戦略を XGBoost で再実装してバックテスト

元リポジトリの戦略:
  - UPモデル:   翌営業日の終値が当日比 +0.5% 超かを予測
  - DOWNモデル: 翌営業日の終値が当日比 -0.5% 超かを予測
  - 信頼度 0.7 以上の銘柄を対象に 始値で買い/売り → 終値で決済
  - 特徴量: MA5, MA25, MACD, RSI, Bollinger Bands など（元コードと同一）

変更点:
  - TensorFlow/LSTM → XGBoost（Python 3.9 互換のため）
  - J-Quants API → yfinance（APIキー不要）
  - Walk-forward バックテスト（ルックアヘッドバイアスなし）
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 設定 ──────────────────────────────────────────────
TICKERS = [
    "7203.T",  # トヨタ
    "6758.T",  # ソニー
    "9984.T",  # ソフトバンクG
    "4063.T",  # 信越化学
    "6861.T",  # キーエンス
    "8306.T",  # 三菱UFJ
    "9432.T",  # NTT
    "7974.T",  # 任天堂
    "6367.T",  # ダイキン
    "4519.T",  # 中外製薬
    "8035.T",  # 東京エレクトロン
    "2914.T",  # JT
    "6098.T",  # リクルート
    "7267.T",  # ホンダ
    "6501.T",  # 日立
]

WINDOW        = 30       # 時系列ウィンドウ（元コードと同じ）
THRESHOLD     = 0.55     # XGBoostはLSTMより確率が保守的なため0.7→0.55に調整
UP_TARGET     = 1.005    # +0.5%
DOWN_TARGET   = 0.995    # -0.5%
TRAIN_MONTHS  = 12       # 学習期間（月）
TEST_MONTHS   = 1        # テスト期間（月）
START_DATE    = "2022-01-01"
END_DATE      = "2025-03-31"
INITIAL_CASH  = 1_000_000  # 初期資金（円）
TRADE_AMOUNT  = 100_000    # 1銘柄あたりの取引金額（円）

# XGBoost向け特徴量: 直近値 + ラグ特徴量（LSTMの時系列処理を代替）
BASE_COLS = [
    "MA5_rate", "MA25_rate", "MACD_rate", "RSI_rate", "Upper_rate",
    "MA_diff", "close_rate", "trunk", "HISTOGRAM", "Volume_rate",
]
LAG_DAYS = [1, 3, 5, 10]  # ラグ特徴量の日数

# ── 特徴量エンジニアリング（元コードと同一） ──────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA5"]       = d["Close"].rolling(5).mean()
    d["MA25"]      = d["Close"].rolling(25).mean()
    d["MACD"]      = d["Close"].ewm(span=12).mean() - d["Close"].ewm(span=26).mean()
    d["SIGNAL"]    = d["MACD"].ewm(span=9).mean()
    d["HISTOGRAM"] = d["MACD"] - d["SIGNAL"]

    sma20          = d["Close"].rolling(20).mean()
    std20          = d["Close"].rolling(20).std()
    d["Upper"]     = sma20 + std20 * 2

    delta          = d["Close"].diff()
    gain           = delta.where(delta > 0, 0).rolling(14).mean()
    loss           = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["RSI"]       = 100 - (100 / (1 + gain / loss))

    d["close_rate"]  = d["Close"].pct_change()
    d["trunk"]       = d["Open"] - d["Close"]
    d["MA5_rate"]    = (d["Close"] - d["MA5"]) / d["MA5"]
    d["MA25_rate"]   = (d["Close"] - d["MA25"]) / d["MA25"]
    d["MACD_rate"]   = (d["MACD"] - d["SIGNAL"]) / d["SIGNAL"].replace(0, np.nan)
    d["RSI_rate"]    = (d["RSI"] - 50) / 50
    d["Upper_rate"]  = (d["Close"] - d["Upper"]) / d["Upper"]
    d["MA_diff"]     = d["MA5"] - d["MA25"]
    d["Volume_rate"] = d["Volume"].pct_change()

    # ラグ特徴量（XGBoostで時系列的な文脈を持たせる）
    for col in ["close_rate", "RSI_rate", "MA5_rate", "HISTOGRAM", "Volume_rate"]:
        for lag in LAG_DAYS:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)

    return d.dropna()

def get_feature_cols(df: pd.DataFrame):
    lag_cols = [c for c in df.columns if "_lag" in c]
    return BASE_COLS + lag_cols

# ── データ取得 ──────────────────────────────────────────
def fetch_data(tickers):
    print("株価データを取得中...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    data = {}
    for t in tickers:
        try:
            df = raw.xs(t, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) > WINDOW + 60:
                data[t] = add_features(df)
        except Exception:
            pass

    print(f"  取得成功: {len(data)} 銘柄 / {len(tickers)} 銘柄")
    return data

# ── ラベル生成 ──────────────────────────────────────────
def make_labels(close_series, target_ratio):
    labels = []
    for i in range(len(close_series) - 1):
        cur = close_series.iloc[i]
        nxt = close_series.iloc[i + 1]
        if target_ratio > 1:
            labels.append(1 if nxt >= cur * target_ratio else 0)
        else:
            labels.append(1 if nxt <= cur * target_ratio else 0)
    return labels

# ── Walk-forward バックテスト ──────────────────────────
def run_backtest(data):
    dates_all = sorted({d for df in data.values() for d in df.index})
    dates_all = pd.DatetimeIndex(dates_all)

    trades   = []
    cash     = INITIAL_CASH
    equity   = [INITIAL_CASH]
    eq_dates = [dates_all[0]]

    train_end = dates_all[0] + pd.DateOffset(months=TRAIN_MONTHS)
    step      = pd.DateOffset(months=TEST_MONTHS)

    period = 0
    while True:
        test_start = train_end
        test_end   = test_start + step

        if test_end > dates_all[-1]:
            break

        train_dates = dates_all[(dates_all >= dates_all[0]) & (dates_all < train_end)]
        test_dates  = dates_all[(dates_all >= test_start) & (dates_all < test_end)]

        if len(train_dates) < WINDOW + 30 or len(test_dates) == 0:
            train_end += step
            continue

        period += 1
        print(f"\n[Period {period}] 学習: {train_dates[0].date()} ～ {train_dates[-1].date()} "
              f"| テスト: {test_dates[0].date()} ～ {test_dates[-1].date()}")

        # ── 学習データ構築 ──
        X_up, y_up, X_dn, y_dn = [], [], [], []

        for ticker, df in data.items():
            df_train = df[df.index < train_end]
            if len(df_train) < WINDOW + 1:
                continue

            feat_cols = get_feature_cols(df_train)
            scaler = StandardScaler()
            feats  = pd.DataFrame(
                scaler.fit_transform(df_train[feat_cols]),
                columns=feat_cols, index=df_train.index
            )
            labels_up = make_labels(df_train["Close"], UP_TARGET)
            labels_dn = make_labels(df_train["Close"], DOWN_TARGET)

            # XGBoost: 直近1日の特徴量のみ使用（ラグ特徴量で時系列を内包）
            for i in range(len(feats) - 1):
                X_up.append(feats.iloc[i].values)
                y_up.append(labels_up[i])
                X_dn.append(feats.iloc[i].values)
                y_dn.append(labels_dn[i])

        if len(X_up) < 100:
            train_end += step
            continue

        X_up = np.array(X_up)
        y_up = np.array(y_up)
        X_dn = np.array(X_dn)
        y_dn = np.array(y_dn)

        # ── モデル学習 ──
        params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        model_up = xgb.XGBClassifier(**params)
        model_dn = xgb.XGBClassifier(**params)
        model_up.fit(X_up, y_up)
        model_dn.fit(X_dn, y_dn)

        print(f"  学習サンプル数: {len(X_up):,}")

        # ── テスト期間の取引シミュレーション ──
        period_trades = 0
        for i, test_date in enumerate(test_dates[:-1]):
            next_date = test_dates[i + 1]

            long_signals  = []
            short_signals = []

            for ticker, df in data.items():
                df_hist = df[df.index <= test_date]
                if len(df_hist) < WINDOW:
                    continue

                feat_cols = get_feature_cols(df_hist)
                scaler = StandardScaler()
                feats  = scaler.fit_transform(df_hist[feat_cols])
                x      = feats[-1:].reshape(1, -1)

                prob_up = model_up.predict_proba(x)[0][1]
                prob_dn = model_dn.predict_proba(x)[0][1]

                if prob_up >= THRESHOLD:
                    long_signals.append((ticker, prob_up))
                if prob_dn >= THRESHOLD:
                    short_signals.append((ticker, prob_dn))

            # 次の日の始値で入り、終値で決済
            for ticker, prob in long_signals:
                if next_date not in data[ticker].index:
                    continue
                row        = data[ticker].loc[next_date]
                entry      = float(row["Open"])
                exit_price = float(row["Close"])
                shares     = int(TRADE_AMOUNT / entry / 100) * 100
                if shares == 0:
                    continue
                pnl = (exit_price - entry) * shares
                cash += pnl
                trades.append({
                    "date": next_date, "ticker": ticker,
                    "side": "LONG", "entry": entry, "exit": exit_price,
                    "shares": shares, "pnl": pnl, "prob": prob,
                })
                period_trades += 1

            for ticker, prob in short_signals:
                if next_date not in data[ticker].index:
                    continue
                row        = data[ticker].loc[next_date]
                entry      = float(row["Open"])
                exit_price = float(row["Close"])
                shares     = int(TRADE_AMOUNT / entry / 100) * 100
                if shares == 0:
                    continue
                pnl = (entry - exit_price) * shares
                cash += pnl
                trades.append({
                    "date": next_date, "ticker": ticker,
                    "side": "SHORT", "entry": entry, "exit": exit_price,
                    "shares": shares, "pnl": pnl, "prob": prob,
                })
                period_trades += 1

            equity.append(cash)
            eq_dates.append(next_date)

        print(f"  取引回数: {period_trades} 回")
        train_end += step

    return trades, equity, eq_dates

# ── パフォーマンス集計 ──────────────────────────────────
def print_stats(trades, equity, eq_dates):
    df_t = pd.DataFrame(trades)
    if df_t.empty:
        print("取引なし")
        return None, None

    total_pnl     = df_t["pnl"].sum()
    win_rate      = (df_t["pnl"] > 0).mean() * 100
    avg_win       = df_t[df_t["pnl"] > 0]["pnl"].mean() if (df_t["pnl"] > 0).any() else 0
    avg_loss      = df_t[df_t["pnl"] < 0]["pnl"].mean() if (df_t["pnl"] < 0).any() else 0
    loss_sum      = df_t[df_t["pnl"] < 0]["pnl"].sum()
    profit_factor = abs(df_t[df_t["pnl"] > 0]["pnl"].sum() / loss_sum) if loss_sum != 0 else float("inf")

    eq_series  = pd.Series(equity, index=pd.DatetimeIndex(eq_dates))
    returns    = eq_series.pct_change().dropna()
    sharpe     = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    peak       = eq_series.cummax()
    drawdown   = (eq_series - peak) / peak
    max_dd     = drawdown.min() * 100
    total_return = (equity[-1] - INITIAL_CASH) / INITIAL_CASH * 100

    long_trades  = df_t[df_t["side"] == "LONG"]
    short_trades = df_t[df_t["side"] == "SHORT"]

    print("\n" + "=" * 55)
    print("  バックテスト結果")
    print("=" * 55)
    print(f"  期間:             {eq_dates[0].date()} ～ {eq_dates[-1].date()}")
    print(f"  初期資金:         {INITIAL_CASH:>12,} 円")
    print(f"  最終資産:         {equity[-1]:>12,.0f} 円")
    print(f"  総損益:           {total_pnl:>+12,.0f} 円")
    print(f"  総リターン:       {total_return:>+11.2f} %")
    print(f"  Sharpe 比:        {sharpe:>12.3f}")
    print(f"  最大ドローダウン: {max_dd:>10.2f} %")
    print("-" * 55)
    print(f"  総取引数:         {len(df_t):>12,}")
    print(f"    うち LONG:      {len(long_trades):>12,}")
    print(f"    うち SHORT:     {len(short_trades):>12,}")
    print(f"  勝率:             {win_rate:>11.1f} %")
    print(f"  平均利益:         {avg_win:>+12,.0f} 円")
    print(f"  平均損失:         {avg_loss:>+12,.0f} 円")
    print(f"  プロフィットF:    {profit_factor:>12.3f}")
    print("=" * 55)

    print("\n上位 10 取引（損益順）:")
    cols = ["date", "ticker", "side", "entry", "exit", "shares", "pnl"]
    print(df_t.nlargest(10, "pnl")[cols].to_string(index=False))

    print("\n下位 10 取引（損益順）:")
    print(df_t.nsmallest(10, "pnl")[cols].to_string(index=False))

    return df_t, eq_series

# ── チャート出力 ──────────────────────────────────────
def plot_results(df_t, eq_series):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("バックテスト結果（hajime-f/stocktrading 戦略 × XGBoost）", fontsize=13)

    ax1 = axes[0]
    ax1.plot(eq_series.index, eq_series.values, color="steelblue", linewidth=1.5)
    ax1.axhline(INITIAL_CASH, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("資産推移")
    ax1.set_ylabel("資産（円）")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    peak = eq_series.cummax()
    dd   = (eq_series - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax2.set_title("ドローダウン")
    ax2.set_ylabel("ドローダウン（%）")
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    daily_pnl = df_t.groupby("date")["pnl"].sum()
    colors = ["steelblue" if v >= 0 else "crimson" for v in daily_pnl.values]
    ax3.bar(daily_pnl.index, daily_pnl.values, color=colors, width=1.0)
    ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.set_title("日別 P&L")
    ax3.set_ylabel("損益（円）")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = "backtest_result.png"
    plt.savefig(out, dpi=150)
    print(f"\nチャートを保存: {out}")

# ── メイン ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  hajime-f/stocktrading 戦略 バックテスト")
    print("  モデル: XGBoost（LSTM を代替）")
    print(f"  銘柄数: {len(TICKERS)}")
    print(f"  期間:   {START_DATE} ～ {END_DATE}")
    print("=" * 55)

    data = fetch_data(TICKERS)
    if not data:
        print("データ取得に失敗しました")
        exit(1)

    trades, equity, eq_dates = run_backtest(data)

    if trades:
        df_t, eq_series = print_stats(trades, equity, eq_dates)
        if df_t is not None:
            plot_results(df_t, eq_series)
    else:
        print("取引シグナルが発生しませんでした")
