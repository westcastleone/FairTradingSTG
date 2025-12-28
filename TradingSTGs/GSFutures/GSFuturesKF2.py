import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

# ---------------------------------------------------------
# 1. 설정 및 초기화
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 2. 칼만 필터 엔진
# ---------------------------------------------------------
class KalmanFilterReg:
    def __init__(self, delta=1e-5, R=1e-3):
        self.n_states = 2 
        self.trans_cov = delta / (1 - delta) * np.eye(self.n_states) 
        self.obs_mat = np.expand_dims(np.vstack([[0], [1]]), axis=0) 
        self.state_mean = np.zeros(self.n_states) 
        self.state_cov = np.ones((self.n_states, self.n_states)) 
        self.R = R 

    def update(self, x, y):
        H = np.array([x, 1.0])
        pred_mean = self.state_mean
        pred_cov = self.state_cov + self.trans_cov
        y_pred = H.dot(pred_mean)
        error = y - y_pred
        S = H.dot(pred_cov).dot(H.T) + self.R 
        K = pred_cov.dot(H.T) / S
        new_state_mean = pred_mean + K * error
        new_state_cov = pred_cov - np.outer(K, H).dot(pred_cov)
        self.state_mean = new_state_mean
        self.state_cov = new_state_cov
        return new_state_mean[0], error, S

# ---------------------------------------------------------
# 3. 데이터 다운로드 및 전처리
# ---------------------------------------------------------
print(">>> 데이터 다운로드 및 분석 시작...")
tickers = ['GC=F', 'SI=F']
data = yf.download(tickers, start='2020-01-01', progress=False)

if isinstance(data.columns, pd.MultiIndex):
    df = pd.DataFrame()
    try:
        df['Gold'] = data['Adj Close']['GC=F']
        df['Silver'] = data['Adj Close']['SI=F']
    except:
        df['Gold'] = data['Close']['GC=F']
        df['Silver'] = data['Close']['SI=F']
else:
    df = data[['GC=F', 'SI=F']].copy()
    df.columns = ['Gold', 'Silver']

df = df.fillna(method='ffill').dropna()
df['Log_Gold'] = np.log(df['Gold'])
df['Log_Silver'] = np.log(df['Silver'])

# ---------------------------------------------------------
# 4. 신호 생성 (Kalman Filter)
# ---------------------------------------------------------
kf = KalmanFilterReg(delta=1e-5, R=1e-3)
betas, errors, variances = [], [], []

for i in range(len(df)):
    b, e, v = kf.update(df['Log_Silver'].iloc[i], df['Log_Gold'].iloc[i])
    betas.append(b)
    errors.append(e)
    variances.append(v)

df['Beta'] = betas
df['Z_Score'] = np.array(errors) / np.sqrt(np.array(variances))

# 시장 필터링 (Regime Filter): 5% ~ 95%
lookback = 252
df['Beta_Lower'] = df['Beta'].rolling(window=lookback).quantile(0.05)
df['Beta_Upper'] = df['Beta'].rolling(window=lookback).quantile(0.95)
df = df.dropna()

# ---------------------------------------------------------
# 5. 백테스팅 (Modified Sniper Strategy)
# ---------------------------------------------------------
capital = 100000        # 초기 자본
leverage = 3.0          # 레버리지 3배

# [수정된 파라미터]
entry_z = 1.0           # 진입: 1.0 시그마 (요청사항 반영)
exit_z = 0.0            # 청산: 0 (평균 회귀)
stop_loss_z = 2.5       # 손절: 2.5 시그마

mult_g = 10             
mult_s = 1000           
commission = 2.0        

equity_curve = [capital]
pos_g = 0
pos_s = 0

for i in range(1, len(df)):
    # 데이터 (Look-ahead Bias 방지)
    curr_z = df['Z_Score'].iloc[i-1]
    curr_beta = df['Beta'].iloc[i-1]
    beta_low = df['Beta_Lower'].iloc[i-1]
    beta_high = df['Beta_Upper'].iloc[i-1]
    
    p_g = df['Gold'].iloc[i]
    p_s = df['Silver'].iloc[i]
    prev_p_g = df['Gold'].iloc[i-1]
    prev_p_s = df['Silver'].iloc[i-1]
    
    # 전략 로직
    target_pos = 0 
    
    curr_state = 0
    if pos_g > 0: curr_state = 1
    elif pos_g < 0: curr_state = -1
    
    # 1. 손절매 (Stop Loss) - 최우선
    if abs(curr_z) > stop_loss_z:
        target_pos = 0 
    
    # 2. 포지션 진입/청산
    elif curr_state == 0:
        # 시장 필터 통과 시에만 진입
        is_regime_normal = beta_low < curr_beta < beta_high
        
        if is_regime_normal:
            if curr_z > entry_z: target_pos = -1   # Short Spread
            elif curr_z < -entry_z: target_pos = 1 # Long Spread
    else:
        # 이익 실현 (평균 회귀)
        if (curr_state == 1 and curr_z > exit_z) or \
           (curr_state == -1 and curr_z < exit_z):
            target_pos = 0
        else:
            target_pos = curr_state 

    # 포지션 사이징
    curr_equity = equity_curve[-1]
    
    if target_pos != 0:
        buying_power = curr_equity * leverage
        alloc_g = buying_power / 2
        
        tgt_qty_g = round(alloc_g / (p_g * mult_g))
        
        if target_pos == 1:
            tgt_g = tgt_qty_g
            tgt_s = -round((alloc_g * curr_beta) / (p_s * mult_s))
        else:
            tgt_g = -tgt_qty_g
            tgt_s = round((alloc_g * curr_beta) / (p_s * mult_s))
    else:
        tgt_g = 0
        tgt_s = 0
        
    cost = (abs(tgt_g - pos_g) + abs(tgt_s - pos_s)) * commission
    pnl = (pos_g * (p_g - prev_p_g) * mult_g) + (pos_s * (p_s - prev_p_s) * mult_s)
    
    new_equity = curr_equity + pnl - cost
    if new_equity < 0: new_equity = 0
    
    equity_curve.append(new_equity)
    pos_g = tgt_g
    pos_s = tgt_s

df = df.iloc[0:len(equity_curve)]
df['Equity'] = equity_curve

# ---------------------------------------------------------
# 6. 성과 지표
# ---------------------------------------------------------
df['Daily_Ret'] = df['Equity'].pct_change().fillna(0)
total_return = (df['Equity'].iloc[-1] - capital) / capital
days = (df.index[-1] - df.index[0]).days
cagr = (df['Equity'].iloc[-1] / capital) ** (365/days) - 1

df['Peak'] = df['Equity'].cummax()
df['Drawdown'] = (df['Equity'] / df['Peak']) - 1
mdd = df['Drawdown'].min()

rf = 0.045
excess_ret = df['Daily_Ret'] - (rf/252)
sharpe = np.sqrt(252) * (excess_ret.mean() / excess_ret.std())

# ---------------------------------------------------------
# 7. 시각화
# ---------------------------------------------------------
print("\n" + "="*50)
print(f"   [전략 리포트: 진입(1.0σ), 손절(2.5σ)]")
print("="*50)
print(f"1. 최종 자산      : ${df['Equity'].iloc[-1]:,.2f}")
print(f"2. 총 수익률      : {total_return*100:.2f}%")
print(f"3. 연평균 수익률  : {cagr*100:.2f}%")
print(f"4. 최대 낙폭(MDD) : {mdd*100:.2f}%")
print(f"5. 샤프 지수      : {sharpe:.4f}")
print("="*50)

plt.figure(figsize=(12, 10))

# 1. 시장 필터
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Beta'], color='purple', label='Beta')
plt.fill_between(df.index, df['Beta_Lower'], df['Beta_Upper'], color='gray', alpha=0.1, label='Normal Range')
plt.title('Step 1: Market Regime Filter')
plt.legend()
plt.grid(True)

# 2. 신호 (진입 라인이 1.0으로 좁혀짐)
plt.subplot(3, 1, 2)
plt.plot(df.index, df['Z_Score'], label='Z-Score', color='green', linewidth=1)
plt.axhline(stop_loss_z, color='red', linestyle='-', linewidth=2, label='Stop (2.5)')
plt.axhline(-stop_loss_z, color='red', linestyle='-', linewidth=2)
plt.axhline(entry_z, color='blue', linestyle=':', label='Entry (1.0)') # 변경됨
plt.axhline(-entry_z, color='blue', linestyle=':')
plt.title('Step 2: Signals (Entry=1.0, Stop=2.5)')
plt.legend()
plt.grid(True)

# 3. 자산 곡선
plt.subplot(3, 1, 3)
plt.plot(df.index, df['Equity'], label='Equity', color='blue')
plt.title(f'Step 3: Equity Curve (Sharpe: {sharpe:.2f})')
plt.ylabel('Capital ($)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()