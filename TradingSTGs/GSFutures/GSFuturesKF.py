import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 칼만 필터 클래스 (Core Engine)
# 외부 라이브러리 없이 행렬 연산으로 직접 구현
# ---------------------------------------------------------
class KalmanFilterReg:
    def __init__(self, delta=1e-4, R=1e-3):
        # delta: Process Noise (베타가 시간이 지남에 따라 얼마나 변할 수 있는가?)
        # R: Measurement Noise (가격 데이터 자체에 낀 노이즈가 얼마나 되는가?)
        self.n_dim = 2 # 추정할 변수 개수 (Beta, Alpha)
        
        # 상태 변수 (State): [Beta, Alpha]
        self.x = np.zeros(self.n_dim) 
        
        # 공분산 행렬 (Covariance): 초기 불확실성은 매우 크게 설정
        self.P = np.eye(self.n_dim) 
        
        # 시스템 노이즈 공분산 (Q) - 베타가 걷는 보폭
        self.Q = np.eye(self.n_dim) * delta
        
        # 관측 노이즈 분산 (R)
        self.R = R
        
    def update(self, y, x_in):
        # y: 관측된 금 가격 (Log)
        # x_in: 관측된 은 가격 (Log) -> [Silver, 1.0] 형태로 변환하여 Alpha 추정
        
        # 1. Prediction Step (예측)
        # 어제의 상태가 오늘 그대로 유지될 것이라 예측 (Random Walk 가정)
        # x_pred = x_prev
        # P_pred = P_prev + Q
        x_pred = self.x 
        P_pred = self.P + self.Q
        
        # 2. Observation Matrix (H) 구성
        H = np.array([x_in, 1.0]) # [은 가격, 1]
        
        # 3. Measurement Residual (잔차 계산)
        # 예측값 = Beta * Silver + Alpha
        y_pred = np.dot(H, x_pred)
        error = y - y_pred # 실제 금값 - 예측 금값
        
        # 4. Kalman Gain (K) 계산 - 얼마나 수정할지 결정
        # S = H * P * H.T + R (예측 오차의 분산)
        S = np.dot(np.dot(H, P_pred), H.T) + self.R
        K = np.dot(P_pred, H.T) / S 
        
        # 5. Update Step (수정)
        # 새로운 상태 = 예측 상태 + (칼만 이득 * 에러)
        self.x = x_pred + K * error
        
        # 공분산 업데이트
        self.P = P_pred - np.outer(K, H) * P_pred
        
        # 결과 반환 (Beta, Alpha, Error, Variance of Error)
        return self.x[0], self.x[1], error, S

# ---------------------------------------------------------
# 2. 데이터 준비
# ---------------------------------------------------------
tickers = ['GC=F', 'SI=F']
data = yf.download(tickers, start='2020-01-01', end='2025-12-01')['Close']
data = data.dropna()
data.columns = ['Gold', 'Silver']

# 로그 변환
df = np.log(data)

# ---------------------------------------------------------
# 3. 칼만 필터 실행 (Signal Generation)
# ---------------------------------------------------------
kf = KalmanFilterReg(delta=1e-4, R=1e-3)

betas = []
alphas = []
spreads = []     # e_t (단순 잔차)
spread_vars = [] # Var(e_t) (잔차의 분산)

for i in range(len(df)):
    curr_gold = df['Gold'].iloc[i]
    curr_silver = df['Silver'].iloc[i]
    
    # 칼만 필터 업데이트
    # 주의: 미래 데이터를 보지 않기 위해 '오늘' 데이터로 업데이트하고,
    # 그 상태값은 '내일'의 예측에 사용됨 (On-line Learning)
    beta, alpha, error, var_s = kf.update(curr_gold, curr_silver)
    
    betas.append(beta)
    alphas.append(alpha)
    spreads.append(error)
    spread_vars.append(var_s)

# 결과 DataFrame
res = pd.DataFrame(index=df.index)
res['Gold_Log'] = df['Gold']
res['Silver_Log'] = df['Silver']
res['Beta'] = betas
res['Alpha'] = alphas
res['Spread'] = spreads
# Z-Score: 잔차 / sqrt(잔차의 예측 분산) -> 표준화된 신호
res['Z_Score'] = res['Spread'] / np.sqrt(spread_vars)

# 앞부분(학습 기간) 노이즈 제거
res = res.iloc[30:] 

# ---------------------------------------------------------
# 4. 백테스팅 (Trading Loop)
# ---------------------------------------------------------
# 파라미터
entry_z = 1.0  # 진입 임계값 (표준편차 1배)
exit_z = 0.0   # 청산 임계값 (평균 회귀)
capital = 100000 
multiplier_gold = 10    # 마이크로 금
multiplier_silver = 1000 # 마이크로 은

# 실제 가격 데이터 (로그 아님)
real_price = data.loc[res.index]

equity = [capital]
pos_gold = 0
pos_silver = 0
position_record = [] # 1: Long Spread, -1: Short Spread, 0: None

for i in range(1, len(res)):
    # 전일 Z-Score 기준 판단 (Look-ahead Bias 방지)
    prev_z = res['Z_Score'].iloc[i-1]
    
    curr_p_gold = real_price['Gold'].iloc[i]
    curr_p_silver = real_price['Silver'].iloc[i]
    prev_p_gold = real_price['Gold'].iloc[i-1]
    prev_p_silver = real_price['Silver'].iloc[i-1]
    
    # 1. 포지션 결정
    target_pos = 0 # -1: Short Spread(Gold Short/Silver Long), 1: Long Spread
    
    # 현재 포지션 상태 확인
    current_state = 0
    if pos_gold > 0: current_state = 1
    elif pos_gold < 0: current_state = -1
    
    if current_state == 0: # 포지션 없을 때
        if prev_z > entry_z: target_pos = -1 # Spread가 너무 큼 -> 줄어들 것 -> Short Spread
        elif prev_z < -entry_z: target_pos = 1 # Spread가 너무 작음 -> 커질 것 -> Long Spread
    else: # 포지션 있을 때
        # Spread가 0(평균)을 지나치면 청산
        if (current_state == 1 and prev_z > exit_z) or \
           (current_state == -1 and prev_z < exit_z):
            target_pos = 0
        else:
            target_pos = current_state # 유지

    # 2. 계약 수 산출 (Dynamic Beta 적용!)
    # 오늘의 적정 헤지 비율 = 어제 마감 기준 Beta
    hedge_beta = res['Beta'].iloc[i-1]
    
    # 투자 비중 50% 가정 (레버리지 조절 가능)
    # Long Spread: Buy Gold, Sell Silver (Beta만큼)
    alloc = equity[-1] * 0.5
    
    if target_pos == 1:
        # 금 매수 금액
        amt_gold = alloc 
        # 은 매도 금액 = 금 금액 * Beta (탄력성 고려)
        amt_silver = alloc * hedge_beta 
        
        new_gold_contracts = round(amt_gold / (curr_p_gold * multiplier_gold))
        new_silver_contracts = -round(amt_silver / (curr_p_silver * multiplier_silver))
        
    elif target_pos == -1:
        # 금 매도 / 은 매수
        amt_gold = alloc
        amt_silver = alloc * hedge_beta
        
        new_gold_contracts = -round(amt_gold / (curr_p_gold * multiplier_gold))
        new_silver_contracts = round(amt_silver / (curr_p_silver * multiplier_silver))
    else:
        new_gold_contracts = 0
        new_silver_contracts = 0

    # 3. 손익 계산
    # 비용은 계약당 $2 가정
    cost = (abs(new_gold_contracts - pos_gold) + abs(new_silver_contracts - pos_silver)) * 2.0
    
    pnl_g = pos_gold * (curr_p_gold - prev_p_gold) * multiplier_gold
    pnl_s = pos_silver * (curr_p_silver - prev_p_silver) * multiplier_silver
    
    # 자본 업데이트
    new_equity = equity[-1] + pnl_g + pnl_s - cost
    if new_equity < 0: new_equity = 0 # 파산
    
    equity.append(new_equity)
    pos_gold = new_gold_contracts
    pos_silver = new_silver_contracts
    position_record.append(target_pos)

# ---------------------------------------------------------
# 5. 성과 지표 계산 및 시각화 (Updated)
# ---------------------------------------------------------
# Equity 데이터를 DataFrame에 병합
res['Equity'] = equity

# --- [성과 지표 계산] ---

# 1. 일간 수익률
res['Daily_Ret'] = res['Equity'].pct_change().fillna(0)

# 2. 총 수익률 (Total Return)
total_return = (res['Equity'].iloc[-1] - capital) / capital

# 3. MDD (Maximum Drawdown) 계산
# 전고점(Peak) 대비 현재 자산이 얼마나 빠졌는지 계산
res['Peak'] = res['Equity'].cummax()
res['Drawdown'] = (res['Equity'] / res['Peak']) - 1
mdd = res['Drawdown'].min()

# 4. 샤프 지수 (Sharpe Ratio) 계산
# 무위험 이자율(Risk-free rate)은 연 4.5%로 가정
risk_free_rate = 0.045
excess_ret = res['Daily_Ret'] - (risk_free_rate / 252)
sharpe = np.sqrt(252) * (excess_ret.mean() / excess_ret.std())

# --- [결과 텍스트 출력] ---
print("\n" + "="*40)
print(f"   [칼만 필터 전략 성과 리포트]")
print("="*40)
print(f"1. 최종 자산      : ${res['Equity'].iloc[-1]:,.2f}")
print(f"2. 총 수익률      : {total_return * 100:.2f}%")
print(f"3. 최대 낙폭(MDD) : {mdd * 100:.2f}%")
print(f"4. 샤프 지수      : {sharpe:.4f}")
print("="*40)

# --- [시각화] ---
plt.figure(figsize=(12, 12))

# 그래프 1: Dynamic Beta (베타의 변화)
plt.subplot(4, 1, 1)
plt.plot(res['Beta'], label='Dynamic Beta', color='purple')
plt.axhline(0.99, color='gray', linestyle=':', label='Static Benchmark')
plt.title('Step 1: Evolution of Beta (Hedge Ratio)')
plt.legend()
plt.grid(True)

# 그래프 2: Z-Score (진입/청산 신호)
plt.subplot(4, 1, 2)
plt.plot(res['Z_Score'], label='Spread Z-Score', color='green', alpha=0.7)
plt.axhline(entry_z, color='red', linestyle='--', label='Short Threshold')
plt.axhline(-entry_z, color='blue', linestyle='--', label='Long Threshold')
plt.axhline(0, color='black', alpha=0.5)
plt.title('Step 2: Trading Signals (Normalized Spread)')
plt.legend()
plt.grid(True)

# 그래프 3: 자산 곡선 (Equity Curve)
plt.subplot(4, 1, 3)
plt.plot(res.index, res['Equity'], label='Strategy Equity', color='blue')
plt.title(f'Step 3: Equity Curve (Return: {total_return*100:.2f}%)')
plt.ylabel('Capital ($)')
plt.legend()
plt.grid(True)

# 그래프 4: 언더워터 차트 (Drawdown) - 리스크 시각화
plt.subplot(4, 1, 4)
plt.fill_between(res.index, res['Drawdown'] * 100, 0, color='red', alpha=0.3, label='Drawdown')
plt.plot(res.index, res['Drawdown'] * 100, color='red', linewidth=1)
plt.title(f'Step 4: Underwater Plot (Max Drawdown: {mdd*100:.2f}%)')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()