import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 가져오기
tickers = ['GC=F', 'SI=F'] 
start_date = '2020-01-01'
end_date = '2024-12-31'

data = yf.download(tickers, start=start_date, end=end_date)['Close']
data = data.dropna()
data.columns = ['Gold', 'Silver']

# 2. 파라미터 설정
capital = 100000  # 초기 자본 $100,000
multiplier_gold = 10    # 마이크로 금 (10 oz)
multiplier_silver = 1000 # 마이크로 은 (1,000 oz)
transaction_cost_per_contract = 2.0 

# [중요] 레버리지 설정
# 개시 증거금률 15% 가정 -> 내 돈으로 1/0.15 = 약 6.6배 규모 운영
initial_margin_rate = 0.15 

# 3. 지표 계산
data['GSR'] = data['Gold'] / data['Silver']

window = 60
data['GSR_Mean'] = data['GSR'].rolling(window=window).mean()
data['GSR_Std'] = data['GSR'].rolling(window=window).std()
data['Z_Score'] = (data['GSR'] - data['GSR_Mean']) / data['GSR_Std']

# 4. 백테스팅 루프
entry_threshold = 2.0
exit_threshold = 0.5

equity_curve = [capital]
num_gold_contracts = 0
num_silver_contracts = 0
position_history = []
is_bankrupt = False # 파산 여부

for i in range(1, len(data)):
    # 이미 파산했으면 0으로 유지하고 스킵
    if is_bankrupt:
        equity_curve.append(0)
        position_history.append(0)
        continue

    # 전일 지표 및 가격
    prev_z = data['Z_Score'].iloc[i-1]
    curr_gold_price = data['Gold'].iloc[i]
    curr_silver_price = data['Silver'].iloc[i]
    prev_gold_price = data['Gold'].iloc[i-1]
    prev_silver_price = data['Silver'].iloc[i-1]
    
    # --- 1. 포지션 결정 ---
    target_pos = 0 
    
    # 포지션 유지 여부 확인
    current_signal_state = 0
    if num_gold_contracts > 0: current_signal_state = 1
    elif num_gold_contracts < 0: current_signal_state = -1
    
    if current_signal_state == 0: 
        if prev_z > entry_threshold: target_pos = -1 
        elif prev_z < -entry_threshold: target_pos = 1 
    else: 
        if abs(prev_z) < exit_threshold: target_pos = 0 
        else: target_pos = current_signal_state

    # --- 2. 계약 수 계산 (풀 레버리지 적용) ---
    current_equity = equity_curve[-1]
    
    # 운용 가능 총액 (Buying Power) = 현재 자본 / 증거금률
    # 예: 10만불 / 0.15 = 66만불 운용
    total_buying_power = current_equity / initial_margin_rate
    
    # 50:50 배분
    target_exposure = total_buying_power * 0.5
    
    # 자본이 너무 적으면 거래 불가 (파산 방어)
    if current_equity < 5000: 
        target_gold_contracts = 0
        target_silver_contracts = 0
    elif target_pos == 1: 
        target_gold_contracts = round(target_exposure / (curr_gold_price * multiplier_gold))
        target_silver_contracts = -round(target_exposure / (curr_silver_price * multiplier_silver))
    elif target_pos == -1: 
        target_gold_contracts = -round(target_exposure / (curr_gold_price * multiplier_gold))
        target_silver_contracts = round(target_exposure / (curr_silver_price * multiplier_silver))
    else:
        target_gold_contracts = 0
        target_silver_contracts = 0
        
    # --- 3. 거래 비용 및 PnL ---
    diff_gold = abs(target_gold_contracts - num_gold_contracts)
    diff_silver = abs(target_silver_contracts - num_silver_contracts)
    daily_cost = (diff_gold + diff_silver) * transaction_cost_per_contract
    
    # 포지션 업데이트
    num_gold_contracts = target_gold_contracts
    num_silver_contracts = target_silver_contracts
    
    pnl_gold = (curr_gold_price - prev_gold_price) * multiplier_gold * num_gold_contracts
    pnl_silver = (curr_silver_price - prev_silver_price) * multiplier_silver * num_silver_contracts
    
    # 이자 수익 로직 삭제됨 (0)
    
    # 자본 업데이트
    total_daily_pnl = pnl_gold + pnl_silver - daily_cost
    new_equity = current_equity + total_daily_pnl
    
    # 파산 체크 (자본금이 0 이하로 떨어지면)
    if new_equity <= 0:
        new_equity = 0
        is_bankrupt = True
        print(f"!!! BANKRUPTCY ALERT !!! Date: {data.index[i].date()}")

    equity_curve.append(new_equity)
    position_history.append(target_pos)

# 결과 정리
result_df = data.copy()
result_df = result_df.iloc[0:len(equity_curve)]
result_df['Equity'] = equity_curve
result_df['Returns'] = result_df['Equity'].pct_change().fillna(0)

# 성과 지표 함수
def calculate_metrics(daily_returns, equity_curve):
    if equity_curve.iloc[-1] <= 0: return -1.0, -1.0, 0.0 # 파산 시 처리
    
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = drawdown.min()
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    return total_return, mdd, sharpe_ratio

total_ret, mdd, sharpe = calculate_metrics(result_df['Returns'], result_df['Equity'])

# 출력
print(f"--- Full Leverage Futures Trading (No Cash Interest) ---")
print(f"Leverage: approx {1/initial_margin_rate:.1f}x (Margin Rate: {initial_margin_rate*100}%)")
print("-" * 50)
print(f"1. Total Return: {total_ret * 100:.2f}%")
print(f"2. MDD: {mdd * 100:.2f}%")
print(f"3. Sharpe Ratio: {sharpe:.4f}")
print(f"Final Equity: ${result_df['Equity'].iloc[-1]:,.2f}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(result_df.index, result_df['Equity'], label='High Leverage Equity', color='red')
plt.title('High Risk High Return: Full Leverage Strategy')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.legend()
plt.show()