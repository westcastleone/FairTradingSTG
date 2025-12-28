import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 가져오기 (Signal: 선물 / Execution: ETF)
tickers = ['GC=F', 'SI=F', 'GLD', 'SLV']
start_date = '2015-01-01'
end_date = '2024-12-31'

# 'Close' 기준
data = yf.download(tickers, start=start_date, end=end_date)['Close']
data = data.dropna()

# 2. '진짜' 금/은 비율(True GSR) 계산
data['True_GSR'] = data['GC=F'] / data['SI=F']

# 3. 기술적 지표 생성 (Z-Score)
window = 60
data['GSR_Mean'] = data['True_GSR'].rolling(window=window).mean()
data['GSR_Std'] = data['True_GSR'].rolling(window=window).std()
data['Z_Score'] = (data['True_GSR'] - data['GSR_Mean']) / data['GSR_Std']

# 4. 포지션 로직
entry_threshold = 2.0  
exit_threshold = 0.5   

pos_gld_list = []
pos_slv_list = []
curr_pos = 0 

# 초기값 채우기
for _ in range(window):
    pos_gld_list.append(0)
    pos_slv_list.append(0)

# 시뮬레이션
for i in range(window, len(data)):
    z = data['Z_Score'].iloc[i]
    
    if z > entry_threshold:
        curr_pos = -1 # Short Gold / Long Silver
    elif z < -entry_threshold:
        curr_pos = 1  # Long Gold / Short Silver
    elif abs(z) < exit_threshold:
        curr_pos = 0
            
    # Dollar Neutral (50:50)
    if curr_pos == 1:
        pos_gld_list.append(0.5)
        pos_slv_list.append(-0.5)
    elif curr_pos == -1:
        pos_gld_list.append(-0.5)
        pos_slv_list.append(0.5)
    else:
        pos_gld_list.append(0)
        pos_slv_list.append(0)

data['pos_gld'] = pos_gld_list
data['pos_slv'] = pos_slv_list

# ============================================================
# [NEW] 비용 반영 섹션
# ============================================================

# 설정값 (보수적으로 잡음)
transaction_cost_pct = 0.002  # 거래당 0.2% (수수료 + 슬리피지)
short_interest_rate_annual = 0.03 # 공매도 이자 연 3%

# 1. 턴오버(Turnover) 계산: 포지션이 얼마나 바뀌었는가?
# abs(오늘 포지션 - 어제 포지션) = 변동폭
# 예: 0 -> 0.5 (0.5 매수), 0.5 -> -0.5 (1.0 매도)
data['turnover_gld'] = data['pos_gld'].diff().abs().fillna(0)
data['turnover_slv'] = data['pos_slv'].diff().abs().fillna(0)
data['total_turnover'] = data['turnover_gld'] + data['turnover_slv']

# 2. 거래 비용 차감 (Transaction Cost)
data['cost_transaction'] = data['total_turnover'] * transaction_cost_pct

# 3. 공매도 이자 비용 차감 (Short Borrowing Cost)
# 포지션이 음수(-)인 경우에만 비용 발생. (주말 포함하여 365일로 나눔)
# 어제 포지션을 기준으로 오늘 이자가 나감
data['cost_short_gld'] = np.where(data['pos_gld'].shift(1) < 0, 
                                  abs(data['pos_gld'].shift(1)) * (short_interest_rate_annual / 365), 
                                  0)
data['cost_short_slv'] = np.where(data['pos_slv'].shift(1) < 0, 
                                  abs(data['pos_slv'].shift(1)) * (short_interest_rate_annual / 365), 
                                  0)
data['total_short_cost'] = data['cost_short_gld'] + data['cost_short_slv']


# 5. 순수익률(Net Return) 계산
initial_capital = 100000

data['ret_gld'] = data['GLD'].pct_change()
data['ret_slv'] = data['SLV'].pct_change()

# (1) 총 수익 (Gross)
data['gross_ret'] = (data['pos_gld'].shift(1) * data['ret_gld']) + \
                    (data['pos_slv'].shift(1) * data['ret_slv'])
data['gross_ret'] = data['gross_ret'].fillna(0)

# (2) 순 수익 (Net) = 총 수익 - 거래비용 - 공매도이자
data['net_ret'] = data['gross_ret'] - data['cost_transaction'] - data['total_short_cost']

# 자산 곡선
data['equity_gross'] = (1 + data['gross_ret']).cumprod() * initial_capital
data['equity_net'] = (1 + data['net_ret']).cumprod() * initial_capital

# 6. 성과 지표 함수
def calculate_metrics(daily_returns, equity_curve):
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = drawdown.min()
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    information_ratio = daily_returns.mean() / daily_returns.std()
    return total_return, mdd, sharpe_ratio, information_ratio

# Gross vs Net 비교
g_ret, g_mdd, g_sharpe, g_ir = calculate_metrics(data['gross_ret'], data['equity_gross'])
n_ret, n_mdd, n_sharpe, n_ir = calculate_metrics(data['net_ret'], data['equity_net'])

# 7. 결과 출력
print(f"--- Reality Check: Gross vs Net (Capital: ${initial_capital:,}) ---")
print(f"[설정] 거래비용(수수료+슬리피지): {transaction_cost_pct*100}%, 공매도이자: {short_interest_rate_annual*100}%/년")
print("-" * 60)
print(f"{'Metric':<20} | {'Gross (이론)':<15} | {'Net (실제)':<15}")
print("-" * 60)
print(f"{'Total Return':<20} | {g_ret * 100:.2f}%          | {n_ret * 100:.2f}%")
print(f"{'MDD':<20} | {g_mdd * 100:.2f}%          | {n_mdd * 100:.2f}%")
print(f"{'Sharpe Ratio':<20} | {g_sharpe:.4f}           | {n_sharpe:.4f}")
print(f"{'Final Equity':<20} | ${data['equity_gross'].iloc[-1]:,.0f}      | ${data['equity_net'].iloc[-1]:,.0f}")
print("-" * 60)

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['equity_gross'], label='Gross Return (No Cost)', linestyle='--')
plt.plot(data.index, data['equity_net'], label='Net Return (With Cost)', linewidth=2)
plt.title('Impact of Costs: Gross vs Net Return')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.legend()
plt.show()