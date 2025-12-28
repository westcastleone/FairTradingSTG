import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 가져오기 (GLD: 금 ETF, SLV: 은 ETF)
tickers = ['GLD', 'SLV']
start_date = '2015-01-01'
end_date = '2025-12-01'

data = yf.download(tickers, start=start_date, end=end_date)['Close']
data = data.dropna()

# 2. 지표 계산: 금/은 비율 (Gold/Silver Ratio)
data['GSR'] = data['GLD'] / data['SLV']

# 3. 신호 생성을 위한 기술적 지표 (Rolling Window 적용)
window = 60  # 과거 60일(약 3개월) 데이터를 기준
data['GSR_Mean'] = data['GSR'].rolling(window=window).mean()
data['GSR_Std'] = data['GSR'].rolling(window=window).std()

# Z-Score: 현재 비율이 평균에서 표준편차의 몇 배만큼 떨어져 있는가?
data['Z_Score'] = (data['GSR'] - data['GSR_Mean']) / data['GSR_Std']

# 4. 포지션 로직 (Long/Short)
# 자본 배분: 신호 발생 시 자산의 50%는 매수, 50%는 매도 (Dollar Neutral에 근접)
entry_threshold = 1.5  # 진입 임계값 (표준편차 1.5배)
exit_threshold = 0.5   # 청산 임계값 (표준편차 0.5배 이내)

data['pos_gld'] = 0
data['pos_slv'] = 0

curr_pos = 0 # 0: None, 1: Long GLD / Short SLV (Ratio Low), -1: Short GLD / Long SLV (Ratio High)

pos_gld_list = []
pos_slv_list = []

# 반복문을 통해 시점별 판단 (Look-ahead bias 제거)
# rolling 데이터가 NaN인 초기 구간은 제외하고 계산
valid_idx = data.index[window:] 
# 초기값 채우기
for _ in range(window):
    pos_gld_list.append(0)
    pos_slv_list.append(0)

for i in range(window, len(data)):
    z = data['Z_Score'].iloc[i]
    
    # 로직: 비율이 너무 높으면(Gold 고평가) -> Gold 매도, Silver 매수
    if z > entry_threshold:
        curr_pos = -1 # Short GLD, Long SLV
    # 로직: 비율이 너무 낮으면(Gold 저평가) -> Gold 매수, Silver 매도
    elif z < -entry_threshold:
        curr_pos = 1  # Long GLD, Short SLV
    # 로직: 평균으로 돌아오면 포지션 청산
    elif abs(z) < exit_threshold:
        curr_pos = 0
        
    pos_gld_list.append(curr_pos * 0.5) # 자본의 50% 투입
    pos_slv_list.append(-curr_pos * 0.5) # 자본의 50% 투입 (Short은 -부호)

data['pos_gld'] = pos_gld_list
data['pos_slv'] = pos_slv_list

# 5. 수익률 계산
initial_capital = 100000

# 개별 자산의 일간 수익률
data['ret_gld'] = data['GLD'].pct_change()
data['ret_slv'] = data['SLV'].pct_change()

# 전략 수익률 = (전날 GLD 포지션 * 오늘 GLD 등락) + (전날 SLV 포지션 * 오늘 SLV 등락)
data['strategy_ret'] = (data['pos_gld'].shift(1) * data['ret_gld']) + \
                       (data['pos_slv'].shift(1) * data['ret_slv'])

data['strategy_ret'] = data['strategy_ret'].fillna(0)
data['equity_curve'] = (1 + data['strategy_ret']).cumprod() * initial_capital

# 6. 성과 지표 계산 함수 (재사용)
def calculate_metrics(daily_returns, equity_curve):
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = drawdown.min()
    
    # 샤프 지수 (무위험 수익률 0 가정, 연율화)
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    
    # 정보 비율 (벤치마크 0 가정)
    information_ratio = daily_returns.mean() / daily_returns.std()

    return total_return, mdd, sharpe_ratio, information_ratio

total_ret, mdd, sharpe, ir = calculate_metrics(data['strategy_ret'], data['equity_curve'])

# 7. 결과 출력
print(f"--- Gold/Silver Ratio Strategy Results (Capital: ${initial_capital:,}) ---")
print(f"1. 총 수익률 (Total Return): {total_ret * 100:.2f}%")
print(f"2. 최대 낙폭 (MDD): {mdd * 100:.2f}%")
print(f"3. 샤프 지수 (Sharpe Ratio): {sharpe:.4f}")
print(f"4. 정보 비율 (Information Ratio): {ir:.4f}")
print(f"최종 자산: ${data['equity_curve'].iloc[-1]:,.2f}")

# 시각화 1: 수익 곡선
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['equity_curve'], label='GSR Strategy Equity')
plt.title('Gold/Silver Ratio Trading Strategy Performance')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.legend()
plt.show()

# 시각화 2: 금은비와 매매 구간
plt.figure(figsize=(12, 4))
plt.plot(data.index, data['GSR'], label='Gold/Silver Ratio', color='gold')
plt.plot(data.index, data['GSR_Mean'], label='60-day Moving Avg', linestyle='--', color='black')
plt.title('Gold/Silver Ratio Trend')
plt.legend()
plt.show()