import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import odr
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. 데이터 가져오기
tickers = ['GC=F', 'SI=F']
start_date = '2020-01-01'
end_date = '2025-12-01'

print("데이터 다운로드 중...")
raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
raw_data = raw_data.dropna()
raw_data.columns = ['Gold', 'Silver']

# 로그 변환 (필수)
data = np.log(raw_data)

# 2. TLS (Total Least Squares) 회귀분석 수행
# 목적: 금과 은 모두 오차(노이즈)가 있다고 가정하고, 가장 적합한 베타(기울기)를 찾음

def f(B, x):
    return B[0]*x + B[1]

# 데이터 준비 (X: 은, Y: 금)
x = data['Silver']
y = data['Gold']

# ODR 모델 설정
linear = odr.Model(f)
mydata = odr.Data(x, y) 
# beta0: 초기 추정값 [기울기, 절편]
myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()

beta_tls = myoutput.beta[0] # 기울기 (Hedge Ratio)
alpha_tls = myoutput.beta[1] # 절편

print(f"\n[TLS 결과] 산출된 헤지 비율 (Beta): {beta_tls:.4f}")
print(f"[TLS 결과] 산출된 상수항 (Alpha): {alpha_tls:.4f}")

# 3. TLS 기반 스프레드(잔차) 계산
# Spread = Log(Gold) - Beta * Log(Silver) - Alpha
# 이 스프레드가 정상성(Stationary)을 가져야 공적분 관계가 성립함
data['Spread_TLS'] = data['Gold'] - (beta_tls * data['Silver']) - alpha_tls

# 4. 공적분 검정 (Spread에 대한 ADF 검정)
# TLS로 만든 '목줄(Spread)'이 팽팽하게 유지되는지 검증하는 단계

print("\nStep: TLS 스프레드에 대한 정상성(ADF) 검정")
adf_result = adfuller(data['Spread_TLS'])

p_value = adf_result[1]
t_stat = adf_result[0]
crit_values = adf_result[4]

print(f"ADF p-value: {p_value:.5f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"Critical Values: {crit_values}")

print("-" * 50)
if p_value < 0.05:
    print(">>> [결론] p-value < 0.05 : TLS 스프레드는 '정상 시계열'입니다.")
    print(">>> 금과 은은 TLS 모델 하에서 '공적분 관계'에 있습니다. (Trading OK)")
else:
    print(">>> [결론] p-value > 0.05 : TLS 스프레드는 '랜덤 워크'입니다.")
    print(">>> 공적분 관계가 깨졌거나, TLS 모델이 적합하지 않을 수 있습니다. (Trading Caution)")
print("-" * 50)

# 5. 시각화
plt.figure(figsize=(12, 8))

# 상단: 로그 가격 추이
plt.subplot(2, 1, 1)
plt.plot(data['Gold'], label='Log Gold')
plt.plot(data['Silver'], label='Log Silver')
plt.title('Log Prices of Gold and Silver Futures')
plt.legend()
plt.grid(True)

# 하단: TLS 스프레드 (이게 평균으로 회귀해야 함)
plt.subplot(2, 1, 2)
plt.plot(data['Spread_TLS'], color='green', label=f'TLS Spread (Beta={beta_tls:.2f})')
plt.axhline(0, color='red', linestyle='--', label='Mean (0)')
plt.axhline(data['Spread_TLS'].std()*2, color='gray', linestyle=':', label='+2 Sigma')
plt.axhline(-data['Spread_TLS'].std()*2, color='gray', linestyle=':', label='-2 Sigma')
plt.title('Cointegration Spread based on TLS')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()