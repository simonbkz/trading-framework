import warnings; warnings.filterwarnings('ignore')
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
import yfinance as yf
import pandas as pd, numpy as np

# Load all data at 1H for signal-level analysis
print('Loading 1H data...')
tickers = {
    # Our assets
    'XAUUSD': 'GC=F', 'BTCUSD': 'BTC-USD', 'XAGUSD': 'SI=F', 'ETHUSD': 'ETH-USD', 'XRPUSD': 'XRP-USD',
    # Forex pairs as potential predictors
    'DXY': 'DX-Y.NYB', 'USDJPY': 'USDJPY=X', 'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X',
    'AUDUSD': 'AUDUSD=X', 'USDCHF': 'CHF=X', 'USDCAD': 'CAD=X',
    # Risk indicators
    'VIX': '^VIX', 'SP500': '^GSPC', 'NASDAQ': '^IXIC',
    # Bonds
    'US10Y': '^TNX',
    # Commodities
    'CrudeOil': 'CL=F', 'Copper': 'HG=F',
}

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2025-01-01', interval='1h', progress=False)
        if df is not None and len(df) > 500:
            close = df['Close'].squeeze()
            data[name] = close
            print(f'  {name}: {len(close)} bars')
    except Exception as e:
        print(f'  {name}: FAILED ({e})')

# Build hourly returns
returns = pd.DataFrame({k: v.pct_change() for k, v in data.items()}).dropna(how='all')

# 1. Same-hour correlation
print()
print('=== SAME-HOUR CORRELATION WITH TRADING ASSETS ===')
our_assets = ['XAUUSD','BTCUSD','XAGUSD','ETHUSD','XRPUSD']
predictors = [k for k in returns.columns if k not in our_assets]

header = f'{"Predictor":>12s}'
for a in our_assets:
    if a in returns:
        header += f' {a:>8s}'
print(header)
print('-' * 65)

pred_corrs = {}
for p in sorted(predictors):
    if p not in returns:
        continue
    corrs = {}
    for a in our_assets:
        if a in returns:
            c = returns[p].corr(returns[a])
            if not np.isnan(c):
                corrs[a] = c
    if corrs:
        pred_corrs[p] = corrs

sorted_preds = sorted(pred_corrs.keys(), key=lambda x: max(abs(v) for v in pred_corrs[x].values()), reverse=True)
for p in sorted_preds:
    corrs = pred_corrs[p]
    max_abs = max(abs(v) for v in corrs.values())
    marker = ' ***' if max_abs > 0.3 else (' **' if max_abs > 0.2 else '')
    line = f'{p:>12s}'
    for a in our_assets:
        if a in corrs:
            line += f' {corrs[a]:>+7.3f}'
        else:
            line += f' {"N/A":>7s}'
    print(line + marker)

# 2. Lead-lag: does the predictor at t-1 to t-4 predict our asset at t?
print()
print('=== PREDICTIVE LEAD-LAG (predictor t-N vs asset t, hourly) ===')
print('Only showing correlations > 0.05')
for lag in [1, 2, 4]:
    print(f'\n--- Lag = {lag}h ---')
    for p in sorted_preds[:10]:
        if p not in returns:
            continue
        line = f'{p:>12s}'
        has_signal = False
        for a in our_assets:
            if a in returns:
                c = returns[p].shift(lag).corr(returns[a])
                if not np.isnan(c) and abs(c) > 0.05:
                    has_signal = True
                if not np.isnan(c):
                    line += f' {c:>+7.3f}'
                else:
                    line += f' {"N/A":>7s}'
        if has_signal:
            print(line)

# 3. DXY direction as a filter for gold/silver
print()
print('=== DXY DIRECTION AS FILTER (1H) ===')
if 'DXY' in data:
    dxy_ret = returns.get('DXY', pd.Series())
    # DXY trending: 4h MA direction
    dxy_close = data['DXY']
    dxy_ma4 = dxy_close.rolling(4).mean()
    dxy_trending_up = dxy_close > dxy_ma4
    dxy_trending_down = dxy_close < dxy_ma4

    common = dxy_trending_up.dropna().index.intersection(returns.index)
    for a in ['XAUUSD', 'XAGUSD', 'BTCUSD']:
        if a not in returns:
            continue
        r = returns[a].loc[common]
        up = dxy_trending_up.loc[common]
        dn = dxy_trending_down.loc[common]

        avg_when_dxy_up = r[up].mean() * 100
        avg_when_dxy_dn = r[dn].mean() * 100
        print(f'  {a}: DXY trending up -> avg return {avg_when_dxy_up:+.4f}%, DXY trending down -> {avg_when_dxy_dn:+.4f}%')

# 4. VIX level as regime filter
print()
print('=== VIX LEVEL REGIME (1H) ===')
if 'VIX' in data:
    vix = data['VIX']
    common = vix.dropna().index.intersection(returns.index)
    vix_aligned = vix.loc[common]

    q25 = vix_aligned.quantile(0.25)
    q75 = vix_aligned.quantile(0.75)
    print(f'  VIX 25th={q25:.1f}, 75th={q75:.1f}')

    vix_low = vix_aligned < q25
    vix_high = vix_aligned > q75

    for a in our_assets:
        if a not in returns:
            continue
        r = returns[a].loc[common]
        low_avg = r[vix_low].mean() * 100
        high_avg = r[vix_high].mean() * 100
        low_vol = r[vix_low].std() * 100
        high_vol = r[vix_high].std() * 100
        print(f'  {a}: Low VIX avg={low_avg:+.4f}% vol={low_vol:.3f}% | High VIX avg={high_avg:+.4f}% vol={high_vol:.3f}%')

# 5. USDJPY as risk-on/off proxy
print()
print('=== USDJPY AS RISK PROXY (1H) ===')
if 'USDJPY' in data:
    jpy_ret = returns.get('USDJPY', pd.Series())
    common = jpy_ret.dropna().index.intersection(returns.index)
    # USDJPY up = risk-on (carry trade), USDJPY down = risk-off (yen safe haven)
    jpy_up = jpy_ret.loc[common] > 0.001  # USDJPY up > 0.1%
    jpy_dn = jpy_ret.loc[common] < -0.001

    for a in our_assets:
        if a not in returns:
            continue
        r = returns[a].loc[common]
        up_avg = r[jpy_up].mean() * 100
        dn_avg = r[jpy_dn].mean() * 100
        print(f'  {a}: JPY weakening (risk-on) -> {up_avg:+.4f}% | JPY strengthening (risk-off) -> {dn_avg:+.4f}%')

# 6. Copper as global growth proxy
print()
print('=== COPPER AS GROWTH PROXY ===')
if 'Copper' in data:
    cu_ret = returns.get('Copper', pd.Series())
    common = cu_ret.dropna().index.intersection(returns.index)
    cu_up = cu_ret.loc[common] > 0.002
    cu_dn = cu_ret.loc[common] < -0.002

    for a in our_assets:
        if a not in returns:
            continue
        r = returns[a].loc[common]
        up_avg = r[cu_up].mean() * 100
        dn_avg = r[cu_dn].mean() * 100
        print(f'  {a}: Copper up -> {up_avg:+.4f}% | Copper down -> {dn_avg:+.4f}%')

# 7. Multi-factor signal: combine DXY + VIX + USDJPY into a composite risk score
print()
print('=== COMPOSITE RISK SCORE ===')
if all(k in returns for k in ['DXY', 'VIX', 'USDJPY']):
    common = returns[['DXY','USDJPY']].dropna().index.intersection(data['VIX'].dropna().index).intersection(returns.index)

    # Risk-on score: USDJPY up + DXY down + VIX down = bullish for crypto/metals
    dxy_z = -returns['DXY'].loc[common].rolling(4).mean() / returns['DXY'].loc[common].rolling(20).std()
    jpy_z = returns['USDJPY'].loc[common].rolling(4).mean() / returns['USDJPY'].loc[common].rolling(20).std()
    vix_val = data['VIX'].loc[common]
    vix_z = -(vix_val - vix_val.rolling(20).mean()) / vix_val.rolling(20).std()

    risk_score = (dxy_z.fillna(0) + jpy_z.fillna(0) + vix_z.fillna(0)) / 3
    risk_score = risk_score.dropna()

    # Quintile analysis
    for a in our_assets:
        if a not in returns:
            continue
        r = returns[a]
        common2 = risk_score.index.intersection(r.index)
        rs = risk_score.loc[common2]
        ret = r.loc[common2]

        q20 = rs.quantile(0.2)
        q80 = rs.quantile(0.8)

        bearish_avg = ret[rs < q20].mean() * 100
        neutral_avg = ret[(rs >= q20) & (rs <= q80)].mean() * 100
        bullish_avg = ret[rs > q80].mean() * 100
        print(f'  {a}: Risk-off(bot20%) -> {bearish_avg:+.4f}% | Neutral -> {neutral_avg:+.4f}% | Risk-on(top20%) -> {bullish_avg:+.4f}%')
