"""
generate_data.py
experiments/v2_monthly_2000.py의 로직을 사용하여
최신 월말 기준 매수 종목 리스트 + 백테스트 요약을 JSON으로 출력
→ 웹사이트(index.html)에서 로드
"""
import pandas as pd
import numpy as np
import json, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # 원본 CSV 위치 (data/README.md 참고)
RESULTS_DIR = ROOT / "results"    # 결과물 저장 위치
DOCS_DIR = ROOT / "docs"          # 대시보드 (GitHub Pages)
RESULTS_DIR.mkdir(exist_ok=True)
warnings.filterwarnings('ignore')

print("=" * 60)
print("  웹사이트용 데이터 생성")
print("=" * 60)
t0 = time.time()

# ═══════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════
print("데이터 로딩 중...", flush=True)

prices_raw = pd.read_csv(DATA_DIR / "수정주가_거래대금필터_code.csv", index_col='Date', parse_dates=True, low_memory=False)
prices = prices_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))

total_raw = pd.read_csv(DATA_DIR / "현금배당포함.csv", index_col='코드명', parse_dates=True, low_memory=False)
total_adj = total_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))

mcap_raw = pd.read_csv(DATA_DIR / "시가총액.csv", index_col='코드명', parse_dates=True, low_memory=False)
mcap = mcap_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.replace('"', '').str.strip(), errors='coerce'))

tv_raw = pd.read_csv(DATA_DIR / "거래대금.csv", index_col='Date', parse_dates=True, low_memory=False)
tv = tv_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))

tv60_raw = pd.read_csv(DATA_DIR / "60_거래대금.csv", index_col='Date', parse_dates=True, low_memory=False)
tv60 = tv60_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.strip(), errors='coerce'))

factors_raw = pd.read_csv(DATA_DIR / "4팩터.csv", index_col='Date', parse_dates=True)
for col in factors_raw.columns:
    factors_raw[col] = pd.to_numeric(factors_raw[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

# 코드↔종목명 매핑
code_cols = prices.columns
name_cols = total_adj.columns
code_to_name = dict(zip(code_cols, name_cols))
name_to_code = dict(zip(name_cols, code_cols))

total_adj.columns = [name_to_code.get(c, c) for c in total_adj.columns]
mcap.columns = [name_to_code.get(c, c) for c in mcap.columns]

# 월별 데이터
prices_monthly = prices.resample('ME').last()
total_adj_monthly = total_adj.resample('ME').last()
mcap_monthly = mcap.resample('ME').last()
tv60_monthly = tv60.resample('ME').last()
returns_daily = prices.pct_change()
returns_monthly = total_adj_monthly.pct_change(fill_method=None)

# 윈저라이징
returns_monthly_wins = returns_monthly.copy()
for col in returns_monthly_wins.columns:
    s = returns_monthly_wins[col].dropna()
    if len(s) > 10:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        returns_monthly_wins[col] = returns_monthly_wins[col].clip(lo, hi)

# Amihud
daily_ret_abs = returns_daily.abs()
daily_illiq = daily_ret_abs / tv
amihud_monthly = daily_illiq.resample('ME').mean()

# 4팩터
factors_monthly = factors_raw.resample('ME').last()
kospi_factor_ret = factors_monthly['KOSPI'].pct_change() if 'KOSPI' in factors_monthly.columns else factors_monthly.iloc[:, 0].pct_change()
hml_ret = factors_monthly['HML'].pct_change() if 'HML' in factors_monthly.columns else factors_monthly.iloc[:, 1].pct_change()
smb_ret = factors_monthly['SMB'].pct_change() if 'SMB' in factors_monthly.columns else factors_monthly.iloc[:, 2].pct_change()
mom_ret = factors_monthly['MOM'].pct_change() if 'MOM' in factors_monthly.columns else factors_monthly.iloc[:, 3].pct_change()
cd91 = factors_monthly['CD91'] if 'CD91' in factors_monthly.columns else factors_monthly.iloc[:, 4]
rf_monthly = cd91 / 100 / 12
mkt_factor = kospi_factor_ret - rf_monthly

spac_set = set(c for c in prices.columns if '스팩' in str(c) or 'SPAC' in str(c))

month_ends = returns_monthly_wins.loc['2000-01-31':'2025-12-31'].index
LOOKBACK = 60; N_QUINTILES = 5
HIGH_COST = 0.008; LOW_COST = 0.003; ILLIQ_THRESHOLD = 0.80

print(f"로딩 완료: {time.time()-t0:.1f}초")

# ═══════════════════════════════════════
# 2. 함수 (final_strategy_v2.py와 동일)
# ═══════════════════════════════════════
def compute_cap_scaled_centrality(adj, mcap_vals, beta, max_iter=300, tol=1e-8):
    n=len(adj);x=np.ones(n)/n;mc=mcap_vals.copy()
    mc[mc<=0]=mc[mc>0].min() if (mc>0).any() else 1.0
    penalty=1.0/(mc**beta);penalty/=penalty.max()
    for _ in range(max_iter):
        x_new=penalty*(adj@x);norm=np.linalg.norm(x_new)
        if norm==0:return np.ones(n)/n
        x_new/=norm
        if np.linalg.norm(x_new-x)<tol:break
        x=x_new
    return np.abs(x_new)

def compute_eigenvector_centrality(adj, max_iter=300, tol=1e-8):
    n=len(adj);x=np.ones(n)/n
    for _ in range(max_iter):
        x_new=adj@x;norm=np.linalg.norm(x_new)
        if norm==0:return np.ones(n)/n
        x_new/=norm
        if np.linalg.norm(x_new-x)<tol:break
        x=x_new
    return np.abs(x_new)

def equal_weight_correlation(X):
    T,N=X.shape;Xc=X-X.mean(axis=0);S=(Xc.T@Xc)/T
    std=np.sqrt(np.diag(S));std[std==0]=1
    C=S/np.outer(std,std);np.fill_diagonal(C,1.0);np.clip(C,-1,1,out=C)
    return C

def marchenko_pastur_clipping_full(corr, T):
    N=len(corr);q=N/T;lmax=(1+np.sqrt(q))**2
    ev,evec=np.linalg.eigh(corr)
    idx=np.argsort(ev)[::-1];ev=ev[idx];evec=evec[:,idx]
    ns=max(np.sum(ev>lmax),1);cev=ev.copy()
    if N>ns:cev[ns:]=ev[ns:].mean()
    cev*=N/cev.sum()
    Cc=evec@np.diag(cev)@evec.T
    d=np.sqrt(np.diag(Cc));d[d==0]=1
    Cc/=np.outer(d,d);np.fill_diagonal(Cc,1.0);np.clip(Cc,-1,1,out=Cc)
    return Cc,cev,evec

def get_metrics(ret, nav):
    yr=(nav.index[-1]-nav.index[0]).days/365.25
    cagr=(nav.iloc[-1]/nav.iloc[0])**(1/yr)-1
    vol=ret.std()*np.sqrt(12)
    rfm=(1+0.02)**(1/12)-1
    sharpe=(ret-rfm).mean()/(ret-rfm).std()*np.sqrt(12) if ret.std()>0 else 0
    mdd=(nav/nav.cummax()-1).min()
    return cagr,vol,sharpe,mdd

# ═══════════════════════════════════════
# 3. 백테스트 (전체 기간) + 최신 월 포트폴리오 추출
# ═══════════════════════════════════════
print("백테스트 진행 중...", flush=True)
t_start = time.time()

q_rets_pre = {q: pd.Series(dtype=float) for q in range(N_QUINTILES)}
for q in range(N_QUINTILES):
    q_rets_pre[q][month_ends[0]] = 0.0

q5_rets = pd.Series(dtype=float)
q5_rets[month_ends[0]] = 0.0
q5_nav = 1.0; q5_prev_pf = pd.Series(dtype=float)
q5_turnover = []

# 최신 월 데이터 저장용
latest_portfolio = []
latest_all_scores = []
latest_params = {}
all_monthly_nav = {}  # 월별 NAV 기록

for i in range(len(month_ends) - 1):
    start = month_ends[i]; end = month_ends[i + 1]

    mc = mcap_monthly.loc[start].dropna() if start in mcap_monthly.index else pd.Series(dtype=float)
    mc = mc[mc > 0]

    tv60_now = tv60_monthly.loc[start].dropna() if start in tv60_monthly.index else pd.Series(dtype=float)
    if len(tv60_now) > 0:
        invest_codes = tv60_now[tv60_now > tv60_now.quantile(0.10)].index
    else:
        invest_codes = prices.columns

    dw = returns_daily.loc[:start].tail(LOOKBACK)
    if len(dw) < 40:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    vc = dw.dropna(axis=1, thresh=int(LOOKBACK * 0.8)).columns
    ds = dw[vc].dropna(axis=1)
    if len(ds.columns) < 100:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    cols = ds.columns
    Ta = ds.shape[0]
    mc_vals = mc.reindex(cols).fillna(0).values
    investable = cols.intersection(invest_codes)
    mc_invest = mc.reindex(investable).dropna()

    corr = equal_weight_correlation(ds.values)
    cc, ev, evec = marchenko_pastur_clipping_full(corr, Ta)
    v1 = np.abs(evec[:, 0]); tr = ev.sum()
    adj = np.maximum(cc, 0); np.fill_diagonal(adj, 0)

    cent_raw = compute_eigenvector_centrality(adj)
    valid_ll = (mc_vals > 0) & (cent_raw > 0)
    if valid_ll.sum() > 50:
        log_mc = np.log(mc_vals[valid_ll]); log_cent = np.log(cent_raw[valid_ll])
        X_ll = np.column_stack([np.ones(valid_ll.sum()), log_mc])
        b_ll = np.linalg.lstsq(X_ll, log_cent, rcond=None)[0]
        d_beta = abs(b_ll[1]) / 2
    else:
        d_beta = 0.25

    d_w = np.sqrt(ev[1] / ev[0]) if ev[0] > 0 else 0.5
    d_cut = min(np.sqrt(ev[0] / tr), 0.8)

    centrality = compute_cap_scaled_centrality(adj, mc_vals, beta=d_beta)
    cent_s = pd.Series(centrality, index=cols)
    lam1_s = pd.Series(v1, index=cols)

    if len(mc_invest) > 0:
        exclude = mc_invest[mc_invest >= mc_invest.quantile(1 - d_cut)].index
    else:
        exclude = pd.Index([])

    common = cols.intersection(investable).difference(exclude)
    if len(common) < 50:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    cent_common = cent_s[common]
    lam1_common = lam1_s[common]
    cent_z = (cent_common - cent_common.mean()) / cent_common.std() if cent_common.std() > 0 else cent_common * 0
    lam1_z = (lam1_common - lam1_common.mean()) / lam1_common.std() if lam1_common.std() > 0 else lam1_common * 0
    comp_score = cent_z - d_w * lam1_z

    Nc = len(common); npq = Nc // N_QUINTILES
    sorted_idx = np.argsort(comp_score.values)
    tickers = common

    quintile_map = {}
    for q in range(N_QUINTILES):
        si = q * npq; ei = Nc if q == N_QUINTILES - 1 else si + npq
        for idx in sorted_idx[si:ei]:
            quintile_map[tickers[idx]] = q

    # 수익률
    nr = returns_monthly_wins.loc[end].dropna() if end in returns_monthly_wins.index else pd.Series(dtype=float)
    if len(nr) == 0:
        for q in range(N_QUINTILES): q_rets_pre[q][end] = 0.0
        q5_rets[end] = 0.0; continue

    p_s = total_adj_monthly.loc[start] if start in total_adj_monthly.index else None
    p_e = total_adj_monthly.loc[end] if end in total_adj_monthly.index else None
    if p_s is not None and p_e is not None:
        common_idx = nr.index.intersection(p_s.dropna().index).intersection(p_e.dropna().index)
        suspended = p_s.reindex(common_idx)[(p_s.reindex(common_idx) == p_e.reindex(common_idx))].index
        nr = nr.drop(suspended, errors='ignore')

    amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
    if len(amihud_now) > 0:
        illiquid_set = set(amihud_now[amihud_now >= amihud_now.quantile(ILLIQ_THRESHOLD)].index)
    else:
        illiquid_set = set()

    for q in range(N_QUINTILES):
        q_tickers = [t for t, qq in quintile_map.items() if qq == q]
        q_in_nr = pd.Index(q_tickers).intersection(nr.index)
        q_rets_pre[q][end] = nr[q_in_nr].mean() if len(q_in_nr) > 0 else 0.0

    q5_codes = [t for t, qq in quintile_map.items() if qq == N_QUINTILES - 1]
    q5_filtered = [t for t in q5_codes if t not in spac_set]
    if start in prices_monthly.index:
        pr = prices_monthly.loc[start]
        q5_filtered = [t for t in q5_filtered if t in pr.index and pr[t] >= 1000]
    if len(q5_filtered) < 10:
        q5_filtered = q5_codes

    sel_tickers = pd.Index(q5_filtered).intersection(nr.index)
    if len(sel_tickers) == 0:
        q5_rets[end] = 0.0; continue

    weights = pd.Series(1.0 / len(sel_tickers), index=sel_tickers)

    if q5_prev_pf.sum() > 0:
        prev_weights = q5_prev_pf / q5_prev_pf.sum()
    else:
        prev_weights = pd.Series(dtype=float)

    all_index = weights.index.union(prev_weights.index)
    target_w = weights.reindex(all_index, fill_value=0)
    prev_w = prev_weights.reindex(all_index, fill_value=0)
    delta_w = target_w - prev_w
    trade_amounts = abs(delta_w) * q5_nav
    cost_rate = np.where(delta_w.index.isin(illiquid_set), HIGH_COST, LOW_COST)
    trade_cost = (trade_amounts * cost_rate).sum()
    NAV_after = q5_nav - trade_cost
    current_pv = weights * NAV_after
    ret_seg = nr.reindex(sel_tickers, fill_value=0)
    next_pv = current_pv * (1 + ret_seg)
    nav_new = next_pv.sum()
    portfolio_ret = nav_new / q5_nav - 1 if q5_nav > 0 else 0.0

    q5_rets[end] = portfolio_ret
    q5_nav = nav_new; q5_prev_pf = next_pv
    if q5_prev_pf.sum() > 0:
        q5_turnover.append(trade_amounts.sum() / q5_nav if q5_nav > 0 else 0)

    # NAV 기록
    all_monthly_nav[end.strftime('%Y-%m-%d')] = float((q5_rets + 1).cumprod().iloc[-1])

    # 마지막 월이면 포트폴리오 저장
    if i == len(month_ends) - 2:
        latest_params = {
            'date': start.strftime('%Y-%m-%d'),
            'next_date': end.strftime('%Y-%m-%d'),
            'beta': round(float(d_beta), 4),
            'w': round(float(d_w), 4),
            'cutoff': round(float(d_cut), 4),
            'n_universe': int(len(common)),
            'n_selected': int(len(sel_tickers)),
        }
        # Q5 종목 상세
        for t in sel_tickers:
            price_val = float(prices_monthly.loc[start, t]) if t in prices_monthly.columns and start in prices_monthly.index else 0
            mcap_val = float(mc[t]) if t in mc.index else 0
            latest_portfolio.append({
                'code': str(t),
                'name': str(code_to_name.get(t, t)),
                'score': round(float(comp_score[t]), 4) if t in comp_score.index else 0,
                'centrality': round(float(cent_s[t]), 6) if t in cent_s.index else 0,
                'lambda1': round(float(lam1_s[t]), 6) if t in lam1_s.index else 0,
                'quintile': 5,
                'price': int(price_val),
                'mcap_billion': round(mcap_val / 1e8, 0),
                'weight': round(float(weights[t]), 4) if t in weights.index else 0,
            })
        # 전체 종목 점수 (Q1~Q5)
        for t in common:
            latest_all_scores.append({
                'code': str(t),
                'name': str(code_to_name.get(t, t)),
                'score': round(float(comp_score[t]), 4) if t in comp_score.index else 0,
                'quintile': int(quintile_map.get(t, -1)) + 1,
            })

print(f"백테스트 완료: {time.time()-t_start:.1f}초")

# ═══════════════════════════════════════
# 4. 성과 계산
# ═══════════════════════════════════════
nav_q5 = (q5_rets + 1).cumprod()
m_q5 = get_metrics(q5_rets, nav_q5)
avg_to = np.mean(q5_turnover) if q5_turnover else 0

q_cagrs = []
for q in range(N_QUINTILES):
    nav_q = (q_rets_pre[q] + 1).cumprod()
    m = get_metrics(q_rets_pre[q], nav_q)
    q_cagrs.append(m[0])
mono = all(q_cagrs[j] <= q_cagrs[j+1] for j in range(4))

# OLS
import statsmodels.api as sm
valid_dates = [d for d in month_ends[1:] if d in mkt_factor.index and not np.isnan(mkt_factor.loc[d])]
alpha_ann = 0; alpha_t = 0
if len(valid_dates) > 10:
    y = np.array([q5_rets.get(d, 0) for d in valid_dates])
    rf_a = np.array([rf_monthly.loc[d] if d in rf_monthly.index else 0 for d in valid_dates])
    mkt_a = np.array([mkt_factor.loc[d] for d in valid_dates])
    hml_a = np.array([hml_ret.loc[d] for d in valid_dates])
    smb_a = np.array([smb_ret.loc[d] for d in valid_dates])
    mom_a = np.array([mom_ret.loc[d] for d in valid_dates])
    Y_ols = y - rf_a
    X_ols = pd.DataFrame({'MKT': mkt_a, 'SMB': smb_a, 'HML': hml_a, 'MOM': mom_a})
    X_ols = sm.add_constant(X_ols)
    try:
        model = sm.OLS(Y_ols, X_ols).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        alpha_ann = model.params['const'] * 12 * 100
        alpha_t = model.tvalues['const']
    except:
        pass

# ═══════════════════════════════════════
# 5. NAV 시계열 (차트용, 월별 샘플링)
# ═══════════════════════════════════════
nav_series = []
for q in range(N_QUINTILES):
    nav_q = (q_rets_pre[q] + 1).cumprod()
    for d, v in nav_q.items():
        nav_series.append({
            'date': d.strftime('%Y-%m-%d'),
            'quintile': q + 1,
            'nav': round(float(v), 4)
        })

q5_nav_series = []
for d, v in nav_q5.items():
    q5_nav_series.append({
        'date': d.strftime('%Y-%m-%d'),
        'nav': round(float(v), 4)
    })

# ═══════════════════════════════════════
# 6. JSON 저장
# ═══════════════════════════════════════
# 포트폴리오를 점수 내림차순 정렬
latest_portfolio.sort(key=lambda x: x['score'], reverse=True)
latest_all_scores.sort(key=lambda x: x['score'], reverse=True)

output = {
    'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    'params': latest_params,
    'performance': {
        'cagr': round(float(m_q5[0]) * 100, 2),
        'volatility': round(float(m_q5[1]) * 100, 1),
        'sharpe': round(float(m_q5[2]), 2),
        'mdd': round(float(m_q5[3]) * 100, 1),
        'turnover': round(float(avg_to) * 100, 1),
        'alpha': round(float(alpha_ann), 2),
        'alpha_t': round(float(alpha_t), 2),
        'monotonic': mono,
        'quintile_cagrs': [round(float(c) * 100, 2) for c in q_cagrs],
    },
    'portfolio': latest_portfolio,
    'all_scores': latest_all_scores[:200],  # 상위 200개만
    'nav_quintiles': nav_series,
    'nav_q5': q5_nav_series,
}

with open(DOCS_DIR / 'strategy_data.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nstrategy_data.json 저장 완료!")
print(f"  Q5 종목 수: {len(latest_portfolio)}")
print(f"  CAGR: {m_q5[0]:.2%}, Sharpe: {m_q5[2]:.2f}, MDD: {m_q5[3]:.1%}")
print(f"총 소요: {time.time()-t0:.1f}초")
