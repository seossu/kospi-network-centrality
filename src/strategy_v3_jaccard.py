"""
최종 전략 v3: Z-Score Composite + Jaccard 기반 동적 리밸런싱
- v2 기반, 리밸런싱 주기를 고정 월간 → 동적(Jaccard 트리거)으로 변경
- 매 영업일 Composite Score 산출
- 현재 Q5 vs 직전 리밸런싱 Q5의 Jaccard 유사도 <= 0.3 일 때만 리밸런싱
- 기간: 2000~
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import warnings, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # 원본 CSV 위치 (data/README.md 참고)
RESULTS_DIR = ROOT / "results"    # 결과물 저장 위치
DOCS_DIR = ROOT / "docs"          # 대시보드 (GitHub Pages)
RESULTS_DIR.mkdir(exist_ok=True)
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("  최종 전략 v3: Z-Score Composite + Jaccard 동적 리밸런싱")
print("=" * 70)
print("\n데이터 로딩 중...", flush=True)
t0 = time.time()

# ═══════════════════════════════════════
# 1. 데이터 로드 (parquet 캐시 + thousands=',' 파싱)
# ═══════════════════════════════════════
CACHE_DIR = DATA_DIR / ".cache_parquet"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_with_cache(csv_path, index_col):
    parquet_path = os.path.join(CACHE_DIR, os.path.basename(csv_path).replace('.csv', '.parquet'))
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index)
        return df
    df = pd.read_csv(csv_path, index_col=index_col, parse_dates=True, thousands=',', low_memory=False)
    # 남은 object 컬럼 강제 변환 (따옴표 등)
    for c in df.select_dtypes(include='object').columns:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(',', '', regex=False).str.replace('"', '', regex=False).str.strip(),
            errors='coerce'
        )
    df.to_parquet(parquet_path)
    return df

prices = load_with_cache(DATA_DIR / "수정주가_거래대금필터_code.csv", 'Date')
print(f"  수정주가: {prices.shape} ({time.time()-t0:.1f}초)", flush=True)

total_adj = load_with_cache(DATA_DIR / "현금배당포함.csv", '코드명')
print(f"  현금배당포함: {total_adj.shape} ({time.time()-t0:.1f}초)", flush=True)

mcap = load_with_cache(DATA_DIR / "시가총액.csv", '코드명')
print(f"  시가총액: {mcap.shape} ({time.time()-t0:.1f}초)", flush=True)

tv = load_with_cache(DATA_DIR / "거래대금.csv", 'Date')
print(f"  거래대금: {tv.shape} ({time.time()-t0:.1f}초)", flush=True)

tv60 = load_with_cache(DATA_DIR / "60_거래대금.csv", 'Date')
print(f"  60_거래대금: {tv60.shape} ({time.time()-t0:.1f}초)", flush=True)

factors_raw = pd.read_csv(DATA_DIR / "4팩터.csv", index_col='Date', parse_dates=True, thousands=',')
for col in factors_raw.columns:
    if factors_raw[col].dtype == 'object':
        factors_raw[col] = pd.to_numeric(factors_raw[col].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')
print(f"  4팩터: {factors_raw.shape}", flush=True)

# 코드↔종목명 매핑
code_cols = prices.columns
name_cols = total_adj.columns
code_to_name = dict(zip(code_cols, name_cols))
name_to_code = dict(zip(name_cols, code_cols))
total_adj.columns = [name_to_code.get(c, c) for c in total_adj.columns]
mcap.columns = [name_to_code.get(c, c) for c in mcap.columns]
print(f"  코드 매핑 완료: {len(code_to_name)}개", flush=True)

# 일별 수익률
returns_daily = prices.pct_change()
returns_total_daily = total_adj.pct_change()

# 일별 Amihud (60일 롤링 평균)
daily_ret_abs = returns_daily.abs()
daily_illiq = daily_ret_abs / tv
amihud_daily = daily_illiq.rolling(60, min_periods=30).mean()

# 4팩터 월말 리샘플 (alpha 회귀용)
factors_monthly = factors_raw.resample('ME').last()
kospi_factor_ret = factors_monthly['KOSPI'].pct_change() if 'KOSPI' in factors_monthly.columns else factors_monthly.iloc[:, 0].pct_change()
hml_ret = factors_monthly['HML'].pct_change() if 'HML' in factors_monthly.columns else factors_monthly.iloc[:, 1].pct_change()
smb_ret = factors_monthly['SMB'].pct_change() if 'SMB' in factors_monthly.columns else factors_monthly.iloc[:, 2].pct_change()
mom_ret = factors_monthly['MOM'].pct_change() if 'MOM' in factors_monthly.columns else factors_monthly.iloc[:, 3].pct_change()
cd91 = factors_monthly['CD91'] if 'CD91' in factors_monthly.columns else factors_monthly.iloc[:, 4]
rf_monthly = cd91 / 100 / 12
mkt_factor = kospi_factor_ret - rf_monthly

# 스팩
spac_set = set(c for c in prices.columns if '스팩' in str(c) or 'SPAC' in str(c))

# 백테스트 영업일
business_days = prices.loc['2000-01-01':'2025-12-31'].index
LOOKBACK = 60
N_QUINTILES = 5
JACCARD_THRESHOLD = 0.3
HIGH_COST = 0.008
LOW_COST = 0.003
ILLIQ_THRESHOLD = 0.80

# ─── numpy 변환 (inner loop 가속) ───
canonical_cols = prices.columns
canonical_idx = prices.index
N_STOCKS = len(canonical_cols)

returns_daily = returns_daily.reindex(canonical_idx).reindex(columns=canonical_cols)
returns_total_daily = returns_total_daily.reindex(canonical_idx).reindex(columns=canonical_cols)
mcap = mcap.reindex(canonical_idx).reindex(columns=canonical_cols)
tv60 = tv60.reindex(canonical_idx).reindex(columns=canonical_cols)
amihud_daily = amihud_daily.reindex(canonical_idx).reindex(columns=canonical_cols)
prices_aligned = prices.reindex(canonical_idx).reindex(columns=canonical_cols)

returns_np = returns_daily.values.astype(np.float64)
returns_total_np = returns_total_daily.values.astype(np.float64)
mcap_np = mcap.values.astype(np.float64)
tv60_np = tv60.values.astype(np.float64)
amihud_np = amihud_daily.values.astype(np.float64)
prices_np = prices_aligned.values.astype(np.float64)
date_to_idx = {d: i for i, d in enumerate(canonical_idx)}
spac_mask = np.array([c in spac_set for c in canonical_cols])

print(f"  로딩 완료: {time.time()-t0:.1f}초", flush=True)
print(f"  기간: {business_days[0].strftime('%Y-%m-%d')} ~ {business_days[-1].strftime('%Y-%m-%d')} ({len(business_days)}영업일)")
print(f"  Jaccard 임계값: {JACCARD_THRESHOLD}")

# ═══════════════════════════════════════
# 2. 함수 (v2와 동일)
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

def get_metrics_monthly(ret, nav):
    yr=(nav.index[-1]-nav.index[0]).days/365.25
    cagr=(nav.iloc[-1]/nav.iloc[0])**(1/yr)-1
    vol=ret.std()*np.sqrt(12)
    rfm=(1+0.02)**(1/12)-1
    sharpe=(ret-rfm).mean()/(ret-rfm).std()*np.sqrt(12) if ret.std()>0 else 0
    mdd=(nav/nav.cummax()-1).min()
    return cagr,vol,sharpe,mdd

def compute_q5_set(i_today):
    """today_idx 정수 인덱스 기반. 유효 데이터 부족 시 (None, None) 반환."""
    if i_today < LOOKBACK - 1:
        return None, None

    # 60일 윈도우 (numpy view, zero-copy)
    window = returns_np[i_today - LOOKBACK + 1 : i_today + 1]  # (60, N_STOCKS)
    if window.shape[0] < 40:
        return None, None

    # dropna 컬럼 (80% 이상 유효한 것만 남기고 → 그중에서도 NaN 0개)
    nan_count = np.isnan(window).sum(axis=0)
    initial_mask = (LOOKBACK - nan_count) >= int(LOOKBACK * 0.8)
    if initial_mask.sum() < 100:
        return None, None
    window_first = window[:, initial_mask]
    no_nan_local = ~np.isnan(window_first).any(axis=0)
    if no_nan_local.sum() < 100:
        return None, None

    # 글로벌 mask: initial_mask True + 그 내부에서 no_nan True
    cols_idx_in_initial = np.where(initial_mask)[0]
    cols_idx = cols_idx_in_initial[no_nan_local]  # 글로벌 컬럼 인덱스
    ds_np = window_first[:, no_nan_local]
    cols_tickers = canonical_cols[cols_idx]
    Ta = ds_np.shape[0]

    # 시총
    mcap_today = mcap_np[i_today]
    mc_vals = np.nan_to_num(mcap_today[cols_idx], nan=0.0)

    # 거래대금 필터 (글로벌)
    tv60_today = tv60_np[i_today]
    tv60_valid = ~np.isnan(tv60_today) & (tv60_today > 0)
    if tv60_valid.sum() > 0:
        tv60_q10 = np.nanquantile(tv60_today[tv60_valid], 0.10)
        invest_mask_global = tv60_valid & (tv60_today > tv60_q10)
    else:
        invest_mask_global = np.ones(N_STOCKS, dtype=bool)

    # investable: cols_idx 중에서 invest_mask가 True인 것
    investable_local = invest_mask_global[cols_idx]  # boolean (len = len(cols_idx))

    # 상관행렬 + MP Clipping
    corr = equal_weight_correlation(ds_np)
    cc, ev, evec = marchenko_pastur_clipping_full(corr, Ta)
    v1 = np.abs(evec[:, 0])
    tr = ev.sum()
    adj = np.maximum(cc, 0)
    np.fill_diagonal(adj, 0)

    cent_raw = compute_eigenvector_centrality(adj)
    valid_ll = (mc_vals > 0) & (cent_raw > 0)
    if valid_ll.sum() > 50:
        log_mc = np.log(mc_vals[valid_ll])
        log_cent = np.log(cent_raw[valid_ll])
        X_ll = np.column_stack([np.ones(valid_ll.sum()), log_mc])
        b_ll = np.linalg.lstsq(X_ll, log_cent, rcond=None)[0]
        d_beta = abs(b_ll[1]) / 2
    else:
        d_beta = 0.25

    d_w = np.sqrt(ev[1] / ev[0]) if ev[0] > 0 else 0.5
    d_cut = min(np.sqrt(ev[0] / tr), 0.8)
    centrality = compute_cap_scaled_centrality(adj, mc_vals, beta=d_beta)

    # 시총 cutoff (investable 내에서 상위 d_cut 제외)
    mc_invest_vals = mc_vals[investable_local]
    if (mc_invest_vals > 0).sum() > 0:
        mc_thresh = np.quantile(mc_invest_vals[mc_invest_vals > 0], 1 - d_cut)
        cutoff_mask_local = ~((mc_vals >= mc_thresh) & investable_local)  # exclude 위쪽
    else:
        cutoff_mask_local = np.ones(len(cols_idx), dtype=bool)

    common_mask_local = investable_local & cutoff_mask_local
    if common_mask_local.sum() < 50:
        return None, None

    # Z-Score Composite (common 종목만)
    cent_common = centrality[common_mask_local]
    lam1_common = v1[common_mask_local]
    cent_std = cent_common.std()
    lam1_std = lam1_common.std()
    cent_z = (cent_common - cent_common.mean()) / cent_std if cent_std > 0 else np.zeros_like(cent_common)
    lam1_z = (lam1_common - lam1_common.mean()) / lam1_std if lam1_std > 0 else np.zeros_like(lam1_common)
    comp_score = cent_z - d_w * lam1_z

    # Q5 선정 (점수 상위 1/N_QUINTILES)
    Nc = len(comp_score)
    npq = Nc // N_QUINTILES
    sorted_local = np.argsort(comp_score)
    q5_start = (N_QUINTILES - 1) * npq
    q5_local_in_common = sorted_local[q5_start:]

    # common_mask_local의 글로벌 인덱스
    common_global_idx = cols_idx[common_mask_local]
    q5_global_idx = common_global_idx[q5_local_in_common]
    q5_tickers_raw = canonical_cols[q5_global_idx]

    # 스팩 필터
    q5_not_spac = ~spac_mask[q5_global_idx]
    # 동전주 필터 (price >= 1000)
    prices_today = prices_np[i_today]
    q5_prices = prices_today[q5_global_idx]
    q5_valid_price = (~np.isnan(q5_prices)) & (q5_prices >= 1000)
    filter_mask = q5_not_spac & q5_valid_price
    q5_filtered = q5_tickers_raw[filter_mask]
    if len(q5_filtered) < 10:
        q5_filtered = q5_tickers_raw  # fallback

    info = {
        'comp_score': comp_score,
        'centrality_common': cent_common,
        'lam1_common': lam1_common,
        'common_global_idx': common_global_idx,
        'beta': d_beta,
        'w': d_w,
        'cutoff': d_cut,
    }
    return set(q5_filtered.tolist()), info

# ═══════════════════════════════════════
# 3. 일별 백테스트 (Jaccard 트리거)
# ═══════════════════════════════════════
print(f"\n일별 백테스트 시작...", flush=True)
t_start = time.time()

ticker_to_idx = {t: i for i, t in enumerate(canonical_cols)}

# 포트폴리오 상태: numpy 배열로 관리
pf_idx = np.empty(0, dtype=np.int64)   # 보유 종목의 글로벌 인덱스
pf_val = np.empty(0, dtype=np.float64) # 보유 금액
last_rebal_q5 = set()
last_rebal_date = None
nav = 1.0

daily_records = []
jaccard_records = []
rebal_dates = []
score_rows = []
turnover_list = []

n_days = len(business_days)
business_day_indices = np.array([date_to_idx.get(d, -1) for d in business_days])

for loop_i, today in enumerate(business_days):
    i_today = business_day_indices[loop_i]
    if i_today < 0:
        continue
    if loop_i % 250 == 0:
        print(f"  {today.strftime('%Y-%m-%d')} ({loop_i}/{n_days}, {time.time()-t_start:.0f}초, 리밸 {len(rebal_dates)}회)", flush=True)

    # 1) 일간 NAV 업데이트
    if pf_idx.size > 0:
        ret_today = returns_total_np[i_today, pf_idx]
        ret_today = np.nan_to_num(ret_today, nan=0.0)
        pf_val = pf_val * (1.0 + ret_today)
        nav = pf_val.sum()

    # 2) 오늘의 Q5 산출
    q5_today, info = compute_q5_set(i_today)
    if q5_today is None:
        daily_records.append({'date': today, 'nav': nav, 'jaccard': np.nan, 'q5_size': 0, 'holdings': pf_idx.size})
        continue

    # 3) Jaccard
    if last_rebal_q5:
        inter = len(q5_today & last_rebal_q5)
        union = len(q5_today | last_rebal_q5)
        jaccard = inter / union if union > 0 else 1.0
    else:
        jaccard = 0.0

    jaccard_records.append({'date': today, 'jaccard': jaccard, 'q5_size': len(q5_today)})
    daily_records.append({'date': today, 'nav': nav, 'jaccard': jaccard, 'q5_size': len(q5_today), 'holdings': pf_idx.size})

    if (last_rebal_q5) and jaccard > JACCARD_THRESHOLD:
        continue

    # 5) 리밸런싱 실행
    target_idx = np.array([ticker_to_idx[t] for t in q5_today if t in ticker_to_idx], dtype=np.int64)
    # 오늘 가격이 NaN인 종목 추가 제외
    target_prices = prices_np[i_today, target_idx]
    target_idx = target_idx[~np.isnan(target_prices)]
    if target_idx.size == 0:
        continue
    target_w_val = 1.0 / target_idx.size

    # Amihud illiquid (전역 mask)
    amihud_today = amihud_np[i_today]
    amihud_valid = ~np.isnan(amihud_today)
    if amihud_valid.sum() > 0:
        amihud_thresh = np.quantile(amihud_today[amihud_valid], ILLIQ_THRESHOLD)
        illiquid_mask = amihud_valid & (amihud_today >= amihud_thresh)
    else:
        illiquid_mask = np.zeros(N_STOCKS, dtype=bool)

    # delta_w: 합집합 인덱스에 대해
    prev_total = pf_val.sum() if pf_idx.size > 0 else 0
    if prev_total > 0:
        prev_w_dict = dict(zip(pf_idx.tolist(), (pf_val / prev_total).tolist()))
    else:
        prev_w_dict = {}
    target_w_dict = {int(idx): target_w_val for idx in target_idx}

    union_idx = sorted(set(prev_w_dict) | set(target_w_dict))
    delta = np.array([abs(target_w_dict.get(k, 0) - prev_w_dict.get(k, 0)) for k in union_idx])
    union_idx_arr = np.array(union_idx, dtype=np.int64)
    cost_rate = np.where(illiquid_mask[union_idx_arr], HIGH_COST, LOW_COST)
    trade_cost = (delta * nav * cost_rate).sum()
    nav_after = nav - trade_cost

    pf_idx = target_idx
    pf_val = np.full(target_idx.size, target_w_val * nav_after)
    nav = nav_after
    last_rebal_q5 = set(q5_today)
    last_rebal_date = today
    rebal_dates.append(today)
    if nav > 0:
        turnover_list.append(delta.sum())  # 일방향 turnover (sum of abs delta_w)

    # 점수 저장 (common 종목)
    date_str = str(today.date())
    comp_score = info['comp_score']
    cent_common = info['centrality_common']
    lam1_common = info['lam1_common']
    common_global_idx = info['common_global_idx']
    for j in range(len(comp_score)):
        t_idx = int(common_global_idx[j])
        ticker = canonical_cols[t_idx]
        score_rows.append({
            'date': date_str,
            'ticker': ticker,
            'composite_score': float(comp_score[j]),
            'in_q5': int(ticker in q5_today),
            'centrality': float(cent_common[j]),
            'lambda1_loading': float(lam1_common[j]),
            'beta': float(info['beta']),
            'w': float(info['w']),
            'cutoff': float(info['cutoff']),
            'jaccard_at_rebal': float(jaccard),
        })

elapsed = time.time() - t_start
print(f"  백테스트 완료: {elapsed:.1f}초")
print(f"  총 리밸런싱 횟수: {len(rebal_dates)}회")
if len(rebal_dates) > 1:
    gaps = np.diff([(d - rebal_dates[0]).days for d in rebal_dates])
    print(f"  평균 리밸런싱 간격: {gaps.mean():.1f}일 (중앙값 {np.median(gaps):.0f}일, 최소 {gaps.min()}일, 최대 {gaps.max()}일)")

# ═══════════════════════════════════════
# 4. 성과 계산
# ═══════════════════════════════════════
daily_df = pd.DataFrame(daily_records).set_index('date')
nav_daily = daily_df['nav']
# 월간 리샘플 (지표용)
nav_monthly = nav_daily.resample('ME').last()
ret_monthly = nav_monthly.pct_change().fillna(0)
m_q5 = get_metrics_monthly(ret_monthly[ret_monthly.index >= rebal_dates[0]] if rebal_dates else ret_monthly,
                            nav_monthly[nav_monthly.index >= rebal_dates[0]] if rebal_dates else nav_monthly)
avg_to = np.mean(turnover_list) if turnover_list else 0

# 4팩터 OLS (월간 기준, HAC)
valid_dates = [d for d in nav_monthly.index[1:] if d in mkt_factor.index and not np.isnan(mkt_factor.loc[d])]
if len(valid_dates) > 10:
    y = np.array([ret_monthly.get(d, 0) for d in valid_dates])
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
    except Exception:
        alpha_ann = 0; alpha_t = 0; model = None
else:
    alpha_ann = 0; alpha_t = 0; model = None

# ═══════════════════════════════════════
# 5. 출력
# ═══════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  최종 전략 v3 성과 (Jaccard 동적 리밸런싱, 수수료후, 2000~)")
print(f"{'=' * 70}")
print(f"\n  [Q5 수수료후]")
print(f"  CAGR        = {m_q5[0]:.2%}")
print(f"  Vol         = {m_q5[1]:.1%}")
print(f"  Sharpe      = {m_q5[2]:.2f}")
print(f"  MDD         = {m_q5[3]:.1%}")
print(f"  평균 턴오버 = {avg_to:.1%}")
print(f"  리밸런싱 수 = {len(rebal_dates)}회 ({(business_days[-1]-business_days[0]).days/365.25:.1f}년)")
if alpha_ann:
    print(f"  α(HAC)      = {alpha_ann:.2f}% (t={alpha_t:.2f})")
if model:
    print(f"\n  [OLS 결과]")
    print(model.summary())

# ═══════════════════════════════════════
# 6. 시각화
# ═══════════════════════════════════════
jaccard_df = pd.DataFrame(jaccard_records).set_index('date') if jaccard_records else pd.DataFrame()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Panel 1: Q5 NAV (일별)
ax1 = axes[0, 0]
ax1.plot(nav_daily.index, nav_daily.values, color='#1f77b4', linewidth=1.5,
         label=f'Q5 수수료후: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}')
ax1.set_title('Q5 NAV (일별, Jaccard 동적 리밸런싱)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.set_ylabel('NAV'); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')

# Panel 2: Jaccard 시계열 + 리밸 시점
ax2 = axes[0, 1]
if not jaccard_df.empty:
    ax2.plot(jaccard_df.index, jaccard_df['jaccard'].values, color='#888', linewidth=0.7, alpha=0.7, label='Jaccard')
    ax2.axhline(y=JACCARD_THRESHOLD, color='red', linestyle='--', linewidth=1, label=f'임계값 {JACCARD_THRESHOLD}')
    for d in rebal_dates:
        ax2.axvline(x=d, color='#2ca02c', alpha=0.15, linewidth=0.6)
ax2.set_title(f'Jaccard 유사도 시계열 (리밸 {len(rebal_dates)}회 — 녹색 세로선)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Jaccard'); ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=10)

# Panel 3: 리밸 간격 히스토그램
ax3 = axes[1, 0]
if len(rebal_dates) > 1:
    gaps = np.diff([(d - rebal_dates[0]).days for d in rebal_dates])
    ax3.hist(gaps, bins=30, color='#1f77b4', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax3.axvline(x=gaps.mean(), color='red', linestyle='--', linewidth=1, label=f'평균 {gaps.mean():.0f}일')
    ax3.axvline(x=np.median(gaps), color='orange', linestyle='--', linewidth=1, label=f'중앙값 {np.median(gaps):.0f}일')
    ax3.legend(fontsize=10)
ax3.set_title('리밸런싱 간격 분포 (일)', fontsize=14, fontweight='bold')
ax3.set_xlabel('간격(일)'); ax3.set_ylabel('빈도'); ax3.grid(True, alpha=0.3)

# Panel 4: 보유 종목 수 시계열
ax4 = axes[1, 1]
ax4.plot(daily_df.index, daily_df['holdings'].values, color='#9467bd', linewidth=0.8)
ax4.set_title('보유 종목 수 (일별)', fontsize=14, fontweight='bold')
ax4.set_ylabel('종목 수'); ax4.grid(True, alpha=0.3)

fig.suptitle(
    f'네트워크 기반 전략 v3 (Jaccard ≤ {JACCARD_THRESHOLD} 동적 리밸런싱, 2000~)\n'
    f'Q5: CAGR={m_q5[0]:.1%}, Sharpe={m_q5[2]:.2f}, α={alpha_ann:.1f}%(t={alpha_t:.2f}), MDD={m_q5[3]:.1%}, 리밸 {len(rebal_dates)}회',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'v3_jaccard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nv3_jaccard.png 저장 완료!")

# ═══════════════════════════════════════
# 7. 저장
# ═══════════════════════════════════════
df_out = pd.DataFrame({
    'Date': nav_daily.index,
    'NAV': nav_daily.values,
    'Log_NAV': np.log(nav_daily.values),
    'Daily_Return': nav_daily.pct_change().fillna(0).values,
    'Jaccard': daily_df['jaccard'].values,
    'Q5_Size': daily_df['q5_size'].values,
    'Holdings': daily_df['holdings'].values,
    'Is_Rebal': [int(d in set(rebal_dates)) for d in nav_daily.index],
})
df_out.to_excel(RESULTS_DIR / "v3_nav.xlsx", index=False)
print("v3_nav.xlsx 저장 완료!")

# 점수 (리밸 시점만)
if score_rows:
    df_scores = pd.DataFrame(score_rows)
    df_scores.to_csv(RESULTS_DIR / "monthly_scores_v3.csv", index=False, encoding='utf-8-sig')
    print(f"monthly_scores_v3.csv 저장 완료 ({len(df_scores):,}행, 리밸 시점만)")

# 리밸 이벤트 로그
df_rebal = pd.DataFrame({
    'rebal_date': rebal_dates,
    'gap_days': [0] + [(rebal_dates[k] - rebal_dates[k-1]).days for k in range(1, len(rebal_dates))],
})
df_rebal.to_csv(RESULTS_DIR / "rebal_events_v3.csv", index=False, encoding='utf-8-sig')
print(f"rebal_events_v3.csv 저장 완료 ({len(df_rebal)}건)")

print(f"\n총 소요: {time.time()-t0:.1f}초")
