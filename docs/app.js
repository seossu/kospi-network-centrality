/**
 * 네트워크 기반 퀀트 전략 대시보드
 * strategy_data.json을 로드하여 시각화
 */

const QUINTILE_COLORS = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#4f8ff7'];
const QUINTILE_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'];

async function loadData() {
    try {
        const res = await fetch('strategy_data.json');
        if (!res.ok) throw new Error('strategy_data.json을 찾을 수 없습니다. python generate_data.py를 먼저 실행하세요.');
        return await res.json();
    } catch (e) {
        document.querySelector('main').innerHTML = `
            <div class="card" style="text-align:center;padding:3rem">
                <h2 style="color:#e74c3c">데이터 로드 실패</h2>
                <p style="margin-top:1rem;color:#8b8fa3">${e.message}</p>
                <p style="margin-top:0.5rem;color:#8b8fa3">터미널에서 아래 명령어를 실행하세요:</p>
                <code style="display:block;margin-top:1rem;padding:1rem;background:#0d0f15;border-radius:8px;color:#2ecc71">
                    cd "봄학기 1차 최종" && python generate_data.py
                </code>
            </div>`;
        return null;
    }
}

function renderGenTime(data) {
    document.getElementById('genTime').textContent = `데이터 생성: ${data.generated_at}`;
}

function renderParams(data) {
    const p = data.params;
    const grid = document.getElementById('paramGrid');
    const items = [
        { label: '기준일', value: p.date },
        { label: '유니버스', value: `${p.n_universe}개` },
        { label: 'Beta (β)', value: p.beta.toFixed(4) },
        { label: 'Weight (w)', value: p.w.toFixed(4) },
        { label: 'Cutoff', value: `${(p.cutoff * 100).toFixed(1)}%` },
        { label: 'Q5 종목 수', value: `${p.n_selected}개` },
    ];
    grid.innerHTML = items.map(i => `
        <div class="param-item">
            <div class="label">${i.label}</div>
            <div class="value">${i.value}</div>
        </div>
    `).join('');
}

function renderPerformance(data) {
    const p = data.performance;
    const grid = document.getElementById('perfGrid');
    const items = [
        { label: 'CAGR', value: `${p.cagr.toFixed(2)}%`, cls: p.cagr > 0 ? 'positive' : 'negative' },
        { label: 'Sharpe Ratio', value: p.sharpe.toFixed(2), cls: p.sharpe > 0 ? 'positive' : 'negative' },
        { label: 'MDD', value: `${p.mdd.toFixed(1)}%`, cls: 'negative' },
        { label: 'Volatility', value: `${p.volatility.toFixed(1)}%`, cls: '' },
        { label: 'Alpha (HAC)', value: `${p.alpha.toFixed(2)}%`, cls: p.alpha > 0 ? 'positive' : 'negative' },
        { label: 'Alpha t-stat', value: p.alpha_t.toFixed(2), cls: Math.abs(p.alpha_t) > 2 ? 'positive' : '' },
        { label: '월 턴오버', value: `${p.turnover.toFixed(1)}%`, cls: '' },
        { label: '단조성', value: p.monotonic ? 'O (확인)' : 'X', cls: p.monotonic ? 'positive' : 'negative' },
    ];
    grid.innerHTML = items.map(i => `
        <div class="perf-item">
            <div class="label">${i.label}</div>
            <div class="value ${i.cls}">${i.value}</div>
        </div>
    `).join('');
}

function renderPortfolioTable(data) {
    const tbody = document.querySelector('#portfolioTable tbody');
    tbody.innerHTML = data.portfolio.map((s, i) => `
        <tr>
            <td>${i + 1}</td>
            <td><code>${s.code}</code></td>
            <td><strong>${s.name}</strong></td>
            <td class="num">${s.score.toFixed(4)}</td>
            <td class="num">${s.centrality.toFixed(6)}</td>
            <td class="num">${s.lambda1.toFixed(6)}</td>
            <td class="num">${s.price ? s.price.toLocaleString() + '원' : '-'}</td>
            <td class="num">${s.mcap_billion ? Math.round(s.mcap_billion).toLocaleString() : '-'}</td>
            <td class="num">${(s.weight * 100).toFixed(2)}%</td>
        </tr>
    `).join('');

    document.getElementById('filterResult').textContent =
        `복합점수 상위 20% → ${data.portfolio.length}종목 등가중 편입`;
}

function renderAllScoresTable(data) {
    const tbody = document.querySelector('#allScoresTable tbody');
    tbody.innerHTML = data.all_scores.map((s, i) => `
        <tr>
            <td>${i + 1}</td>
            <td><code>${s.code}</code></td>
            <td>${s.name}</td>
            <td class="num">${s.score.toFixed(4)}</td>
            <td><span class="quintile-badge q${s.quintile}">Q${s.quintile}</span></td>
        </tr>
    `).join('');
}

function renderQuintileChart(data) {
    const ctx = document.getElementById('quintileChart').getContext('2d');

    // 퀀타일별 데이터 그룹핑
    const quintileData = {};
    for (let q = 1; q <= 5; q++) {
        quintileData[q] = data.nav_quintiles
            .filter(d => d.quintile === q)
            .map(d => ({ x: d.date, y: d.nav }));
    }

    // 너무 많은 포인트 → 간격 샘플링
    function sampleData(arr, maxPoints) {
        if (arr.length <= maxPoints) return arr;
        const step = Math.ceil(arr.length / maxPoints);
        const result = [];
        for (let i = 0; i < arr.length; i += step) result.push(arr[i]);
        if (result[result.length - 1] !== arr[arr.length - 1]) result.push(arr[arr.length - 1]);
        return result;
    }

    const datasets = [];
    for (let q = 1; q <= 5; q++) {
        const sampled = sampleData(quintileData[q], 150);
        datasets.push({
            label: `Q${q} (CAGR ${data.performance.quintile_cagrs[q - 1].toFixed(1)}%)`,
            data: sampled,
            borderColor: QUINTILE_COLORS[q - 1],
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
        });
    }

    new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    ticks: {
                        maxTicksLimit: 8,
                        color: '#8b8fa3',
                        font: { size: 10 },
                        callback: function(val) {
                            const label = this.getLabelForValue(val);
                            return label ? label.substring(0, 7) : '';
                        }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    type: 'logarithmic',
                    ticks: { color: '#8b8fa3', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e1e4ed', font: { size: 11 } }
                }
            }
        }
    });
}

function renderQ5Chart(data) {
    const ctx = document.getElementById('q5Chart').getContext('2d');
    const sampled = sampleData(data.nav_q5, 150);

    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: `Q5 수수료후 (CAGR ${data.performance.cagr.toFixed(1)}%)`,
                data: sampled,
                borderColor: '#4f8ff7',
                backgroundColor: 'rgba(79,143,247,0.1)',
                fill: true,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'category',
                    ticks: {
                        maxTicksLimit: 8,
                        color: '#8b8fa3',
                        font: { size: 10 },
                        callback: function(val) {
                            const label = this.getLabelForValue(val);
                            return label ? label.substring(0, 7) : '';
                        }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    type: 'logarithmic',
                    ticks: { color: '#8b8fa3', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e1e4ed', font: { size: 11 } }
                }
            }
        }
    });
}

function renderCAGRChart(data) {
    const ctx = document.getElementById('cagrChart').getContext('2d');
    const cagrs = data.performance.quintile_cagrs;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: QUINTILE_LABELS,
            datasets: [{
                label: 'CAGR (%)',
                data: cagrs,
                backgroundColor: QUINTILE_COLORS,
                borderColor: QUINTILE_COLORS.map(c => c),
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    ticks: { color: '#e1e4ed', font: { size: 12, weight: 'bold' } },
                    grid: { display: false },
                },
                y: {
                    ticks: {
                        color: '#8b8fa3',
                        callback: v => v.toFixed(1) + '%'
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `CAGR: ${ctx.raw.toFixed(2)}%`
                    }
                }
            }
        }
    });
}

function sampleData(arr, maxPoints) {
    if (arr.length <= maxPoints) return arr;
    const step = Math.ceil(arr.length / maxPoints);
    const result = [];
    for (let i = 0; i < arr.length; i += step) result.push(arr[i]);
    if (result[result.length - 1] !== arr[arr.length - 1]) result.push(arr[arr.length - 1]);
    return result;
}

// 메인 실행
async function main() {
    const data = await loadData();
    if (!data) return;

    renderGenTime(data);
    renderParams(data);
    renderPerformance(data);
    renderPortfolioTable(data);
    renderAllScoresTable(data);
    renderQuintileChart(data);
    renderQ5Chart(data);
    renderCAGRChart(data);
}

main();
