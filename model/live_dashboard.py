#!/usr/bin/env python3
"""
live_dashboard.py — Lightweight live web dashboard for the simulation.

Serves a single-page dashboard at http://localhost:8050 that polls
Redis every second and renders real-time charts.

Usage:
    # Terminal 1: run the simulation
    python run_simulation.py

    # Terminal 2: launch the dashboard
    python live_dashboard.py

    Open http://localhost:8050
"""

import json

from flask import Flask, jsonify, Response
import redis as _redis

# ── Redis connection (localhost only) ────────────────────────────────────
r = _redis.Redis(host="localhost", port=6379, db=0,
                 decode_responses=True, socket_timeout=3)
try:
    r.ping()
    print("  ✓ Redis connected: localhost:6379")
except Exception as e:
    print(f"  ✗ Redis unavailable: {e}")

app = Flask(__name__)


@app.route("/api/state")
def api_state():
    """Return full Redis snapshot as JSON."""
    pipe = r.pipeline()
    for key in ("step", "n_banks", "aggregate_liq", "aggregate_exp",
                "n_stressed", "n_defaulted", "margin_rate"):
        pipe.get(f"system:{key}")
    pipe.get("market:latest")
    results = pipe.execute()

    fields = ["step", "n_banks", "aggregate_liq", "aggregate_exp",
              "n_stressed", "n_defaulted", "margin_rate"]
    sys_data = {}
    for i, f in enumerate(fields):
        v = results[i]
        sys_data[f] = float(v) if v is not None else 0.0

    market_raw = results[7]
    market = (json.loads(market_raw) if market_raw
              else {"new_volatility": 0, "price_change_signal": 0})

    n_banks = int(sys_data.get("n_banks", 0))
    banks = {}
    if n_banks > 0:
        pipe2 = r.pipeline()
        for bid in range(n_banks):
            pipe2.hgetall(f"bank:{bid}:state")
        bank_results = pipe2.execute()
        for bid, raw in enumerate(bank_results):
            if raw:
                banks[str(bid)] = {k: float(v) for k, v in raw.items()}

    # recent intents (last 10)
    raw_intents = r.lrange("intents:queue", -10, -1)
    intents = [json.loads(x) for x in raw_intents] if raw_intents else []
    intent_count = r.llen("intents:queue")

    # CCP payoff components
    ccp_raw = r.hgetall("ccp:state")
    ccp = {k: float(v) for k, v in ccp_raw.items()} if ccp_raw else {}

    # Network topology
    net_raw = r.get("network:topology")
    network = json.loads(net_raw) if net_raw else {"nodes": [], "edges": []}

    return jsonify({
        "system": sys_data,
        "market": market,
        "banks": banks,
        "intents": intents,
        "intent_count": intent_count,
        "ccp": ccp,
        "network": network,
    })


@app.route("/")
def index():
    return Response(DASHBOARD_HTML, content_type="text/html")


# ── Inline HTML/JS dashboard ────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Financial Network — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0e17; color: #d0d0d0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 16px 24px;
  }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 18px; border-bottom: 1px solid #1e2433; padding-bottom: 12px;
  }
  .header h1 { font-size: 1.4rem; color: #00d4aa; font-weight: 600; }
  .header .meta { font-size: 0.8rem; opacity: 0.6; }
  .status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; animation: pulse 2s infinite;
  }
  .status-dot.live { background: #00d4aa; }
  .status-dot.stale { background: #ff4d6a; animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* KPI row */
  .kpi-row { display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 18px; }
  .kpi {
    background: #111827; border-radius: 10px; padding: 14px 12px;
    text-align: center; border: 1px solid #1e2433;
  }
  .kpi .label { font-size: 0.7rem; opacity: 0.5; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
  .kpi .value.green { color: #00d4aa; }
  .kpi .value.blue { color: #4dabf7; }
  .kpi .value.yellow { color: #ffc107; }
  .kpi .value.red { color: #ff4d6a; }
  .kpi .value.purple { color: #b197fc; }

  /* Chart grid */
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
  .chart-card {
    background: #111827; border-radius: 10px; padding: 14px;
    border: 1px solid #1e2433;
  }
  .chart-card h3 { font-size: 0.8rem; color: #7a8599; margin-bottom: 8px; font-weight: 500; }

  /* Bank table */
  .bank-section { margin-top: 14px; }
  .bank-section h3 { font-size: 0.85rem; color: #7a8599; margin-bottom: 8px; }
  table {
    width: 100%; border-collapse: collapse; font-size: 0.78rem;
    background: #111827; border-radius: 10px; overflow: hidden;
  }
  th {
    background: #1a2035; padding: 8px 12px; text-align: right;
    color: #7a8599; font-weight: 500; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  th:first-child { text-align: left; }
  td { padding: 7px 12px; text-align: right; border-top: 1px solid #1e2433; }
  td:first-child { text-align: left; font-weight: 600; }
  tr:hover { background: #151d2f; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 0.65rem; font-weight: 600;
  }
  .badge.ok { background: #0d3320; color: #00d4aa; }
  .badge.stressed { background: #3d2e00; color: #ffc107; }
  .badge.defaulted { background: #3d0d15; color: #ff4d6a; }

  /* Intent feed */
  .intent-feed {
    background: #111827; border-radius: 10px; padding: 14px;
    border: 1px solid #1e2433; max-height: 250px; overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.72rem;
  }
  .intent-item {
    padding: 4px 0; border-bottom: 1px solid #1a2035;
    display: flex; gap: 8px;
  }
  .intent-item .tick { color: #7a8599; min-width: 50px; }
  .intent-item .agent { color: #4dabf7; min-width: 70px; }
  .intent-item .action { color: #00d4aa; }
  .intent-item .vis { font-size: 0.65rem; }

  .bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
</style>
</head>
<body>

<div class="header">
  <h1>📊 Financial Network — Live Dashboard</h1>
  <div class="meta">
    <span class="status-dot live" id="statusDot"></span>
    <span id="headerMeta">Connecting…</span>
  </div>
</div>

<div class="kpi-row" id="kpiRow"></div>

<div class="charts">
  <div class="chart-card">
    <h3>Aggregate Liquidity & Exposure</h3>
    <canvas id="chartLiqExp" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>Market: Volatility & Margin Rate</h3>
    <canvas id="chartMarket" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>Bank Status Breakdown</h3>
    <canvas id="chartStatus" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>Per-Bank Liquidity</h3>
    <canvas id="chartBankLiq" height="180"></canvas>
  </div>
  <div class="chart-card">
    <h3>CCP Payoff Function — Utility Decomposition</h3>
    <canvas id="chartCCPPayoff" height="180"></canvas>
  </div>
  <div class="chart-card" style="padding: 0; overflow: hidden;">
    <h3 style="padding: 14px 14px 0 14px;">Network Snapshot <span style="font-size:0.65rem;opacity:0.5">(drag • scroll)</span></h3>
    <div id="networkGraph" style="width:100%; height:220px; background:#0d1117;"></div>
  </div>
</div>

<div class="bottom-grid">
  <div class="bank-section">
    <h3>Per-Bank State</h3>
    <table>
      <thead><tr>
        <th>Bank</th><th>Liquidity</th><th>Capital</th>
        <th>Exposure</th><th>Status</th>
      </tr></thead>
      <tbody id="bankTableBody"></tbody>
    </table>
  </div>
  <div>
    <h3 style="font-size:0.85rem;color:#7a8599;margin-bottom:8px">Recent Intents</h3>
    <div class="intent-feed" id="intentFeed"></div>
  </div>
</div>

<script>
// ── Config ──────────────────────────────────────────────────────────
const MAX_POINTS = 500;
const POLL_MS = 1000;
const COLORS = {
  green: '#00d4aa', blue: '#4dabf7', yellow: '#ffc107',
  red: '#ff4d6a', purple: '#b197fc', grid: '#1e2433',
  text: '#7a8599',
};
const BANK_COLORS = [
  '#00d4aa','#4dabf7','#b197fc','#ffc107','#ff4d6a',
  '#20c997','#748ffc','#e599f7','#f59f00','#ff6b6b',
];

// ── History buffers ─────────────────────────────────────────────────
const hist = {
  tick: [], liq: [], exp: [], vol: [], mr: [], ps: [],
  stressed: [], defaulted: [], healthy: [],
};
const ccpHist = {
  utility: [], comp_stability: [], comp_fund: [],
  comp_defaults: [], comp_firesale: [],
};
const bankHist = {};  // bid -> [values]
let lastTick = -1;

// ── Chart setup ─────────────────────────────────────────────────────
const chartOpts = {
  responsive: true,
  animation: { duration: 0 },
  plugins: { legend: { labels: { color: COLORS.text, font: { size: 10 } } } },
  scales: {
    x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.text, font: { size: 9 }, maxTicksLimit: 10 } },
    y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.text, font: { size: 9 } } },
  },
};

const chartLiqExp = new Chart(document.getElementById('chartLiqExp'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Liquidity', data: [], borderColor: COLORS.green, backgroundColor: 'rgba(0,212,170,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 },
      { label: 'Exposure', data: [], borderColor: COLORS.red, borderWidth: 2, borderDash: [5,3], pointRadius: 0, tension: 0.3 },
    ],
  },
  options: chartOpts,
});

const chartMarket = new Chart(document.getElementById('chartMarket'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Volatility', data: [], borderColor: COLORS.yellow, borderWidth: 2, pointRadius: 0, tension: 0.3 },
      { label: 'Margin Rate', data: [], borderColor: COLORS.purple, borderWidth: 2, borderDash: [4,3], pointRadius: 0, tension: 0.3 },
    ],
  },
  options: chartOpts,
});

const chartStatus = new Chart(document.getElementById('chartStatus'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Healthy', data: [], borderColor: COLORS.green, backgroundColor: 'rgba(0,212,170,0.15)', fill: true, borderWidth: 1.5, pointRadius: 0 },
      { label: 'Stressed', data: [], borderColor: COLORS.yellow, backgroundColor: 'rgba(255,193,7,0.15)', fill: true, borderWidth: 1.5, pointRadius: 0 },
      { label: 'Defaulted', data: [], borderColor: COLORS.red, backgroundColor: 'rgba(255,77,106,0.15)', fill: true, borderWidth: 1.5, pointRadius: 0 },
    ],
  },
  options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, stacked: true } } },
});

const chartBankLiq = new Chart(document.getElementById('chartBankLiq'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: chartOpts,
});

const chartCCPPayoff = new Chart(document.getElementById('chartCCPPayoff'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: '+Stability (w1)', data: [], borderColor: '#4caf50', backgroundColor: 'rgba(76,175,80,0.25)', fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.3, order: 3 },
      { label: '+Fund (w2)', data: [], borderColor: '#2196f3', backgroundColor: 'rgba(33,150,243,0.25)', fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.3, order: 2 },
      { label: '−Defaults (w3)', data: [], borderColor: '#d32f2f', backgroundColor: 'rgba(211,47,47,0.25)', fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.3, order: 1 },
      { label: '−Fire Sale (w4)', data: [], borderColor: '#ff9800', backgroundColor: 'rgba(255,152,0,0.25)', fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.3, order: 0 },
      { label: 'Net Utility', data: [], borderColor: '#ffffff', borderWidth: 3, pointRadius: 0, tension: 0.3, fill: false, order: -1 },
    ],
  },
  options: {
    ...chartOpts,
    plugins: {
      ...chartOpts.plugins,
      legend: { labels: { color: COLORS.text, font: { size: 10 }, usePointStyle: true } },
    },
  },
});

// ── Poll + update ───────────────────────────────────────────────────
function pushHist(arr, val) {
  arr.push(val);
  if (arr.length > MAX_POINTS) arr.shift();
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const d = await res.json();
    const sys = d.system;
    const mkt = d.market;
    const banks = d.banks;
    const tick = Math.round(sys.step);
    const nBanks = Math.round(sys.n_banks);
    const nStr = Math.round(sys.n_stressed);
    const nDef = Math.round(sys.n_defaulted);
    const nOk = nBanks - nStr - nDef;

    // Dot
    const dot = document.getElementById('statusDot');
    dot.className = tick > 0 ? 'status-dot live' : 'status-dot stale';
    document.getElementById('headerMeta').textContent =
      `Tick ${tick}  •  ${nBanks} banks  •  ${d.intent_count} intents`;

    // KPIs
    document.getElementById('kpiRow').innerHTML = [
      kpi('Tick', tick, 'green'),
      kpi('Active', nOk, 'blue'),
      kpi('Stressed', nStr, nStr > 0 ? 'yellow' : ''),
      kpi('Defaulted', nDef, nDef > 0 ? 'red' : ''),
      kpi('Margin Rate', sys.margin_rate.toFixed(4), 'purple'),
      kpi('Volatility', mkt.new_volatility.toFixed(4), 'green'),
      kpi('Intents', d.intent_count, 'blue'),
    ].join('');

    // History (only on new tick)
    if (tick > lastTick && tick > 0) {
      lastTick = tick;
      pushHist(hist.tick, tick);
      pushHist(hist.liq, sys.aggregate_liq);
      pushHist(hist.exp, sys.aggregate_exp);
      pushHist(hist.vol, mkt.new_volatility);
      pushHist(hist.mr, sys.margin_rate);
      pushHist(hist.stressed, nStr);
      pushHist(hist.defaulted, nDef);
      pushHist(hist.healthy, nOk);

      for (const [bid, bdata] of Object.entries(banks)) {
        if (!bankHist[bid]) bankHist[bid] = [];
        pushHist(bankHist[bid], bdata.liquidity || 0);
      }

      // CCP payoff components
      const ccp = d.ccp || {};
      pushHist(ccpHist.utility, ccp.utility || 0);
      pushHist(ccpHist.comp_stability, ccp.comp_stability || 0);
      pushHist(ccpHist.comp_fund, ccp.comp_fund || 0);
      pushHist(ccpHist.comp_defaults, -(ccp.comp_defaults || 0));
      pushHist(ccpHist.comp_firesale, -(ccp.comp_firesale || 0));
    }

    // Update charts
    chartLiqExp.data.labels = hist.tick;
    chartLiqExp.data.datasets[0].data = hist.liq;
    chartLiqExp.data.datasets[1].data = hist.exp;
    chartLiqExp.update();

    chartMarket.data.labels = hist.tick;
    chartMarket.data.datasets[0].data = hist.vol;
    chartMarket.data.datasets[1].data = hist.mr;
    chartMarket.update();

    chartStatus.data.labels = hist.tick;
    chartStatus.data.datasets[0].data = hist.healthy;
    chartStatus.data.datasets[1].data = hist.stressed;
    chartStatus.data.datasets[2].data = hist.defaulted;
    chartStatus.update();

    // Per-bank liquidity
    const bids = Object.keys(bankHist).sort((a,b) => +a - +b);
    if (bids.length > 0 && chartBankLiq.data.datasets.length !== bids.length) {
      chartBankLiq.data.datasets = bids.map((bid, i) => ({
        label: `B${bid}`,
        data: bankHist[bid],
        borderColor: BANK_COLORS[i % BANK_COLORS.length],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
      }));
    } else {
      bids.forEach((bid, i) => {
        if (chartBankLiq.data.datasets[i])
          chartBankLiq.data.datasets[i].data = bankHist[bid];
      });
    }
    chartBankLiq.data.labels = hist.tick;
    chartBankLiq.update();

    // CCP payoff chart
    chartCCPPayoff.data.labels = hist.tick;
    chartCCPPayoff.data.datasets[0].data = ccpHist.comp_stability;
    chartCCPPayoff.data.datasets[1].data = ccpHist.comp_fund;
    chartCCPPayoff.data.datasets[2].data = ccpHist.comp_defaults;
    chartCCPPayoff.data.datasets[3].data = ccpHist.comp_firesale;
    chartCCPPayoff.data.datasets[4].data = ccpHist.utility;
    chartCCPPayoff.update();

    // Network graph
    updateNetwork(d.network);

    // Bank table
    const tbody = document.getElementById('bankTableBody');
    tbody.innerHTML = Object.entries(banks)
      .sort(([a],[b]) => +a - +b)
      .map(([bid, b]) => {
        const isDef = b.defaulted > 0;
        const isStr = b.stressed > 0;
        const badge = isDef ? '<span class="badge defaulted">DEFAULT</span>'
                    : isStr ? '<span class="badge stressed">STRESSED</span>'
                    : '<span class="badge ok">OK</span>';
        return `<tr>
          <td>B${bid}</td>
          <td>${b.liquidity?.toFixed(1) ?? '—'}</td>
          <td>${b.capital?.toFixed(1) ?? '—'}</td>
          <td>${b.total_exposure?.toFixed(1) ?? '—'}</td>
          <td>${badge}</td>
        </tr>`;
      }).join('');

    // Intent feed
    const feed = document.getElementById('intentFeed');
    feed.innerHTML = (d.intents || []).reverse().map(it => {
      const vis = it.visibility === 'public' ? '🔓' : '🔒';
      return `<div class="intent-item">
        <span class="tick">t=${it.tick}</span>
        <span class="agent">${it.agent_id}</span>
        <span class="action">${it.action_type}</span>
        <span class="vis">${vis}</span>
      </div>`;
    }).join('') || '<div style="opacity:0.4;padding:8px">Waiting for intents…</div>';

  } catch (e) {
    document.getElementById('statusDot').className = 'status-dot stale';
    document.getElementById('headerMeta').textContent = 'Connection lost — retrying…';
  }
}

function kpi(label, value, colorClass) {
  return `<div class="kpi">
    <div class="label">${label}</div>
    <div class="value ${colorClass}">${value}</div>
  </div>`;
}

// ── Network Graph (vis.js) ──────────────────────────────────────────
const netNodes = new vis.DataSet();
const netEdges = new vis.DataSet();
const netContainer = document.getElementById('networkGraph');
const netData = { nodes: netNodes, edges: netEdges };
const netOptions = {
  physics: {
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -40,
      centralGravity: 0.008,
      springLength: 130,
      springConstant: 0.06,
      damping: 0.35,
      avoidOverlap: 0.4,
    },
    stabilization: { enabled: true, iterations: 80, fit: true },
    maxVelocity: 25,
    minVelocity: 0.3,
    timestep: 0.4,
  },
  interaction: {
    dragNodes: true,
    dragView: true,
    zoomView: true,
    hover: true,
    tooltipDelay: 100,
  },
  nodes: {
    font: { color: '#d0d0d0', size: 12, face: 'Inter, sans-serif' },
    borderWidth: 2,
    shadow: { enabled: true, color: 'rgba(0,0,0,0.4)', size: 8 },
  },
  edges: {
    smooth: { type: 'continuous', roundness: 0.3 },
    font: { color: '#555', size: 8, strokeWidth: 0 },
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
  },
};
const network = new vis.Network(netContainer, netData, netOptions);

const NODE_COLORS = {
  ok:       { background: '#0d3320', border: '#00d4aa', highlight: { background: '#145535', border: '#00ffcc' } },
  stressed: { background: '#3d2e00', border: '#ffc107', highlight: { background: '#5a4400', border: '#ffd54f' } },
  defaulted:{ background: '#3d0d15', border: '#ff4d6a', highlight: { background: '#5c1525', border: '#ff6b8a' } },
  ccp_ok:   { background: '#1a1040', border: '#b197fc', highlight: { background: '#2a1860', border: '#d0bfff' } },
  ccp_panic:{ background: '#4a0020', border: '#ff4d6a', highlight: { background: '#6a0030', border: '#ff6b8a' } },
};

function updateNetwork(net) {
  if (!net || !net.nodes || net.nodes.length === 0) return;

  const incomingNodeIds = new Set();
  const incomingEdgeIds = new Set();

  const nodeUpdates = net.nodes.map(n => {
    const isCCP = n.type === 'ccp';
    const id = isCCP ? 'ccp' : `b${n.id}`;
    incomingNodeIds.add(id);

    let colors, shape, sz, lbl, title;
    if (isCCP) {
      colors = n.status === 'panic' ? NODE_COLORS.ccp_panic : NODE_COLORS.ccp_ok;
      shape = 'diamond';
      sz = 30;
      lbl = 'CCP';
      title = `CCP\nDefault Fund: ${n.default_fund}\nStatus: ${n.status}`;
    } else {
      colors = NODE_COLORS[n.status] || NODE_COLORS.ok;
      shape = 'dot';
      sz = Math.max(10, Math.min(30, 8 + (n.liquidity || 0) / 15));
      lbl = `B${n.id}`;
      title = `Bank ${n.id}\nLiquidity: ${n.liquidity}\nStatus: ${n.status}`;
    }
    return { id, label: lbl, title, shape, size: sz, color: colors };
  });

  const edgeUpdates = net.edges.map((e, i) => {
    const fromId = typeof e.from === 'string' ? e.from : `b${e.from}`;
    const toId   = typeof e.to === 'string' ? e.to : `b${e.to}`;
    const eid = `${fromId}-${toId}-${e.type}`;
    incomingEdgeIds.add(eid);

    const isMargin = e.type === 'margin';
    return {
      id: eid,
      from: fromId,
      to: toId,
      value: e.weight,
      title: isMargin ? `Margin: ${e.weight}%` : `Exposure: ${e.weight}`,
      color: {
        color: isMargin ? 'rgba(177,151,252,0.25)' : 'rgba(0,212,170,0.35)',
        highlight: isMargin ? 'rgba(177,151,252,0.6)' : 'rgba(0,212,170,0.7)',
        hover: isMargin ? 'rgba(177,151,252,0.5)' : 'rgba(0,212,170,0.6)',
      },
      width: isMargin ? 0.5 : Math.max(0.5, Math.min(4, e.weight / 20)),
      dashes: isMargin ? [4, 4] : false,
      arrows: isMargin ? '' : { to: { enabled: true, scaleFactor: 0.4 } },
    };
  });

  const existingNodeIds = netNodes.getIds();
  existingNodeIds.forEach(id => { if (!incomingNodeIds.has(id)) netNodes.remove(id); });
  const existingEdgeIds = netEdges.getIds();
  existingEdgeIds.forEach(id => { if (!incomingEdgeIds.has(id)) netEdges.remove(id); });

  netNodes.update(nodeUpdates);
  netEdges.update(edgeUpdates);
}

setInterval(poll, POLL_MS);
poll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\n  📊 Live Dashboard → http://localhost:8050")
    print(f"  Redis: localhost:6379\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
