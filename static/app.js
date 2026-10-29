(() => {
  'use strict';

  const FIELDS = ['Open', 'High', 'Low', 'Volume'];
  const $ = (id) => document.getElementById(id);
  const form = $('predict-form');
  const tickerEl = $('ticker');
  const loadBtn = $('load-btn');
  const predictBtn = $('predict-btn');
  const autoEl = $('auto');
  const STORE_KEY = 'spp_history_v1';

  const state = { ticker: null, history: null, chart: null, seq: 0, timer: null };

  // ---------- helpers ----------
  const num = (el) => (el.value.trim() === '' ? NaN : Number(el.value));
  const fmt = (v, d = 2) => Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const sign = (v) => (v >= 0 ? '+' : '−');

  function showError(msg) {
    const box = $('error');
    box.textContent = msg;
    box.hidden = false;
  }
  function clearError() {
    $('error').hidden = true;
    FIELDS.forEach((f) => $(f).classList.remove('invalid'));
  }

  function setValue(name, value) {
    const input = $(name);
    const slider = $(name + '-range');
    const v = Number(value);
    input.value = name === 'Volume' ? Math.round(v) : v.toFixed(2);
    slider.min = name === 'Volume' ? 0 : (v * 0.8).toFixed(2);
    slider.max = name === 'Volume' ? Math.max(1000, Math.round(v * 2)) : (v * 1.2).toFixed(2);
    slider.step = name === 'Volume' ? 1 : 0.01;
    slider.value = v;
  }

  function setAll(values) {
    FIELDS.forEach((f) => setValue(f, values[f]));
  }

  function readValues() {
    const out = {};
    for (const f of FIELDS) {
      const v = num($(f));
      if (!Number.isFinite(v)) {
        $(f).classList.add('invalid');
        throw new Error(`${f} must be a number.`);
      }
      out[f] = v;
    }
    return out;
  }

  async function api(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      throw new Error('Could not reach the server. Is the app still running?');
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status}).`);
    return data;
  }

  // ---------- chart ----------
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function drawChart(prediction) {
    const wrap = document.querySelector('.chart-wrap');
    const ph = $('chart-placeholder');
    if (!state.history) return;
    if (typeof Chart === 'undefined') {
      ph.textContent = "The chart library couldn't load (check your internet connection).";
      return;
    }
    const { dates, close } = state.history;
    const labels = dates.concat(['Predicted']);
    const hist = close.concat([null]);
    const pred = new Array(labels.length).fill(null);
    if (prediction != null) {
      pred[labels.length - 2] = close[close.length - 1];
      pred[labels.length - 1] = prediction;
    }
    const accent = cssVar('--accent'), muted = cssVar('--muted'), line = cssVar('--line'), down = cssVar('--down');

    if (!state.chart) {
      state.chart = new Chart($('chart'), {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'Close', data: hist, borderColor: accent, backgroundColor: accent, borderWidth: 2, pointRadius: 0, tension: 0.15 },
            { label: 'Predicted close', data: pred, borderColor: down, backgroundColor: down, borderDash: [6, 4],
              borderWidth: 2, pointRadius: (c) => (c.dataIndex === labels.length - 1 ? 6 : 0), spanGaps: true },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
          interaction: { mode: 'index', intersect: false },
          plugins: { legend: { labels: { color: muted } } },
          scales: {
            x: { ticks: { color: muted, maxTicksLimit: 6 }, grid: { color: line } },
            y: { ticks: { color: muted }, grid: { color: line } },
          },
        },
      });
    } else {
      const c = state.chart;
      c.data.labels = labels;
      c.data.datasets[0].data = hist;
      c.data.datasets[1].data = pred;
      c.data.datasets[1].pointRadius = (ctx) => (ctx.dataIndex === labels.length - 1 ? 6 : 0);
      c.update();
    }
    wrap.classList.add('ready');
  }

  // ---------- result ----------
  function showResult(r) {
    $('pred').textContent = fmt(r.prediction);
    const delta = $('delta');
    delta.textContent = `${sign(r.change)}${fmt(Math.abs(r.change))} (${sign(r.change_pct)}${Math.abs(r.change_pct)}%) vs. open`;
    delta.className = 'delta ' + (r.change >= 0 ? 'up' : 'down');

    const info = r.model === 'default'
      ? 'Built-in Ridge model (trained on the original project data).'
      : `Ridge model trained live on ${r.model} price history.`;
    $('model-info').textContent = info + (r.note ? ' ' + r.note : '');

    const m = $('metrics');
    m.textContent = '';
    if (r.metrics) {
      [['R² (held-out)', fmt(r.metrics.r2, 3)], ['MAE', fmt(r.metrics.mae)], ['RMSE', fmt(r.metrics.rmse)],
       ['Test days', r.metrics.test_rows]].forEach(([k, v]) => {
        const d = document.createElement('div');
        d.className = 'metric';
        d.append(k, document.createElement('b'));
        d.lastChild.textContent = v;
        m.append(d);
      });
      m.hidden = false;
    } else {
      m.hidden = true;
    }
    const card = $('result-card');
    card.classList.remove('pop');
    void card.offsetWidth;
    card.classList.add('pop');
  }

  // ---------- history (localStorage, optional) ----------
  function loadStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; } catch (e) { return []; }
  }
  function saveStore(rows) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(rows)); } catch (e) { /* storage blocked */ }
  }
  let rows = loadStore();

  function renderHistory() {
    const body = document.querySelector('#history-table tbody');
    body.textContent = '';
    rows.forEach((r) => {
      const tr = document.createElement('tr');
      [r.model === 'default' ? 'Built-in' : r.model, fmt(r.Open), fmt(r.High), fmt(r.Low), fmt(r.Volume, 0), fmt(r.prediction)]
        .forEach((t) => { const td = document.createElement('td'); td.textContent = t; tr.append(td); });
      body.append(tr);
    });
    $('history-table').classList.toggle('has-rows', rows.length > 0);
    $('history-empty').hidden = rows.length > 0;
  }

  function remember(r) {
    const last = rows[0];
    const row = { model: r.model, ...r.inputs, prediction: r.prediction };
    if (last && JSON.stringify(last) === JSON.stringify(row)) return;
    rows = [row, ...rows].slice(0, 8);
    saveStore(rows);
    renderHistory();
  }

  // ---------- actions ----------
  async function predict({ remember: keep = true } = {}) {
    clearError();
    let values;
    try { values = readValues(); } catch (e) { showError(e.message); return; }

    const ticker = tickerEl.value.trim();
    const mySeq = ++state.seq;
    predictBtn.disabled = true;
    try {
      if (ticker && ticker.toUpperCase() !== state.ticker) {
        await loadTicker(false).catch(() => {}); // chart only; errors are reported by the prediction note
      }
      const r = await api('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...values, ticker }),
      });
      if (mySeq !== state.seq) return; // a newer request superseded this one
      showResult(r);
      drawChart(r.prediction);
      if (keep) remember(r);
    } catch (e) {
      if (mySeq === state.seq) showError(e.message);
    } finally {
      predictBtn.disabled = false;
    }
  }

  async function loadTicker(fillInputs = true) {
    clearError();
    const t = tickerEl.value.trim();
    if (!t) { showError('Type a ticker symbol first, e.g. AAPL.'); return; }
    loadBtn.disabled = true;
    const old = loadBtn.textContent;
    loadBtn.textContent = 'Loading…';
    try {
      const d = await api('/api/stock/' + encodeURIComponent(t));
      state.ticker = d.ticker;
      state.history = d.history;
      tickerEl.value = d.ticker;
      if (fillInputs) {
        setAll(d.latest);
        $('model-info').textContent = `Loaded ${d.ticker} for ${d.latest.date} (latest close ${fmt(d.latest.Close)}).`;
        drawChart(null);
        await predict();
      } else {
        drawChart(null);
      }
    } catch (e) {
      if (fillInputs) showError(e.message);
      else throw e;
    } finally {
      loadBtn.disabled = false;
      loadBtn.textContent = old;
    }
  }

  function schedule() {
    if (!autoEl.checked) return;
    clearTimeout(state.timer);
    state.timer = setTimeout(() => predict({ remember: false }), 300);
  }

  // ---------- wiring ----------
  FIELDS.forEach((f) => {
    const input = $(f), slider = $(f + '-range');
    slider.addEventListener('input', () => {
      input.value = f === 'Volume' ? Math.round(slider.value) : Number(slider.value).toFixed(2);
      schedule();
    });
    input.addEventListener('input', () => {
      input.classList.remove('invalid');
      const v = num(input);
      if (Number.isFinite(v)) slider.value = v;
      schedule();
    });
    const v = num(input);
    if (Number.isFinite(v)) setValue(f, v);
  });

  form.addEventListener('submit', (e) => { e.preventDefault(); clearTimeout(state.timer); predict(); });
  loadBtn.addEventListener('click', () => loadTicker(true));
  tickerEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); loadTicker(true); } });
  document.querySelectorAll('.chip').forEach((c) =>
    c.addEventListener('click', () => { tickerEl.value = c.dataset.ticker; loadTicker(true); }));
  $('sample-btn').addEventListener('click', () => {
    tickerEl.value = '';
    state.ticker = null;
    setAll(window.SPP_DEFAULTS);
    predict();
  });
  $('clear-history').addEventListener('click', () => { rows = []; saveStore(rows); renderHistory(); });

  renderHistory();
})();
