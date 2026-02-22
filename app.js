const MIN_ASSETS = 2;
const MAX_ASSETS = 15;

const state = {
  n: 3,
  names: ["Asset A", "Asset B", "Asset C"],
  muPct: [8, 12, 10],
  volPct: [15, 22, 18],
  corr: [
    [1, 0.3, 0.45],
    [0.3, 1, 0.4],
    [0.45, 0.4, 1]
  ],
  lastResult: null
};

function defaultAssetName(index) {
  const letter = String.fromCharCode(65 + (index % 26));
  const cycle = Math.floor(index / 26);
  return cycle === 0 ? `Asset ${letter}` : `Asset ${letter}${cycle + 1}`;
}

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function shortName(name) {
  return name.length <= 12 ? name : `${name.slice(0, 11)}.`;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

function matVec(m, v) {
  const n = v.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    let s = 0;
    for (let j = 0; j < n; j += 1) {
      s += m[i][j] * v[j];
    }
    out[i] = s;
  }
  return out;
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) {
        pivot = r;
      }
    }

    if (Math.abs(aug[pivot][col]) < 1e-12) {
      throw new Error("Covariance matrix is singular or nearly singular.");
    }

    if (pivot !== col) {
      [aug[pivot], aug[col]] = [aug[col], aug[pivot]];
    }

    const pivotVal = aug[col][col];
    for (let k = col; k <= n; k += 1) {
      aug[col][k] /= pivotVal;
    }

    for (let r = 0; r < n; r += 1) {
      if (r === col) {
        continue;
      }
      const factor = aug[r][col];
      for (let k = col; k <= n; k += 1) {
        aug[r][k] -= factor * aug[col][k];
      }
    }
  }

  return aug.map((row) => row[n]);
}

function projectToSimplex(v) {
  const u = [...v].sort((a, b) => b - a);
  let cssv = 0;
  let rho = -1;

  for (let i = 0; i < u.length; i += 1) {
    cssv += u[i];
    const t = (cssv - 1) / (i + 1);
    if (u[i] - t > 0) {
      rho = i;
    }
  }

  if (rho < 0) {
    return new Array(v.length).fill(1 / v.length);
  }

  const theta = (u.slice(0, rho + 1).reduce((a, x) => a + x, 0) - 1) / (rho + 1);
  return v.map((x) => Math.max(x - theta, 0));
}

function portfolioStats(w, mu, cov, rf) {
  const ret = dot(w, mu);
  const covW = matVec(cov, w);
  const variance = Math.max(dot(w, covW), 0);
  const vol = Math.sqrt(variance);
  const sharpe = vol > 0 ? (ret - rf) / vol : -Infinity;
  return { w, ret, vol, sharpe, covW };
}

function optimizeUnconstrained(mu, cov, rf) {
  const n = mu.length;
  const excess = mu.map((m) => m - rf);
  const raw = solveLinearSystem(cov, excess);
  const denom = raw.reduce((a, x) => a + x, 0);

  if (Math.abs(denom) < 1e-12) {
    throw new Error("Degenerate unconstrained solution (normalization denominator is zero).");
  }

  const w = raw.map((x) => x / denom);
  return portfolioStats(w, mu, cov, rf);
}

function optimizeLongOnly(mu, cov, rf, maxIter) {
  const n = mu.length;
  let w = new Array(n).fill(1 / n);
  let best = portfolioStats(w, mu, cov, rf);
  let step = 0.2;

  for (let iter = 0; iter < maxIter; iter += 1) {
    const vol = Math.max(best.vol, 1e-10);
    const vol3 = vol * vol * vol;
    const gradient = new Array(n);

    for (let i = 0; i < n; i += 1) {
      gradient[i] = (mu[i] / vol) - (((best.ret - rf) * best.covW[i]) / vol3);
    }

    let improved = false;
    let localStep = step;

    for (let trial = 0; trial < 10; trial += 1) {
      const candidate = projectToSimplex(w.map((wi, i) => wi + localStep * gradient[i]));
      const candStats = portfolioStats(candidate, mu, cov, rf);

      if (candStats.sharpe > best.sharpe + 1e-9) {
        w = candidate;
        best = candStats;
        step = Math.min(localStep * 1.2, 1);
        improved = true;
        break;
      }

      localStep *= 0.5;
    }

    if (!improved) {
      step *= 0.6;
      if (step < 1e-5) {
        break;
      }
    }
  }

  return best;
}

function pct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

function buildCov(vol, corr) {
  const n = vol.length;
  const cov = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      cov[i][j] = corr[i][j] * vol[i] * vol[j];
    }
  }

  return cov;
}

function getRandomSimplexWeights(n) {
  const x = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const u = Math.max(Math.random(), 1e-12);
    x[i] = -Math.log(u);
    sum += x[i];
  }
  return x.map((v) => v / sum);
}

function resizeState(newN) {
  const oldN = state.n;
  const names = new Array(newN);
  const muPct = new Array(newN);
  const volPct = new Array(newN);
  const corr = Array.from({ length: newN }, () => new Array(newN).fill(0));

  for (let i = 0; i < newN; i += 1) {
    names[i] = i < oldN ? state.names[i] : defaultAssetName(i);
    muPct[i] = i < oldN ? state.muPct[i] : 8 + (i % 5);
    volPct[i] = i < oldN ? state.volPct[i] : 15 + (i % 6) * 2;
  }

  for (let i = 0; i < newN; i += 1) {
    for (let j = 0; j < newN; j += 1) {
      if (i === j) {
        corr[i][j] = 1;
      } else if (i < oldN && j < oldN) {
        corr[i][j] = state.corr[i][j];
      } else {
        corr[i][j] = 0.25;
      }
    }
  }

  state.n = newN;
  state.names = names;
  state.muPct = muPct;
  state.volPct = volPct;
  state.corr = corr;
}

function renderNameList() {
  const wrap = document.getElementById("nameList");
  wrap.innerHTML = "";

  for (let i = 0; i < state.n; i += 1) {
    const label = document.createElement("label");
    label.innerHTML = `
      Asset ${i + 1}
      <input class="name-input" data-i="${i}" type="text" maxlength="24" value="${state.names[i]}" />
    `;
    wrap.appendChild(label);
  }
}

function renderAssetRows() {
  const tbody = document.getElementById("assetRows");
  tbody.innerHTML = "";

  for (let i = 0; i < state.n; i += 1) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="asset-name-in" data-i="${i}">${state.names[i]}</td>
      <td><input class="mu-input" data-i="${i}" type="number" step="0.1" value="${state.muPct[i]}" /></td>
      <td><input class="vol-input" data-i="${i}" type="number" step="0.1" min="0" value="${state.volPct[i]}" /></td>
    `;
    tbody.appendChild(tr);
  }
}

function renderCorrGrid() {
  const grid = document.getElementById("corrGrid");
  grid.style.setProperty("--asset-count", String(state.n));
  grid.innerHTML = "";

  const corner = document.createElement("span");
  corner.className = "corr-label corner";
  corner.textContent = "\\";
  grid.appendChild(corner);

  for (let j = 0; j < state.n; j += 1) {
    const el = document.createElement("span");
    el.className = "corr-label col";
    el.dataset.i = String(j);
    el.textContent = shortName(state.names[j]);
    grid.appendChild(el);
  }

  for (let i = 0; i < state.n; i += 1) {
    const rowLabel = document.createElement("span");
    rowLabel.className = "corr-label row";
    rowLabel.dataset.i = String(i);
    rowLabel.textContent = shortName(state.names[i]);
    grid.appendChild(rowLabel);

    for (let j = 0; j < state.n; j += 1) {
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.01";
      input.min = "-1";
      input.max = "1";
      input.id = `c-${i}-${j}`;
      input.className = "corr-input";
      input.dataset.i = String(i);
      input.dataset.j = String(j);
      input.value = state.corr[i][j].toFixed(2);
      input.ariaLabel = `Correlation ${state.names[i]} with ${state.names[j]}`;
      if (i === j) {
        input.disabled = true;
      }
      grid.appendChild(input);
    }
  }
}

function columnOffDiagAverage(col) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < state.n; i += 1) {
    if (i === col) {
      continue;
    }
    sum += state.corr[i][col];
    count += 1;
  }
  return count > 0 ? sum / count : 0;
}

function renderCorrBulk() {
  const bulk = document.getElementById("corrBulk");
  bulk.style.setProperty("--asset-count", String(state.n));
  bulk.innerHTML = "";

  const label = document.createElement("span");
  label.className = "corr-bulk-label";
  label.textContent = "Set Column";
  bulk.appendChild(label);

  for (let col = 0; col < state.n; col += 1) {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.01";
    input.min = "-1";
    input.max = "1";
    input.className = "corr-bulk-input";
    input.dataset.col = String(col);
    input.value = columnOffDiagAverage(col).toFixed(2);
    input.title = `Set all correlations for ${state.names[col]}`;
    input.ariaLabel = `Set full column correlations for ${state.names[col]}`;
    bulk.appendChild(input);
  }
}

function renderWeights(result) {
  const wrap = document.getElementById("weights");
  wrap.innerHTML = "";

  for (let i = 0; i < state.n; i += 1) {
    const wPct = result.w[i] * 100;
    const hue = (i * 37) % 360;

    const row = document.createElement("div");
    row.className = "w-row";
    row.innerHTML = `<span>${state.names[i]}</span><span>${pct(result.w[i])}</span>`;

    const bar = document.createElement("div");
    bar.className = "bar";

    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${Math.max(wPct, 0).toFixed(2)}%`;
    fill.style.background = `hsl(${hue}deg 58% 42%)`;

    bar.appendChild(fill);
    wrap.appendChild(row);
    wrap.appendChild(bar);
  }
}

function syncNameLabels() {
  document.querySelectorAll(".asset-name-in").forEach((el) => {
    const i = Number(el.dataset.i);
    el.textContent = state.names[i];
  });

  document.querySelectorAll(".corr-label.col").forEach((el) => {
    const i = Number(el.dataset.i);
    el.textContent = shortName(state.names[i]);
  });

  document.querySelectorAll(".corr-label.row").forEach((el) => {
    const i = Number(el.dataset.i);
    el.textContent = shortName(state.names[i]);
  });

  document.querySelectorAll(".corr-input").forEach((el) => {
    const i = Number(el.dataset.i);
    const j = Number(el.dataset.j);
    el.ariaLabel = `Correlation ${state.names[i]} with ${state.names[j]}`;
  });

  renderCorrBulk();

  if (state.lastResult) {
    renderWeights(state.lastResult);
  }
}

function readInputs() {
  const n = state.n;
  const mu = new Array(n);
  const vol = new Array(n);
  const corr = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i += 1) {
    const muEl = document.querySelector(`.mu-input[data-i="${i}"]`);
    const volEl = document.querySelector(`.vol-input[data-i="${i}"]`);
    const muVal = Number(muEl.value);
    const volVal = Number(volEl.value);

    if (!Number.isFinite(muVal) || !Number.isFinite(volVal)) {
      throw new Error(`Invalid expected return or volatility for ${state.names[i]}.`);
    }

    if (volVal <= 0) {
      throw new Error(`Volatility must be > 0 for ${state.names[i]}.`);
    }

    state.muPct[i] = muVal;
    state.volPct[i] = volVal;

    mu[i] = muVal / 100;
    vol[i] = volVal / 100;
  }

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (i === j) {
        corr[i][j] = 1;
        continue;
      }

      if (j < i) {
        corr[i][j] = corr[j][i];
        continue;
      }

      const el = document.getElementById(`c-${i}-${j}`);
      const v = Number(el.value);
      if (!Number.isFinite(v) || v < -1 || v > 1) {
        throw new Error(`Correlation between ${state.names[i]} and ${state.names[j]} must be between -1 and 1.`);
      }

      corr[i][j] = v;
      corr[j][i] = v;
      state.corr[i][j] = v;
      state.corr[j][i] = v;
    }
  }

  const rf = Number(document.getElementById("rf").value) / 100;
  if (!Number.isFinite(rf)) {
    throw new Error("Invalid risk-free rate.");
  }

  return { mu, cov: buildCov(vol, corr), rf };
}

function drawFrontier(mu, cov, rf, optimum) {
  const canvas = document.getElementById("frontier");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#fffef8";
  ctx.fillRect(0, 0, W, H);

  const n = mu.length;
  const sampleCount = n <= 5 ? 3500 : 2200;
  const points = [];

  for (let i = 0; i < n; i += 1) {
    const w = new Array(n).fill(0);
    w[i] = 1;
    points.push(portfolioStats(w, mu, cov, rf));
  }

  for (let i = 0; i < sampleCount; i += 1) {
    points.push(portfolioStats(getRandomSimplexWeights(n), mu, cov, rf));
  }

  const vols = points.map((p) => p.vol);
  const rets = points.map((p) => p.ret);
  const minX = Math.min(...vols, optimum.vol);
  const maxX = Math.max(...vols, optimum.vol);
  const minY = Math.min(...rets, optimum.ret);
  const maxY = Math.max(...rets, optimum.ret);

  const pad = 34;
  const xScale = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const yScale = (y) => H - pad - ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);

  ctx.strokeStyle = "#b7a88b";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad);
  ctx.moveTo(pad, H - pad);
  ctx.lineTo(pad, pad);
  ctx.stroke();

  for (const p of points) {
    ctx.fillStyle = "rgba(11, 110, 104, 0.28)";
    ctx.beginPath();
    ctx.arc(xScale(p.vol), yScale(p.ret), 1.9, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.fillStyle = "#e07a2f";
  ctx.beginPath();
  ctx.arc(xScale(optimum.vol), yScale(optimum.ret), 5, 0, 2 * Math.PI);
  ctx.fill();

  ctx.fillStyle = "#333";
  ctx.font = "12px Space Grotesk";
  ctx.fillText("Volatility", W - 80, H - 12);
  ctx.save();
  ctx.translate(12, 60);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Expected Return", 0, 0);
  ctx.restore();
}

function runOptimization() {
  const errorEl = document.getElementById("error");
  errorEl.textContent = "";

  try {
    const { mu, cov, rf } = readInputs();
    const mode = document.getElementById("mode").value;
    const iters = Number(document.getElementById("iters").value);

    const result = mode === "long_only"
      ? optimizeLongOnly(mu, cov, rf, iters)
      : optimizeUnconstrained(mu, cov, rf);

    state.lastResult = result;
    document.getElementById("outReturn").textContent = pct(result.ret);
    document.getElementById("outVol").textContent = pct(result.vol);
    document.getElementById("outSharpe").textContent = result.sharpe.toFixed(4);

    renderWeights(result);
    drawFrontier(mu, cov, rf, result);
  } catch (err) {
    errorEl.textContent = err.message || "Optimization failed.";
  }
}

function renderDynamicSections() {
  renderNameList();
  renderAssetRows();
  renderCorrBulk();
  renderCorrGrid();
  if (state.lastResult && state.lastResult.w.length === state.n) {
    renderWeights(state.lastResult);
  } else {
    document.getElementById("weights").innerHTML = "";
  }
}

function setupAssetCount() {
  const select = document.getElementById("assetCount");
  for (let n = MIN_ASSETS; n <= MAX_ASSETS; n += 1) {
    const opt = document.createElement("option");
    opt.value = String(n);
    opt.textContent = String(n);
    if (n === state.n) {
      opt.selected = true;
    }
    select.appendChild(opt);
  }

  select.addEventListener("change", () => {
    const newN = clamp(Number(select.value), MIN_ASSETS, MAX_ASSETS);
    resizeState(newN);
    state.lastResult = null;
    renderDynamicSections();
    runOptimization();
  });
}

document.getElementById("nameList").addEventListener("input", (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLInputElement) || !target.classList.contains("name-input")) {
    return;
  }
  const i = Number(target.dataset.i);
  state.names[i] = target.value.trim() || defaultAssetName(i);
  if (!target.value.trim()) {
    target.value = state.names[i];
  }
  syncNameLabels();
});

document.getElementById("assetRows").addEventListener("input", (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLInputElement)) {
    return;
  }

  const i = Number(target.dataset.i);
  if (target.classList.contains("mu-input")) {
    state.muPct[i] = Number(target.value);
  }
  if (target.classList.contains("vol-input")) {
    state.volPct[i] = Number(target.value);
  }
});

document.getElementById("corrGrid").addEventListener("input", (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLInputElement) || !target.classList.contains("corr-input")) {
    return;
  }

  const i = Number(target.dataset.i);
  const j = Number(target.dataset.j);
  if (i === j) {
    return;
  }

  const v = clamp(Number(target.value), -1, 1);
  target.value = v.toFixed(2);
  state.corr[i][j] = v;
  state.corr[j][i] = v;

  const mirror = document.getElementById(`c-${j}-${i}`);
  if (mirror && mirror !== target) {
    mirror.value = v.toFixed(2);
  }

  renderCorrBulk();
});

document.getElementById("corrBulk").addEventListener("change", (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLInputElement) || !target.classList.contains("corr-bulk-input")) {
    return;
  }

  const col = Number(target.dataset.col);
  const value = clamp(Number(target.value), -1, 1);
  target.value = value.toFixed(2);

  for (let row = 0; row < state.n; row += 1) {
    if (row === col) {
      continue;
    }

    state.corr[row][col] = value;
    state.corr[col][row] = value;

    const a = document.getElementById(`c-${row}-${col}`);
    const b = document.getElementById(`c-${col}-${row}`);
    if (a) {
      a.value = value.toFixed(2);
    }
    if (b) {
      b.value = value.toFixed(2);
    }
  }
});

const itersInput = document.getElementById("iters");
document.getElementById("itersValue").textContent = itersInput.value;
itersInput.addEventListener("input", () => {
  document.getElementById("itersValue").textContent = itersInput.value;
});

document.getElementById("optimizeBtn").addEventListener("click", runOptimization);

setupAssetCount();
renderDynamicSections();
runOptimization();
