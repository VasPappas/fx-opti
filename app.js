function parseNumber(id) {
  const value = Number(document.getElementById(id).value);
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid number in ${id}`);
  }
  return value;
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function matVec(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
  ];
}

function inv3(m) {
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h = m[2][1], i = m[2][2];

  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;

  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) {
    throw new Error("Covariance matrix is singular or nearly singular.");
  }

  return [
    [A / det, D / det, G / det],
    [B / det, E / det, H / det],
    [C / det, F / det, I / det]
  ];
}

function portfolioStats(w, mu, cov, rf) {
  const ret = dot(w, mu);
  const varTerm = dot(w, matVec(cov, w));
  const vol = Math.sqrt(Math.max(varTerm, 0));
  const sharpe = vol > 0 ? (ret - rf) / vol : -Infinity;
  return { w, ret, vol, sharpe };
}

function optimizeUnconstrained(mu, cov, rf) {
  const excess = [mu[0] - rf, mu[1] - rf, mu[2] - rf];
  const inv = inv3(cov);
  const raw = matVec(inv, excess);
  const s = raw[0] + raw[1] + raw[2];
  if (Math.abs(s) < 1e-12) {
    throw new Error("Degenerate normalization denominator.");
  }
  const w = [raw[0] / s, raw[1] / s, raw[2] / s];
  return portfolioStats(w, mu, cov, rf);
}

function optimizeLongOnly(mu, cov, rf, gridN) {
  let best = { w: [1, 0, 0], ret: -Infinity, vol: Infinity, sharpe: -Infinity };

  for (let i = 0; i <= gridN; i += 1) {
    const w1 = i / gridN;
    for (let j = 0; j <= gridN - i; j += 1) {
      const w2 = j / gridN;
      const w3 = 1 - w1 - w2;
      const stats = portfolioStats([w1, w2, w3], mu, cov, rf);
      if (stats.sharpe > best.sharpe) {
        best = stats;
      }
    }
  }

  return best;
}

function buildCov(vols, corr) {
  const cov = [
    [vols[0] * vols[0], corr[0][1] * vols[0] * vols[1], corr[0][2] * vols[0] * vols[2]],
    [corr[1][0] * vols[1] * vols[0], vols[1] * vols[1], corr[1][2] * vols[1] * vols[2]],
    [corr[2][0] * vols[2] * vols[0], corr[2][1] * vols[2] * vols[1], vols[2] * vols[2]]
  ];
  return cov;
}

function readInputs() {
  const mu = [parseNumber("mu1"), parseNumber("mu2"), parseNumber("mu3")].map(v => v / 100);
  const vol = [parseNumber("vol1"), parseNumber("vol2"), parseNumber("vol3")].map(v => Math.max(v, 0) / 100);
  const rf = parseNumber("rf") / 100;

  const corr = [
    [1, parseNumber("c12"), parseNumber("c13")],
    [parseNumber("c21"), 1, parseNumber("c23")],
    [parseNumber("c31"), parseNumber("c32"), 1]
  ];

  for (const row of corr) {
    for (const c of row) {
      if (c < -1 || c > 1) {
        throw new Error("Correlations must be between -1 and 1.");
      }
    }
  }

  if (vol.some(v => v === 0)) {
    throw new Error("Volatility must be > 0 for all assets.");
  }

  const cov = buildCov(vol, corr);
  return { mu, cov, rf };
}

function pct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

function renderResult(result) {
  document.getElementById("outReturn").textContent = pct(result.ret);
  document.getElementById("outVol").textContent = pct(result.vol);
  document.getElementById("outSharpe").textContent = result.sharpe.toFixed(4);

  ["1", "2", "3"].forEach((n, idx) => {
    document.getElementById(`w${n}`).style.width = `${Math.max(0, result.w[idx] * 100).toFixed(2)}%`;
    document.getElementById(`w${n}v`).textContent = pct(result.w[idx]);
  });
}

function drawFrontier(mu, cov, rf, optimum) {
  const canvas = document.getElementById("frontier");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#fffef8";
  ctx.fillRect(0, 0, W, H);

  const points = [];
  const n = 90;
  for (let i = 0; i <= n; i += 1) {
    const w1 = i / n;
    for (let j = 0; j <= n - i; j += 1) {
      const w2 = j / n;
      const w3 = 1 - w1 - w2;
      points.push(portfolioStats([w1, w2, w3], mu, cov, rf));
    }
  }

  const vols = points.map(p => p.vol);
  const rets = points.map(p => p.ret);
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
    ctx.fillStyle = "rgba(11, 110, 104, 0.35)";
    ctx.beginPath();
    ctx.arc(xScale(p.vol), yScale(p.ret), 2.2, 0, 2 * Math.PI);
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
    const gridN = Number(document.getElementById("grid").value);

    const result = mode === "long_only"
      ? optimizeLongOnly(mu, cov, rf, gridN)
      : optimizeUnconstrained(mu, cov, rf);

    renderResult(result);
    drawFrontier(mu, cov, rf, result);
  } catch (err) {
    errorEl.textContent = err.message || "Optimization failed.";
  }
}

const gridInput = document.getElementById("grid");
gridInput.addEventListener("input", () => {
  document.getElementById("gridValue").textContent = gridInput.value;
});

document.getElementById("optimizeBtn").addEventListener("click", runOptimization);

["c12", "c21", "c13", "c31", "c23", "c32"].forEach((id) => {
  document.getElementById(id).addEventListener("change", () => {
    const v = document.getElementById(id).value;
    const map = { c12: "c21", c21: "c12", c13: "c31", c31: "c13", c23: "c32", c32: "c23" };
    document.getElementById(map[id]).value = v;
  });
});

runOptimization();
