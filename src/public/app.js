const form = document.getElementById('optimizer-form');
const errorEl = document.getElementById('error');
const wEurEl = document.getElementById('w-eur');
const wJpyEl = document.getElementById('w-jpy');
const wUsdEl = document.getElementById('w-usd');
const pVolEl = document.getElementById('p-vol');

function formatPct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function optimize(volEurUsd, volUsdJpy, corrInput) {
  if (volEurUsd < 0 || volUsdJpy < 0) {
    throw new Error('Volatilities must be non-negative.');
  }
  if (corrInput < -1 || corrInput > 1) {
    throw new Error('Correlation must be between -1 and 1.');
  }

  if (volEurUsd === 0 && volUsdJpy === 0) {
    throw new Error('Both volatilities are zero. Weights are not unique.');
  }

  if (volEurUsd === 0) {
    return {
      wEur: 1,
      wJpy: 0,
      pVol: 0,
    };
  }

  if (volUsdJpy === 0) {
    return {
      wEur: 0,
      wJpy: 1,
      pVol: 0,
    };
  }

  const sigmaE = volEurUsd;
  const sigmaJ = volUsdJpy;
  const corrEurJpy = -corrInput;

  const a = sigmaE * sigmaE;
  const d = sigmaJ * sigmaJ;
  const b = corrEurJpy * sigmaE * sigmaJ;

  const det = a * d - b * b;
  if (det <= 1e-14) {
    throw new Error('Covariance matrix is singular. Adjust inputs.');
  }

  const inv11 = d / det;
  const inv12 = -b / det;
  const inv22 = a / det;

  const numE = inv11 + inv12;
  const numJ = inv12 + inv22;
  const den = numE + numJ;

  if (Math.abs(den) <= 1e-14) {
    throw new Error('Optimization denominator is zero.');
  }

  const wEur = numE / den;
  const wJpy = numJ / den;
  const pVar =
    wEur * wEur * a + 2 * wEur * wJpy * b + wJpy * wJpy * d;

  return {
    wEur,
    wJpy,
    pVol: Math.sqrt(Math.max(pVar, 0)),
  };
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  errorEl.textContent = '';

  const volEurUsd = Number(document.getElementById('vol-eurusd').value);
  const volUsdJpy = Number(document.getElementById('vol-usdjpy').value);
  const corrInput = Number(document.getElementById('corr').value);

  if ([volEurUsd, volUsdJpy, corrInput].some((v) => Number.isNaN(v))) {
    errorEl.textContent = 'Please fill all fields with numeric values.';
    return;
  }

  try {
    const result = optimize(volEurUsd, volUsdJpy, corrInput);
    wEurEl.textContent = formatPct(result.wEur);
    wJpyEl.textContent = formatPct(result.wJpy);
    wUsdEl.textContent = '0.00%';
    pVolEl.textContent = formatPct(result.pVol);
  } catch (err) {
    errorEl.textContent = err.message;
  }
});
