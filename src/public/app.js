const form = document.getElementById('optimizer-form');
const errorEl = document.getElementById('error');
const wEurEl = document.getElementById('w-eur');
const wJpyEl = document.getElementById('w-jpy');
const wUsdEl = document.getElementById('w-usd');
const pVolEl = document.getElementById('p-vol');
const longOnlyEl = document.getElementById('long-only');
const constraintNoteEl = document.getElementById('constraint-note');
const resetBtn = document.getElementById('reset-btn');

function formatPct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function optimize(volEurUsd, volUsdJpy, corrInput, longOnly) {
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

  const denom = a + d - 2 * b;
  let wEur;
  if (Math.abs(denom) <= 1e-14) {
    wEur = 0.5;
  } else {
    wEur = (d - b) / denom;
  }

  if (longOnly) {
    wEur = Math.max(0, Math.min(1, wEur));
  }

  const wJpy = 1 - wEur;
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
  const longOnly = longOnlyEl.checked;

  if ([volEurUsd, volUsdJpy, corrInput].some((v) => Number.isNaN(v))) {
    errorEl.textContent = 'Please fill all fields with numeric values.';
    return;
  }

  try {
    const result = optimize(volEurUsd, volUsdJpy, corrInput, longOnly);
    wEurEl.textContent = formatPct(result.wEur);
    wJpyEl.textContent = formatPct(result.wJpy);
    wUsdEl.textContent = '0.00%';
    pVolEl.textContent = formatPct(result.pVol);
    constraintNoteEl.textContent = longOnly
      ? 'Constraint: Long only'
      : 'Constraint: None (shorting allowed)';
  } catch (err) {
    errorEl.textContent = err.message;
  }
});

resetBtn.addEventListener('click', () => {
  form.reset();
  longOnlyEl.checked = true;
  errorEl.textContent = '';
  wEurEl.textContent = '-';
  wJpyEl.textContent = '-';
  wUsdEl.textContent = '0.00%';
  pVolEl.textContent = '-';
  constraintNoteEl.textContent = 'Constraint: Long only';
});
