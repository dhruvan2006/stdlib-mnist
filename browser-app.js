let model = null;
let inferenceScheduled = false;

const TWO_PI = 2 * PI;

function getMathApi() {
  if (typeof globalThis !== 'undefined' && globalThis.StdlibMnistMath) {
    return globalThis.StdlibMnistMath;
  }
  throw new Error('StdlibMnistMath is not loaded. Ensure shared-math.js is included before browser-app.js.');
}

function mlpForward(input, weights, biases, sizes) {
  const mathApi = getMathApi();
  let a = new Float64Array(input);

  for (let l = 0; l < sizes.length - 1; l++) {
    const m = sizes[l + 1];
    const n = sizes[l];
    const z = new Float64Array(biases[l]);

    // z = 1.0 * W * a + 1.0 * z
    dgemv('row-major', 'no-transpose', m, n, 1.0, weights[l], n, a, 1, 1.0, z, 1);

    a = new Float64Array(m);
    const isOutputLayer = l === sizes.length - 2;
    for (let j = 0; j < m; j++) {
      a[j] = isOutputLayer ? z[j] : mathApi.relu(z[j]);
    }
  }

  return a;
}

function applyLoadedModel(data) {
  if (!data || !Array.isArray(data.sizes) || !Array.isArray(data.weights) || !Array.isArray(data.biases)) {
    throw new Error('Invalid model.json format');
  }

  model = {
    sizes: data.sizes,
    weights: data.weights.map((w) => new Float64Array(w)),
    biases: data.biases.map((b) => new Float64Array(b))
  };
}

async function loadLocalModel() {
  try {
    const response = await fetch('model.json', { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`Could not load model.json (HTTP ${response.status})`);
    }

    const data = await response.json();
    applyLoadedModel(data);
  } catch (err) {
    console.error('Failed to load model:', err);
  }
}

const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');
const preview = document.getElementById('preview-canvas');
const pctx = preview.getContext('2d');

let drawing = false;
let lastX = 0;
let lastY = 0;

ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 280, 280);
ctx.strokeStyle = '#fff';
ctx.shadowBlur = 3;
ctx.shadowColor = '#fff';
ctx.lineWidth = 22;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  if (e.touches) {
    return [
      (e.touches[0].clientX - rect.left) * scaleX,
      (e.touches[0].clientY - rect.top) * scaleY
    ];
  }

  return [
    (e.clientX - rect.left) * scaleX,
    (e.clientY - rect.top) * scaleY
  ];
}

function updatePreview() {
  pctx.drawImage(canvas, 0, 0, 28, 28);
}

function getPixels() {
  const src = document.createElement('canvas');
  src.width = src.height = 28;
  const sctx = src.getContext('2d');
  sctx.drawImage(canvas, 0, 0, 28, 28);

  const data = sctx.getImageData(0, 0, 28, 28);
  const gray = new Float64Array(784);

  for (let i = 0; i < 784; i++) {
    gray[i] = data.data[i * 4] / 255.0;
  }

  let minX = 27;
  let minY = 27;
  let maxX = 0;
  let maxY = 0;
  let mass = 0;
  let mx = 0;
  let my = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = gray[y * 28 + x];
      if (v > 0.05) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
      mass += v;
      mx += x * v;
      my += y * v;
    }
  }

  if (mass < 1e-6) {
    return new Float64Array(784);
  }

  const boxW = max(1, maxX - minX + 1);
  const boxH = max(1, maxY - minY + 1);
  const scale = 20 / max(boxW, boxH);
  const drawW = max(1, round(boxW * scale));
  const drawH = max(1, round(boxH * scale));

  const crop = document.createElement('canvas');
  crop.width = boxW;
  crop.height = boxH;
  const cctx = crop.getContext('2d');
  cctx.putImageData(sctx.getImageData(minX, minY, boxW, boxH), 0, 0);

  const norm = document.createElement('canvas');
  norm.width = norm.height = 28;
  const nctx = norm.getContext('2d');
  nctx.fillStyle = '#000';
  nctx.fillRect(0, 0, 28, 28);

  const offsetX = floor((28 - drawW) / 2);
  const offsetY = floor((28 - drawH) / 2);
  nctx.drawImage(crop, 0, 0, boxW, boxH, offsetX, offsetY, drawW, drawH);

  const normData = nctx.getImageData(0, 0, 28, 28);
  let nMass = 0;
  let nMx = 0;
  let nMy = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = normData.data[(y * 28 + x) * 4] / 255.0;
      nMass += v;
      nMx += x * v;
      nMy += y * v;
    }
  }

  if (nMass > 1e-6) {
    const cx = nMx / nMass;
    const cy = nMy / nMass;
    const shiftX = round(13.5 - cx);
    const shiftY = round(13.5 - cy);

    const shifted = document.createElement('canvas');
    shifted.width = shifted.height = 28;
    const shctx = shifted.getContext('2d');
    shctx.fillStyle = '#000';
    shctx.fillRect(0, 0, 28, 28);
    shctx.drawImage(norm, shiftX, shiftY);

    const out = shctx.getImageData(0, 0, 28, 28).data;
    const pixels = new Float64Array(784);
    for (let i = 0; i < 784; i++) {
      pixels[i] = out[i * 4] / 255.0;
    }

    pctx.clearRect(0, 0, 28, 28);
    pctx.drawImage(shifted, 0, 0);
    return pixels;
  }

  const out = nctx.getImageData(0, 0, 28, 28).data;
  const pixels = new Float64Array(784);
  for (let i = 0; i < 784; i++) {
    pixels[i] = out[i * 4] / 255.0;
  }

  pctx.clearRect(0, 0, 28, 28);
  pctx.drawImage(norm, 0, 0);
  return pixels;
}

function displayResults(probs, top) {
  const pred = document.getElementById('pred-digit');
  const fill = document.getElementById('conf-fill');
  const confVal = document.getElementById('conf-value');
  const bigPred = document.getElementById('big-pred');
  const barList = document.getElementById('bar-list');

  pred.textContent = top;
  pred.classList.remove('dim');

  const topPct = min(100, max(0, probs[top] * 100));
  const pct = topPct.toFixed(1);
  fill.style.width = pct + '%';
  confVal.textContent = pct + '%';
  bigPred.classList.add('lit');

  barList.innerHTML = '';
  const sorted = probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]);

  for (const [digit, prob] of sorted) {
    const isTop = digit === top;
    const widthPct = min(100, max(0, prob * 100));
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-digit">${digit}</div>
      <div class="bar-track">
        <div class="bar-fill${isTop ? ' top' : ''}" style="width:${widthPct.toFixed(2)}%"></div>
      </div>
      <div class="bar-pct${isTop ? ' top' : ''}">${widthPct.toFixed(1)}%</div>`;
    barList.appendChild(row);
  }
}

function resetResults() {
  const pred = document.getElementById('pred-digit');
  pred.textContent = '?';
  pred.classList.add('dim');

  document.getElementById('conf-fill').style.width = '0%';
  document.getElementById('conf-value').textContent = '-';
  document.getElementById('big-pred').classList.remove('lit');

  const barList = document.getElementById('bar-list');
  barList.innerHTML = '<div id="idle-hint">Draw a digit and press Predict</div>';
}

function runInference() {
  if (!model) {
    return;
  }

  const mathApi = getMathApi();
  const pixels = getPixels();
  const output = mlpForward(pixels, model.weights, model.biases, model.sizes);
  const probs = Array.from(mathApi.softmaxFromLogits(output));

  let top = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[top]) {
      top = i;
    }
  }

  displayResults(probs, top);
}

function runInferenceSilent() {
  if (inferenceScheduled) {
    return;
  }

  inferenceScheduled = true;
  requestAnimationFrame(() => {
    inferenceScheduled = false;
    runInference();
  });
}

function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 280, 280);
  updatePreview();
  resetResults();
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  [lastX, lastY] = getPos(e);
  wrap.classList.add('active');

  ctx.beginPath();
  ctx.arc(lastX, lastY, 11, 0, TWO_PI);
  ctx.fillStyle = '#fff';
  ctx.fill();

  updatePreview();
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) {
    return;
  }

  const [x, y] = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  [lastX, lastY] = [x, y];
  updatePreview();

  if (model) {
    runInferenceSilent();
  }
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
  wrap.classList.remove('active');
  if (model) {
    runInference();
  }
});

canvas.addEventListener('mouseleave', () => {
  drawing = false;
  wrap.classList.remove('active');
});

canvas.addEventListener(
  'touchstart',
  (e) => {
    e.preventDefault();
    canvas.dispatchEvent(
      new MouseEvent('mousedown', {
        clientX: e.touches[0].clientX,
        clientY: e.touches[0].clientY
      })
    );
  },
  { passive: false }
);

canvas.addEventListener(
  'touchmove',
  (e) => {
    e.preventDefault();
    canvas.dispatchEvent(
      new MouseEvent('mousemove', {
        clientX: e.touches[0].clientX,
        clientY: e.touches[0].clientY
      })
    );
  },
  { passive: false }
);

canvas.addEventListener(
  'touchend',
  (e) => {
    e.preventDefault();
    canvas.dispatchEvent(new MouseEvent('mouseup'));
  },
  { passive: false }
);

globalThis.runInference = runInference;
globalThis.clearCanvas = clearCanvas;

updatePreview();
loadLocalModel();
