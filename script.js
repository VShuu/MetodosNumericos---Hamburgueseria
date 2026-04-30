/* =====================================================
   BURGERMATH — SCRIPT.JS
   Numerical Methods: LU, Jacobi, Gauss-Seidel, SOR, CG, PCG
   Context: Hamburger shop ingredient optimization
===================================================== */

"use strict";

/* ==========================
   1. PREDEFINED SCENARIOS
========================== */
const SCENARIOS = {
  ideal: {
    name: "Caso Ideal",
    A: [
      [10, 2, 1],
      [1, 8, 2],
      [2, 1, 9]
    ],
    b: [130, 110, 120],
    desc: "Sistema bien condicionado — demanda normal de la hamburguesería"
  },
  stress: {
    name: "Caso Bajo Estrés",
    A: [
      [120, 15, 8],
      [10, 95, 12],
      [7, 9, 110]
    ],
    b: [1450, 1200, 1350],
    desc: "Coeficientes grandes — evento especial, alta demanda"
  },
  ill: {
    name: "Caso Mal Condicionado",
    A: [
      [10.0, 9.9,  9.8],
      [9.9,  9.8,  9.7],
      [9.8,  9.7,  9.6]
    ],
    b: [29.7, 29.4, 29.1],
    desc: "Ingredientes casi equivalentes — mal condicionamiento numérico"
  }
};

let currentScenario = "ideal";
let convergenceChartInstance = null;
let contourChartInstance = null;

/* ==========================
   2. MATRIX UTILITIES
========================== */

function matCopy(A) {
  return A.map(row => [...row]);
}

function matMulVec(A, x) {
  const n = A.length;
  const result = new Array(n).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      result[i] += A[i][j] * x[j];
  return result;
}

function vecSub(a, b) {
  return a.map((v, i) => v - b[i]);
}

function vecAdd(a, b) {
  return a.map((v, i) => v + b[i]);
}

function vecScale(a, s) {
  return a.map(v => v * s);
}

function vecDot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function vecNorm(v) {
  return Math.sqrt(vecDot(v, v));
}

function vecNormInf(v) {
  return Math.max(...v.map(Math.abs));
}

/* Compute condition number via power iteration (approximation) */
function conditionNumber(A) {
  const n = A.length;
  // Use ratio of Frobenius norm of A and A_inv approximation via LU
  try {
    const { L, U, P } = luDecompose(matCopy(A));
    // Estimate via 1-norm
    let norm1A = 0;
    let norm1Ainv = 0;
    for (let j = 0; j < n; j++) {
      let col1 = 0;
      let ej = new Array(n).fill(0); ej[j] = 1;
      let ainv_col = luSolve(L, U, P, ej);
      for (let i = 0; i < n; i++) {
        col1 += Math.abs(A[i][j]);
        norm1Ainv += Math.abs(ainv_col[i]);
      }
      norm1A = Math.max(norm1A, col1);
    }
    // Frobenius of inv
    let ainv_norm = 0;
    for (let j = 0; j < n; j++) {
      let ej = new Array(n).fill(0); ej[j] = 1;
      let col = luSolve(L, U, P, ej);
      for (let v of col) ainv_norm = Math.max(ainv_norm, Math.abs(v));
    }
    // condition as max absolute column sum of A * max of inv
    let normA = 0;
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.abs(A[i][j]);
      normA = Math.max(normA, s);
    }
    let normInv = 0;
    for (let j = 0; j < n; j++) {
      let ej = new Array(n).fill(0); ej[j] = 1;
      let col = luSolve(L, U, P, ej);
      let s = 0;
      for (let v of col) s += Math.abs(v);
      normInv = Math.max(normInv, s);
    }
    return normA * normInv;
  } catch (e) {
    return Infinity;
  }
}

/* ==========================
   3. LU DECOMPOSITION
========================== */

function luDecompose(A) {
  const n = A.length;
  const L = Array.from({length: n}, (_, i) => new Array(n).fill(0).map((_, j) => i === j ? 1 : 0));
  const U = matCopy(A);
  const P = Array.from({length: n}, (_, i) => i); // permutation

  for (let k = 0; k < n; k++) {
    // Partial pivoting
    let maxVal = Math.abs(U[k][k]);
    let maxRow = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(U[i][k]) > maxVal) {
        maxVal = Math.abs(U[i][k]);
        maxRow = i;
      }
    }
    if (maxRow !== k) {
      [U[k], U[maxRow]] = [U[maxRow], U[k]];
      [P[k], P[maxRow]] = [P[maxRow], P[k]];
      for (let j = 0; j < k; j++) {
        [L[k][j], L[maxRow][j]] = [L[maxRow][j], L[k][j]];
      }
    }

    for (let i = k + 1; i < n; i++) {
      if (Math.abs(U[k][k]) < 1e-15) continue;
      const m = U[i][k] / U[k][k];
      L[i][k] = m;
      for (let j = k; j < n; j++) {
        U[i][j] -= m * U[k][j];
      }
    }
  }
  return { L, U, P };
}

function luSolve(L, U, P, b) {
  const n = L.length;
  // Apply permutation
  const pb = P.map(i => b[i]);

  // Forward substitution: Ly = pb
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = pb[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }

  // Back substitution: Ux = y
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= U[i][j] * x[j];
    if (Math.abs(U[i][i]) < 1e-15) throw new Error("Singular matrix");
    x[i] = s / U[i][i];
  }
  return x;
}

function solveLU(A, b) {
  const Ac = matCopy(A);
  const { L, U, P } = luDecompose(Ac);
  const x = luSolve(L, U, P, b);
  return {
    x,
    method: "Factorización LU",
    iterations: 1,
    converged: true,
    errors: [],
    log: `LU con pivoteo parcial\nL y U calculadas exitosamente.\nSolución: x = [${x.map(v=>v.toFixed(6)).join(', ')}]`
  };
}

/* ==========================
   4. JACOBI
========================== */

function solveJacobi(A, b, tol = 1e-6, maxIter = 1000) {
  const n = A.length;
  let x = new Array(n).fill(0);
  const errors = [];
  let log = "";

  for (let iter = 0; iter < maxIter; iter++) {
    const xNew = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let j = 0; j < n; j++) {
        if (j !== i) sum -= A[i][j] * x[j];
      }
      if (Math.abs(A[i][i]) < 1e-15) return { x, method: "Jacobi", iterations: iter, converged: false, errors, log: "Diagonal nula" };
      xNew[i] = sum / A[i][i];
    }

    const res = vecSub(matMulVec(A, xNew), b);
    const err = vecNorm(res) / (vecNorm(b) + 1e-15);
    errors.push(err);

    if (iter < 10 || iter % 20 === 0)
      log += `Iter ${iter+1}: error = ${err.toExponential(3)}, x = [${xNew.map(v=>v.toFixed(4)).join(', ')}]\n`;

    if (err < tol) {
      log += `\n✓ CONVERGIÓ en iteración ${iter+1}`;
      return { x: xNew, method: "Jacobi", iterations: iter + 1, converged: true, errors, log };
    }
    x = xNew;
  }
  log += `\n✗ No convergió en ${maxIter} iteraciones`;
  return { x, method: "Jacobi", iterations: maxIter, converged: false, errors, log };
}

/* ==========================
   5. GAUSS-SEIDEL
========================== */

function solveGaussSeidel(A, b, tol = 1e-6, maxIter = 1000) {
  const n = A.length;
  let x = new Array(n).fill(0);
  const errors = [];
  let log = "";

  for (let iter = 0; iter < maxIter; iter++) {
    const xOld = [...x];
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let j = 0; j < n; j++) {
        if (j !== i) sum -= A[i][j] * x[j];
      }
      if (Math.abs(A[i][i]) < 1e-15) return { x, method: "Gauss-Seidel", iterations: iter, converged: false, errors, log: "Diagonal nula" };
      x[i] = sum / A[i][i];
    }

    const res = vecSub(matMulVec(A, x), b);
    const err = vecNorm(res) / (vecNorm(b) + 1e-15);
    errors.push(err);

    if (iter < 10 || iter % 20 === 0)
      log += `Iter ${iter+1}: error = ${err.toExponential(3)}, x = [${x.map(v=>v.toFixed(4)).join(', ')}]\n`;

    if (err < tol) {
      log += `\n✓ CONVERGIÓ en iteración ${iter+1}`;
      return { x: [...x], method: "Gauss-Seidel", iterations: iter + 1, converged: true, errors, log };
    }
  }
  log += `\n✗ No convergió en ${maxIter} iteraciones`;
  return { x: [...x], method: "Gauss-Seidel", iterations: maxIter, converged: false, errors, log };
}

/* ==========================
   6. SOR
========================== */

function solveSOR(A, b, omega = 1.25, tol = 1e-6, maxIter = 1000) {
  const n = A.length;
  let x = new Array(n).fill(0);
  const errors = [];
  let log = `SOR con ω = ${omega}\n`;

  for (let iter = 0; iter < maxIter; iter++) {
    const xOld = [...x];
    for (let i = 0; i < n; i++) {
      let sigma = b[i];
      for (let j = 0; j < n; j++) {
        if (j !== i) sigma -= A[i][j] * x[j];
      }
      if (Math.abs(A[i][i]) < 1e-15) return { x, method: "SOR", iterations: iter, converged: false, errors, log: "Diagonal nula" };
      const xGS = sigma / A[i][i];
      x[i] = (1 - omega) * x[i] + omega * xGS;
    }

    const res = vecSub(matMulVec(A, x), b);
    const err = vecNorm(res) / (vecNorm(b) + 1e-15);
    errors.push(err);

    if (iter < 10 || iter % 20 === 0)
      log += `Iter ${iter+1}: error = ${err.toExponential(3)}, x = [${x.map(v=>v.toFixed(4)).join(', ')}]\n`;

    if (err < tol) {
      log += `\n✓ CONVERGIÓ en iteración ${iter+1}`;
      return { x: [...x], method: "SOR", iterations: iter + 1, converged: true, errors, log };
    }
  }
  log += `\n✗ No convergió en ${maxIter} iteraciones`;
  return { x: [...x], method: "SOR", iterations: maxIter, converged: false, errors, log };
}

/* ==========================
   7. GRADIENT DESCENT (Steepest Descent)
========================== */

function solveSteepestDescent(A, b, tol = 1e-6, maxIter = 1000) {
  const n = A.length;
  let x = new Array(n).fill(0);
  let r = vecSub(b, matMulVec(A, x));
  const errors = [];
  let log = "Máximo Descenso (Gradient Descent)\n";

  for (let iter = 0; iter < maxIter; iter++) {
    const Ar = matMulVec(A, r);
    const rTr = vecDot(r, r);
    const rTAr = vecDot(r, Ar);

    if (Math.abs(rTAr) < 1e-30) break;

    const alpha = rTr / rTAr;
    x = vecAdd(x, vecScale(r, alpha));
    r = vecSub(r, vecScale(Ar, alpha));

    const err = vecNorm(r) / (vecNorm(b) + 1e-15);
    errors.push(err);

    if (iter < 10 || iter % 20 === 0)
      log += `Iter ${iter+1}: ||r|| = ${err.toExponential(3)}\n`;

    if (err < tol) {
      log += `\n✓ CONVERGIÓ en iteración ${iter+1}`;
      return { x, method: "Grad. Descenso", iterations: iter + 1, converged: true, errors, log };
    }
  }
  log += `\n✗ No convergió en ${maxIter} iteraciones`;
  return { x, method: "Grad. Descenso", iterations: maxIter, converged: false, errors, log };
}

/* ==========================
   8. CONJUGATE GRADIENT
========================== */

function solveConjugateGradient(A, b, tol = 1e-6, maxIter = 1000) {
  const n = A.length;
  let x = new Array(n).fill(0);
  let r = vecSub(b, matMulVec(A, x));
  let p = [...r];
  let rTr = vecDot(r, r);
  const errors = [];
  let log = "Gradiente Conjugado\n";

  for (let iter = 0; iter < maxIter; iter++) {
    const Ap = matMulVec(A, p);
    const pTAp = vecDot(p, Ap);

    if (Math.abs(pTAp) < 1e-30) break;

    const alpha = rTr / pTAp;
    x = vecAdd(x, vecScale(p, alpha));
    r = vecSub(r, vecScale(Ap, alpha));

    const rTrNew = vecDot(r, r);
    const err = Math.sqrt(rTrNew) / (vecNorm(b) + 1e-15);
    errors.push(err);

    if (iter < 10 || iter % 5 === 0)
      log += `Iter ${iter+1}: ||r|| = ${err.toExponential(3)}\n`;

    if (err < tol) {
      log += `\n✓ CONVERGIÓ en iteración ${iter+1}`;
      return { x, method: "Grad. Conjugado", iterations: iter + 1, converged: true, errors, log };
    }

    const beta = rTrNew / rTr;
    p = vecAdd(r, vecScale(p, beta));
    rTr = rTrNew;
  }
  log += `\n✗ No convergió en ${maxIter} iteraciones`;
  return { x, method: "Grad. Conjugado", iterations: maxIter, converged: false, errors, log };
}

/* ==========================
   9. PRECONDITIONED CG (Jacobi Preconditioner)
========================== */

function solvePCG(A, b, tol = 1e-6, maxIter = 1000) {
  const n = A.length;
  // Jacobi preconditioner: M = diag(A)
  const M_inv = A.map((row, i) => Math.abs(row[i]) > 1e-15 ? 1.0 / row[i] : 1.0);

  const applyPrec = (v) => v.map((vi, i) => vi * M_inv[i]);

  let x = new Array(n).fill(0);
  let r = vecSub(b, matMulVec(A, x));
  let z = applyPrec(r);
  let p = [...z];
  let rz = vecDot(r, z);
  const errors = [];
  let log = "PCG — Gradiente Conjugado Precondicionado (M = diag(A))\n";

  for (let iter = 0; iter < maxIter; iter++) {
    const Ap = matMulVec(A, p);
    const pTAp = vecDot(p, Ap);

    if (Math.abs(pTAp) < 1e-30) break;

    const alpha = rz / pTAp;
    x = vecAdd(x, vecScale(p, alpha));
    r = vecSub(r, vecScale(Ap, alpha));

    const err = vecNorm(r) / (vecNorm(b) + 1e-15);
    errors.push(err);

    if (iter < 10 || iter % 5 === 0)
      log += `Iter ${iter+1}: ||r|| = ${err.toExponential(3)}\n`;

    if (err < tol) {
      log += `\n✓ CONVERGIÓ en iteración ${iter+1}`;
      return { x, method: "PCG", iterations: iter + 1, converged: true, errors, log };
    }

    z = applyPrec(r);
    const rzNew = vecDot(r, z);
    const beta = rzNew / rz;
    p = vecAdd(z, vecScale(p, beta));
    rz = rzNew;
  }
  log += `\n✗ No convergió en ${maxIter} iteraciones`;
  return { x, method: "PCG", iterations: maxIter, converged: false, errors, log };
}

/* ==========================
   10. DISPATCHER
========================== */

function runMethod(A, b, method, tol, maxIter, omega) {
  switch (method) {
    case "lu":     return solveLU(A, b);
    case "jacobi": return solveJacobi(A, b, tol, maxIter);
    case "gs":     return solveGaussSeidel(A, b, tol, maxIter);
    case "sor":    return solveSOR(A, b, omega, tol, maxIter);
    case "cg":     return solveConjugateGradient(A, b, tol, maxIter);
    case "pcg":    return solvePCG(A, b, tol, maxIter);
    default:       return solveLU(A, b);
  }
}

/* ==========================
   11. UI HELPERS
========================== */

function getMatrixFromUI() {
  const A = [];
  for (let i = 0; i < 3; i++) {
    const row = [];
    for (let j = 0; j < 3; j++) {
      const val = parseFloat(document.getElementById(`a${i}${j}`).value) || 0;
      row.push(val);
    }
    A.push(row);
  }
  const b = [0, 1, 2].map(i => parseFloat(document.getElementById(`b${i}`).value) || 0);
  return { A, b };
}

function setMatrixToUI(scenario) {
  const s = SCENARIOS[scenario];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      document.getElementById(`a${i}${j}`).value = s.A[i][j];
    }
    document.getElementById(`b${i}`).value = s.b[i];
  }
}

function buildMatrixInputs() {
  const grid = document.getElementById('matrixInputGrid');
  grid.innerHTML = '';
  // Header row
  ['', 'x₁', 'x₂', 'x₃'].forEach(h => {
    const el = document.createElement('div');
    el.style.cssText = 'text-align:center;font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);padding:2px 0;';
    el.textContent = h;
    grid.appendChild(el);
  });
  for (let i = 0; i < 3; i++) {
    const label = document.createElement('div');
    label.style.cssText = 'text-align:center;font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);display:flex;align-items:center;justify-content:center;';
    label.textContent = `eq${i+1}`;
    grid.appendChild(label);
    for (let j = 0; j < 3; j++) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.id = `a${i}${j}`;
      inp.step = 'any';
      grid.appendChild(inp);
    }
  }

  const bGrid = document.getElementById('bInputGrid');
  bGrid.innerHTML = '';
  for (let i = 0; i < 3; i++) {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.id = `b${i}`;
    inp.step = 'any';
    inp.className = 'b-input';
    inp.placeholder = `b${i+1}`;
    bGrid.appendChild(inp);
  }
}

function displaySolution(result) {
  const sv = document.getElementById('solValues');
  const sm = document.getElementById('solMeta');
  sv.innerHTML = result.x.map((v, i) =>
    `<span>x${i+1} = <strong>${v.toFixed(6)}</strong> ${getIngLabel(i)}</span>`
  ).join('');
  sm.innerHTML = result.converged
    ? `<span style="color:var(--green-light)">✓ ${result.method} · ${result.iterations} iter${result.iterations>1?'s':''}</span>`
    : `<span style="color:var(--red)">✗ ${result.method} · No convergió (${result.iterations} iters)</span>`;
}

function getIngLabel(i) {
  return ['(kg carne)', '(und. pan)', '(porciones extra)'][i] || '';
}

function displayCondNumber(A) {
  const kappa = conditionNumber(A);
  const cv = document.getElementById('condValue');
  const ci = document.getElementById('condInterp');
  cv.textContent = kappa > 1e10 ? '> 10¹⁰ (∞)' : kappa.toFixed(2);
  let interp = '', color = 'var(--green-light)';
  if (kappa < 10) { interp = '✅ Bien condicionado'; color = 'var(--green-light)'; }
  else if (kappa < 1000) { interp = '⚡ Condicionamiento moderado'; color = 'var(--amber)'; }
  else if (kappa < 1e8) { interp = '⚠️ Mal condicionado'; color = 'orange'; }
  else { interp = '🚨 Muy mal condicionado — inestable'; color = 'var(--red)'; }
  cv.style.color = color;
  ci.innerHTML = `<span style="color:${color}">${interp}</span>`;
}

function updateConvergenceChart(results) {
  const canvas = document.getElementById('convergenceChart');
  if (convergenceChartInstance) convergenceChartInstance.destroy();

  const colors = [
    '#F5A623', '#E63946', '#4CAF50', '#2196F3', '#9C27B0', '#FF9800'
  ];

  const datasets = results.map((r, idx) => ({
    label: r.method,
    data: r.errors.slice(0, 100).map((e, i) => ({ x: i + 1, y: Math.max(e, 1e-16) })),
    borderColor: colors[idx % colors.length],
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3
  })).filter(d => d.data.length > 0);

  convergenceChartInstance = new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Iteración', color: '#A08060', font: { family: 'JetBrains Mono', size: 11 } },
          ticks: { color: '#A08060', font: { family: 'JetBrains Mono', size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: '||r||/||b||', color: '#A08060', font: { family: 'JetBrains Mono', size: 11 } },
          ticks: { color: '#A08060', font: { family: 'JetBrains Mono', size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#F0E6D3', font: { family: 'JetBrains Mono', size: 11 }, boxWidth: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(26,16,8,0.95)',
          titleColor: '#F5A623',
          bodyColor: '#F0E6D3',
          borderColor: '#F5A623',
          borderWidth: 1
        }
      }
    }
  });
}

function displayIterLog(result) {
  document.getElementById('logBody').textContent = result.log || '(sin log)';
}

/* ==========================
   12. MATRIX DISPLAY CARDS
========================== */

function renderScenarioMatrices() {
  ['ideal', 'stress', 'ill'].forEach(key => {
    const sc = SCENARIOS[key];
    const el = document.getElementById(`mat-${key}`);
    if (!el) return;
    let text = 'A =\n';
    sc.A.forEach(row => {
      text += ' [' + row.map(v => String(v).padStart(7)).join('  ') + ' ]\n';
    });
    text += '\nb = [' + sc.b.join(', ') + ']';
    el.textContent = text;
  });
}

/* ==========================
   13. COMPARISON TABLE
========================== */

function fillComparisonTable() {
  const methods = [
    { key: 'jacobi', name: 'Jacobi' },
    { key: 'gs',     name: 'Gauss-Seidel' },
    { key: 'sor',    name: 'SOR (ω=1.25)' },
    { key: 'cg',     name: 'Grad. Conjugado' },
    { key: 'pcg',    name: 'PCG' },
    { key: 'lu',     name: 'Factorización LU' }
  ];

  const tol = 1e-6;
  const maxIter = 2000;
  const omega = 1.25;

  const tbody = document.getElementById('compTableBody');
  tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--amber);font-family:var(--font-mono)">Calculando...</td></tr>';

  // Run async to avoid blocking UI
  setTimeout(() => {
    tbody.innerHTML = '';
    methods.forEach(m => {
      const results = ['ideal', 'stress', 'ill'].map(sk => {
        const s = SCENARIOS[sk];
        try {
          return runMethod(s.A, s.b, m.key, tol, maxIter, omega);
        } catch(e) {
          return { iterations: '—', converged: false, errors: [] };
        }
      });

      const tr = document.createElement('tr');
      const iterCells = results.map(r => {
        const iters = r.iterations;
        let cls = 'badge-converge';
        if (!r.converged) cls = 'badge-fail';
        else if (m.key === 'lu') cls = 'badge-direct';
        else if (iters > 200) cls = 'badge-slow';
        return `<td class="${cls}">${iters}</td>`;
      });

      const allConverged = results.every(r => r.converged);
      const someConverged = results.some(r => r.converged);
      let convBadge, convClass;
      if (m.key === 'lu') { convBadge = '✓ Directo (siempre)'; convClass = 'badge-direct'; }
      else if (allConverged) { convBadge = '✓ Sí'; convClass = 'badge-converge'; }
      else if (someConverged) { convBadge = '⚠ Parcial'; convClass = 'badge-slow'; }
      else { convBadge = '✗ No'; convClass = 'badge-fail'; }

      tr.innerHTML = `
        <td style="color:var(--cream);font-weight:500">${m.name}</td>
        ${iterCells.join('')}
        <td class="${convClass}">${convBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  }, 100);
}

/* ==========================
   14. CONTOUR / CONVERGENCE PATH CHART
========================== */

function drawContourChart() {
  const canvas = document.getElementById('contourChart');
  if (!canvas) return;

  // Use simple 2D visualization of f(x1,x2) for fixed x3
  // f(x1,x2) = 0.5*(10*x1^2 + 8*x2^2 + 2*x1*x2) - 130*x1 - 110*x2
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // CG trajectory for ideal system
  const A = SCENARIOS.ideal.A;
  const b = SCENARIOS.ideal.b;
  const result = solveConjugateGradient(A, b, 1e-8, 50);

  if (contourChartInstance) contourChartInstance.destroy();

  // Build trajectory dataset (just x1, x2)
  // We'll track iterates manually
  const trajectory = [];
  {
    let x = [0, 0, 0];
    let r = vecSub(b, matMulVec(A, x));
    let p = [...r];
    let rTr = vecDot(r, r);
    trajectory.push({ x: x[0], y: x[1] });
    for (let iter = 0; iter < 15; iter++) {
      const Ap = matMulVec(A, p);
      const alpha = rTr / vecDot(p, Ap);
      x = vecAdd(x, vecScale(p, alpha));
      r = vecSub(r, vecScale(Ap, alpha));
      const rTrNew = vecDot(r, r);
      const beta = rTrNew / rTr;
      p = vecAdd(r, vecScale(p, beta));
      rTr = rTrNew;
      trajectory.push({ x: x[0], y: x[1] });
      if (Math.sqrt(rTr) / (vecNorm(b) + 1e-15) < 1e-8) break;
    }
  }

  // Generate ellipse contours (projected onto x1-x2)
  const xStar = result.x;
  const contourLevels = [0.5, 2, 8, 20, 60, 150];
  const contourDatasets = contourLevels.map((lev, idx) => {
    const pts = [];
    const steps = 80;
    for (let t = 0; t <= steps; t++) {
      const angle = (t / steps) * 2 * Math.PI;
      const scale = Math.sqrt(lev);
      const x1 = xStar[0] + scale * 2.5 * Math.cos(angle);
      const x2 = xStar[1] + scale * 1.5 * Math.sin(angle) + scale * 0.5 * Math.cos(angle);
      pts.push({ x: x1, y: x2 });
    }
    const alpha = 0.15 + idx * 0.05;
    return {
      label: `f=${lev}`,
      data: pts,
      borderColor: `rgba(245,166,35,${0.15 + idx*0.07})`,
      backgroundColor: 'transparent',
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.4,
      showLine: true
    };
  });

  // CG path
  const cgDataset = {
    label: 'Trayectoria CG',
    data: trajectory,
    borderColor: '#E63946',
    backgroundColor: '#E63946',
    borderWidth: 2.5,
    pointRadius: 4,
    pointBackgroundColor: '#E63946',
    tension: 0,
    showLine: true
  };

  // Optimal point
  const optDataset = {
    label: 'Solución x*',
    data: [{ x: xStar[0], y: xStar[1] }],
    borderColor: '#F5A623',
    backgroundColor: '#F5A623',
    pointRadius: 8,
    pointStyle: 'star',
    showLine: false
  };

  contourChartInstance = new Chart(canvas, {
    type: 'scatter',
    data: { datasets: [...contourDatasets, cgDataset, optDataset] },
    options: {
      responsive: false,
      animation: { duration: 800 },
      scales: {
        x: {
          title: { display: true, text: 'x₁ (carne kg)', color: '#A08060', font: { size: 11, family: 'JetBrains Mono' } },
          ticks: { color: '#A08060', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          title: { display: true, text: 'x₂ (pan)', color: '#A08060', font: { size: 11, family: 'JetBrains Mono' } },
          ticks: { color: '#A08060', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#F0E6D3',
            font: { family: 'JetBrains Mono', size: 10 },
            filter: (item) => ['Trayectoria CG', 'Solución x*'].includes(item.text)
          }
        },
        tooltip: {
          backgroundColor: 'rgba(26,16,8,0.95)',
          titleColor: '#F5A623',
          bodyColor: '#F0E6D3',
          borderColor: '#F5A623',
          borderWidth: 1
        }
      }
    }
  });
}

/* ==========================
   15. 3D PLANES (2D Projection)
========================== */

function drawPlanesVisualization() {
  const canvas = document.getElementById('planesChart');
  if (!canvas) return;

  const { A, b } = getMatrixFromUI();
  const n = A.length;

  // Project 3D planes onto 2D: show each plane as x1 vs x2 (with x3 = optimal x3)
  let sol;
  try {
    sol = solveLU(A, b).x;
  } catch(e) {
    sol = [1, 1, 1];
  }

  const x3_fixed = sol[2];
  const datasets = [];
  const colors = ['#F5A623', '#E63946', '#4CAF50'];
  const names = ['Plano 1 (Carne-Pan)', 'Plano 2 (Pan-Extra)', 'Plano 3 (Balance)'];

  for (let eq = 0; eq < 3; eq++) {
    const pts = [];
    const xRange = [sol[0] - 20, sol[0] + 20];
    for (let k = 0; k <= 40; k++) {
      const x1 = xRange[0] + k * (xRange[1] - xRange[0]) / 40;
      if (Math.abs(A[eq][1]) > 1e-10) {
        const x2 = (b[eq] - A[eq][0] * x1 - A[eq][2] * x3_fixed) / A[eq][1];
        pts.push({ x: x1, y: x2 });
      }
    }
    datasets.push({
      label: names[eq],
      data: pts,
      borderColor: colors[eq],
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 0,
      showLine: true,
      tension: 0
    });
  }

  // Solution point
  datasets.push({
    label: `Solución (${sol.map(v=>v.toFixed(2)).join(', ')})`,
    data: [{ x: sol[0], y: sol[1] }],
    borderColor: '#fff',
    backgroundColor: '#fff',
    pointRadius: 10,
    pointStyle: 'crossRot',
    showLine: false
  });

  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      animation: { duration: 800 },
      scales: {
        x: {
          title: { display: true, text: 'x₁ — Carne (kg)', color: '#A08060', font: { size: 12, family: 'JetBrains Mono' } },
          ticks: { color: '#A08060' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          title: { display: true, text: 'x₂ — Pan (unidades)', color: '#A08060', font: { size: 12, family: 'JetBrains Mono' } },
          ticks: { color: '#A08060' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#F0E6D3', font: { family: 'JetBrains Mono', size: 11 } }
        },
        tooltip: {
          backgroundColor: 'rgba(26,16,8,0.95)',
          titleColor: '#F5A623',
          bodyColor: '#F0E6D3',
          borderColor: '#F5A623',
          borderWidth: 1
        },
        title: {
          display: true,
          text: `Proyección de planos (x₃ = ${x3_fixed.toFixed(2)} fijo)`,
          color: '#A08060',
          font: { family: 'JetBrains Mono', size: 11 }
        }
      }
    }
  });
}

/* ==========================
   16. EVENT LISTENERS
========================== */

document.addEventListener('DOMContentLoaded', () => {

  // Build matrix inputs
  buildMatrixInputs();

  // Load default scenario
  setMatrixToUI(currentScenario);

  // Render scenario cards
  renderScenarioMatrices();

  // Draw contour chart
  drawContourChart();

  // --- Preset buttons ---
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentScenario = btn.dataset.preset;
      setMatrixToUI(currentScenario);
      const { A } = getMatrixFromUI();
      displayCondNumber(A);
    });
  });

  // --- Tab buttons ---
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    });
  });

  // --- Method buttons ---
  document.querySelectorAll('.meth-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.meth-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const omegaGroup = document.getElementById('omegaGroup');
      omegaGroup.style.display = btn.dataset.method === 'sor' ? 'flex' : 'none';
    });
  });

  // --- Run single method ---
  document.getElementById('runBtn').addEventListener('click', () => {
    const { A, b } = getMatrixFromUI();
    const method = document.querySelector('.meth-btn.active')?.dataset.method || 'lu';
    const tol = parseFloat(document.getElementById('tolInput').value) || 1e-6;
    const maxIter = parseInt(document.getElementById('maxIterInput').value) || 1000;
    const omega = parseFloat(document.getElementById('omegaInput').value) || 1.25;

    const btn = document.getElementById('runBtn');
    btn.classList.add('loading');

    setTimeout(() => {
      try {
        const result = runMethod(A, b, method, tol, maxIter, omega);
        displaySolution(result);
        displayCondNumber(A);
        updateConvergenceChart([result]);
        displayIterLog(result);
      } catch(e) {
        document.getElementById('logBody').textContent = '❌ Error: ' + e.message;
      }
      btn.classList.remove('loading');
    }, 50);
  });

  // --- Run all methods ---
  document.getElementById('runAllBtn').addEventListener('click', () => {
    const { A, b } = getMatrixFromUI();
    const tol = parseFloat(document.getElementById('tolInput').value) || 1e-6;
    const maxIter = parseInt(document.getElementById('maxIterInput').value) || 1000;
    const omega = parseFloat(document.getElementById('omegaInput').value) || 1.25;

    const allBtn = document.getElementById('runAllBtn');
    allBtn.textContent = '⏳ Calculando...';
    allBtn.disabled = true;

    setTimeout(() => {
      const methods = ['jacobi', 'gs', 'sor', 'cg', 'pcg', 'lu'];
      const results = methods.map(m => {
        try { return runMethod(A, b, m, tol, maxIter, omega); }
        catch(e) { return { method: m, x: [0,0,0], iterations: 0, converged: false, errors: [] }; }
      });

      // Display first result as reference
      displaySolution(results[results.length - 1]); // LU
      displayCondNumber(A);
      updateConvergenceChart(results.filter(r => r.errors && r.errors.length > 0));

      let logText = results.map(r =>
        `${r.method}: ${r.converged ? '✓' : '✗'} ${r.iterations} iters | x=[${r.x.map(v=>v.toFixed(4)).join(',')}]`
      ).join('\n');
      document.getElementById('logBody').textContent = logText;

      allBtn.textContent = '⚡ Ejecutar Todos';
      allBtn.disabled = false;
    }, 50);
  });

  // --- Comparison table ---
  document.getElementById('fillTableBtn').addEventListener('click', () => {
    fillComparisonTable();
  });

  // --- Draw planes ---
  document.getElementById('drawPlanesBtn').addEventListener('click', () => {
    drawPlanesVisualization();
  });

  // Initialize condition number on load
  const { A } = getMatrixFromUI();
  displayCondNumber(A);

  // Auto-fill table on load
  fillComparisonTable();

  // Animate elements into view
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animation = 'fadeInUp 0.6s ease both';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.scenario-card, .interp-card, .eq-box').forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });
});