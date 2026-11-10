// Run Forecast core: cost-to-final forecasting for PAI-Pro Auto runs.
//
// Pure ES module, no dependencies. Used by the browser demo (index.html)
// and by the unit tests (node --test).
//
// Pipeline
//   1. Fit a Bayesian logistic model of "a take passes the alignment check"
//      from take history (features: clip length, characters, motion, model
//      version, resolution, character refs). Laplace posterior.
//   2. For a planned run, compute the exact distribution of total spend
//      until every clip passes (retake policy: up to K takes per clip),
//      by convolving per-job cost distributions on a 1-cent grid and
//      averaging over posterior draws.
//   3. Recalibrate the forecast interval with split conformal prediction
//      (CQR-style, relative scores) on held-out historical runs.
//   4. Re-plan each shot (model version, split at beat boundaries) to
//      minimise expected cost-to-final. Separable by linearity of
//      expectation, so per-shot enumeration is exact.

// ───────────────────────── prices (PAI-Pro model_registry.js) ─────────────────────────

export const PRICES = {
  video20PerSec: { '480p': 0.11, '720p': 0.23, '1080p': 0.57 },
  video25Tiers: [
    { ceiling: 20, '480p': 3.10, '720p': 6.95, '1080p': 17.10 },
    { ceiling: 35, '480p': 4.64, '720p': 10.42, '1080p': 25.63 },
    { ceiling: 60, '480p': 5.54, '720p': 12.46, '1080p': 30.64 },
  ],
  imagePro2K: 0.45,
  image2K: 0.16,
  ttsPer500Chars: 0.01,
};

export const LIMITS = { '2.0': { min: 4, max: 15 }, '2.5': { min: 5, max: 30 } };

export function videoCost(version, res, seconds) {
  const billed = Math.ceil(seconds);
  if (version === '2.5') {
    const tier = PRICES.video25Tiers.find((t) => billed <= t.ceiling);
    if (!tier) throw new Error('2.5 billed seconds above 60');
    return tier[res];
  }
  const rate = PRICES.video20PerSec[res];
  if (rate === undefined) throw new Error('unknown resolution ' + res);
  return Math.round(billed * rate * 100) / 100;
}

export function anchorCost(kind) {
  if (kind === 'character_sheet') return PRICES.imagePro2K;
  if (kind === 'location') return PRICES.image2K;
  if (kind === 'voice') return PRICES.ttsPer500Chars;
  throw new Error('unknown anchor kind ' + kind);
}

export const cents = (usd) => Math.round(usd * 100);

// ───────────────────────── seeded RNG ─────────────────────────

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function normal(rng) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gammaSample(rng, shape) {
  // Marsaglia-Tsang
  if (shape < 1) return gammaSample(rng, shape + 1) * Math.pow(rng(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = normal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export function betaSample(rng, a, b) {
  const x = gammaSample(rng, a);
  const y = gammaSample(rng, b);
  return x / (x + y);
}

// ───────────────────────── features + logistic model ─────────────────────────

export const FEATURE_NAMES = [
  'intercept',
  'length (per 10 s)',
  'length² (per 100 s²)',
  'characters in frame',
  'motion level',
  'Video 2.5',
  '1080p',
  '480p',
  'character refs attached',
];

export function featurize(seg) {
  const d = seg.duration / 10;
  return [
    1,
    d,
    d * d,
    seg.chars,
    seg.motion,
    seg.version === '2.5' ? 1 : 0,
    seg.res === '1080p' ? 1 : 0,
    seg.res === '480p' ? 1 : 0,
    seg.hasRef ? 1 : 0,
  ];
}

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

export function invert(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const piv = A[c][c];
    if (Math.abs(piv) < 1e-12) throw new Error('singular matrix');
    for (let j = 0; j < 2 * n; j++) A[c][j] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j];
    }
  }
  return A.map((row) => row.slice(n));
}

export function cholesky(M) {
  const n = M.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('matrix not positive definite');
        L[i][j] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

/**
 * MAP logistic regression with a Gaussian prior (precision `lambda` on all
 * coefficients except the intercept), fitted by Newton-Raphson. Returns the
 * posterior mode and the Laplace covariance (inverse Hessian).
 */
export function fitLogistic(X, y, { lambda = 1, iters = 50, tol = 1e-10 } = {}) {
  const p = X[0].length;
  let beta = new Array(p).fill(0);
  let H = null;
  for (let it = 0; it < iters; it++) {
    const g = new Array(p).fill(0);
    H = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < X.length; i++) {
      const mu = sigmoid(dot(X[i], beta));
      const w = mu * (1 - mu);
      for (let a = 0; a < p; a++) {
        g[a] += X[i][a] * (y[i] - mu);
        for (let b = 0; b <= a; b++) H[a][b] += w * X[i][a] * X[i][b];
      }
    }
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < a; b++) H[b][a] = H[a][b];
      if (a > 0) {
        g[a] -= lambda * beta[a];
        H[a][a] += lambda;
      }
    }
    const step = invert(H).map((row) => dot(row, g));
    beta = beta.map((v, i) => v + step[i]);
    if (Math.max(...step.map(Math.abs)) < tol) break;
  }
  const cov = invert(H);
  let ll = 0;
  for (let i = 0; i < X.length; i++) {
    const mu = sigmoid(dot(X[i], beta));
    ll += y[i] ? Math.log(mu) : Math.log(1 - mu);
  }
  return { beta, cov, logLik: ll, n: X.length };
}

/**
 * Fit everything the forecaster needs from take history:
 *  - logistic pass model for video takes,
 *  - Beta posteriors for anchor (image / voice) pass rates,
 *  - the infra failure rate (content filter, transient, timeout): those
 *    takes are billed (Auto reservations are never refunded) but return
 *    no clip.
 */
export function fitHistory(takes) {
  const video = takes.filter((t) => t.kind === 'video' && t.outcome !== 'failed');
  const X = video.map((t) => featurize(t));
  const y = video.map((t) => (t.outcome === 'pass' ? 1 : 0));
  const model = fitLogistic(X, y);
  const kinds = {};
  for (const k of ['character_sheet', 'location', 'voice']) {
    const ts = takes.filter((t) => t.kind === k && t.outcome !== 'failed');
    const pass = ts.filter((t) => t.outcome === 'pass').length;
    kinds[k] = { a: 1 + pass, b: 1 + ts.length - pass, n: ts.length };
  }
  const nFail = takes.filter((t) => t.outcome === 'failed').length;
  const failRate = (1 + nFail) / (2 + takes.length);
  return { model, kinds, failRate, nTakes: takes.length, nVideo: video.length };
}

export function passProb(beta, seg) {
  return sigmoid(dot(featurize(seg), beta));
}

/** Draw S joint posterior samples (logistic coefficients + anchor pass rates). */
export function posteriorDraws(fit, S, seed = 7) {
  const rng = mulberry32(seed);
  const L = cholesky(fit.model.cov);
  const p = fit.model.beta.length;
  const draws = [];
  for (let s = 0; s < S; s++) {
    const z = Array.from({ length: p }, () => normal(rng));
    const beta = fit.model.beta.map((b, i) => {
      let v = b;
      for (let k = 0; k <= i; k++) v += L[i][k] * z[k];
      return v;
    });
    const kinds = {};
    for (const [k, ab] of Object.entries(fit.kinds)) kinds[k] = betaSample(rng, ab.a, ab.b);
    draws.push({ beta, kinds });
  }
  return draws;
}

export function meanDraw(fit) {
  const kinds = {};
  for (const [k, ab] of Object.entries(fit.kinds)) kinds[k] = ab.a / (ab.a + ab.b);
  return { beta: fit.model.beta, kinds };
}

// ───────────────────────── jobs and per-job cost distributions ─────────────────────────

/**
 * Expand a plan into billable jobs. A plan is
 *   { anchors: [{kind, label}], shots: [{id, res, hasRef, segments:[{duration, chars, motion, version}]}] }
 */
export function planJobs(plan) {
  const jobs = [];
  for (const a of plan.anchors) jobs.push({ type: 'anchor', kind: a.kind, label: a.label, cost: anchorCost(a.kind) });
  for (const shot of plan.shots) {
    shot.segments.forEach((seg, i) => {
      const full = { ...seg, res: shot.res, hasRef: shot.hasRef };
      jobs.push({
        type: 'clip',
        shotId: shot.id,
        segIndex: i,
        seg: full,
        cost: videoCost(seg.version, shot.res, seg.duration),
      });
    });
  }
  return jobs;
}

export function jobSuccess(job, draw, failRate) {
  const p = job.type === 'anchor' ? draw.kinds[job.kind] : passProb(draw.beta, job.seg);
  return p * (1 - failRate);
}

/**
 * Takes N for one job under "retake until it passes, at most K takes".
 * Returns {resolved:[P(N=k and passed)], unresolved: P(K takes, none passed)}.
 */
export function takeDistribution(s, K) {
  const resolved = [];
  let q = 1;
  for (let k = 1; k <= K; k++) {
    resolved.push(q * s);
    q *= 1 - s;
  }
  return { resolved, unresolved: q };
}

export function expectedTakes(s, K) {
  // E[N] = sum_{k=1..K} P(N >= k) = sum (1-s)^{k-1}
  let e = 0;
  let q = 1;
  for (let k = 1; k <= K; k++) {
    e += q;
    q *= 1 - s;
  }
  return e;
}

/**
 * Convolve independent job spend distributions on a 1-cent grid.
 * Tracks two arrays: `all` (total spend, any outcome) and `final` (spend
 * restricted to the event that every job passed). Exact.
 */
export function convolveJobs(jobs, successes, K) {
  let all = new Float64Array(1);
  all[0] = 1;
  let fin = new Float64Array(1);
  fin[0] = 1;
  jobs.forEach((job, j) => {
    const c = cents(job.cost);
    const k = job.type === 'anchor' ? K.anchor : K.clip;
    const td = takeDistribution(successes[j], k);
    const len = all.length + c * k;
    const nAll = new Float64Array(len);
    const nFin = new Float64Array(len);
    for (let n = 1; n <= k; n++) {
      const pr = td.resolved[n - 1] + (n === k ? td.unresolved : 0);
      const pf = td.resolved[n - 1];
      const off = n * c;
      for (let x = 0; x < all.length; x++) {
        if (all[x] !== 0) nAll[x + off] += all[x] * pr;
        if (fin[x] !== 0) nFin[x + off] += fin[x] * pf;
      }
    }
    all = nAll;
    fin = nFin;
  });
  return { all, fin };
}

function addInto(acc, arr, w) {
  if (acc.length < arr.length) {
    const n = new Float64Array(arr.length);
    n.set(acc);
    acc = n;
  }
  for (let i = 0; i < arr.length; i++) acc[i] += arr[i] * w;
  return acc;
}

export function quantile(dist, q) {
  // dist: Float64Array over cents (may be sub-probability; normalised here)
  let total = 0;
  for (const v of dist) total += v;
  let c = 0;
  for (let i = 0; i < dist.length; i++) {
    c += dist[i] / total;
    if (c >= q - 1e-12) return i / 100;
  }
  return (dist.length - 1) / 100;
}

export function cdfAt(dist, usd) {
  const lim = Math.min(dist.length - 1, Math.floor(usd * 100 + 1e-9));
  let c = 0;
  for (let i = 0; i <= lim; i++) c += dist[i];
  return c;
}

export function meanOf(dist) {
  let m = 0;
  let t = 0;
  for (let i = 0; i < dist.length; i++) {
    m += i * dist[i];
    t += dist[i];
  }
  return m / t / 100;
}

/**
 * Forecast a plan.
 *  - cost-to-final: retake each job until it passes (K.clip / K.anchor takes max)
 *  - auto pass: what one Auto run spends (images restage once, video never)
 */
export function forecastPlan(plan, fit, draws, { K = { clip: 5, anchor: 3 }, cap = 30, autoPass = true } = {}) {
  const jobs = planJobs(plan);
  let all = new Float64Array(1);
  let fin = new Float64Array(1);
  let auto = new Float64Array(1);
  const perJob = jobs.map(() => ({ s: 0, takes: 0, cost: 0, resolved: 0 }));
  let expFlagged = 0;
  const w = 1 / draws.length;
  for (const d of draws) {
    const succ = jobs.map((j) => jobSuccess(j, d, fit.failRate));
    const r = convolveJobs(jobs, succ, K);
    all = addInto(all, r.all, w);
    fin = addInto(fin, r.fin, w);
    if (autoPass) {
      const a = convolveJobs(jobs, succ, { clip: 1, anchor: 2 });
      auto = addInto(auto, a.all, w);
    }
    jobs.forEach((j, i) => {
      const k = j.type === 'anchor' ? K.anchor : K.clip;
      const et = expectedTakes(succ[i], k);
      perJob[i].s += succ[i] * w;
      perJob[i].takes += et * w;
      perJob[i].cost += et * j.cost * w;
      perJob[i].resolved += (1 - Math.pow(1 - succ[i], k)) * w;
      if (j.type === 'clip') expFlagged += (1 - succ[i]) * w;
    });
  }
  let pFinal = 0;
  for (const v of fin) pFinal += v;
  const singlePass = jobs.reduce((s, j) => s + j.cost, 0);
  return {
    jobs,
    perJob,
    dist: all,
    finalDist: fin,
    autoDist: auto,
    singlePass: Math.round(singlePass * 100) / 100,
    mean: meanOf(all),
    median: quantile(all, 0.5),
    lo: quantile(all, 0.1),
    hi: quantile(all, 0.9),
    p90: quantile(all, 0.9),
    pAllResolved: pFinal,
    pFinalUnderCap: cdfAt(fin, cap),
    pAutoUnderCap: cdfAt(auto, cap),
    autoMean: meanOf(auto),
    expFlaggedAfterAuto: expFlagged,
  };
}

/** Smallest cap (in cents) at which P(every clip final and spend <= cap) >= target. */
export function capFor(finalDist, target) {
  let c = 0;
  for (let i = 0; i < finalDist.length; i++) {
    c += finalDist[i];
    if (c >= target) return i / 100;
  }
  return null; // target unreachable: P(all resolved within K takes) < target
}

// ───────────────────────── split conformal recalibration ─────────────────────────

/**
 * Split conformal recalibration of the forecast distribution (scaled
 * residuals around the median, in the spirit of locally weighted conformal
 * prediction, Lei et al. 2018). For each held-out calibration run with
 * forecast median m, interval [lo, hi] and actual spend y, the score is
 *   max((m - y) / (m - lo), (y - m) / (hi - m)),
 * i.e. the factor by which the interval must be stretched around its
 * median to contain y. lambda is the ceil((n+1)(1-alpha))-th smallest
 * score, so [m - lambda(m - lo), m + lambda(hi - m)] covers a new
 * exchangeable run with probability >= 1 - alpha. The same stretch is then
 * applied to the whole predictive distribution (Y' = m + lambda (Y - m)),
 * so every probability shown comes from one calibrated distribution.
 */
export function conformalLambda(calib, alpha) {
  const scores = calib
    .map((r) => Math.max((r.med - r.actual) / Math.max(r.med - r.lo, 0.01), (r.actual - r.med) / Math.max(r.hi - r.med, 0.01)))
    .sort((a, b) => a - b);
  const n = scores.length;
  const k = Math.ceil((n + 1) * (1 - alpha));
  if (k > n) return Infinity;
  return Math.max(scores[k - 1], 1e-6);
}

export function widen(r, lambda) {
  return { lo: Math.max(0, r.med - lambda * (r.med - r.lo)), hi: r.med + lambda * (r.hi - r.med) };
}

/** Calibrated quantile / CDF of a stretched distribution Y' = m + lambda (Y - m). */
export function calQuantile(dist, q, med, lambda) {
  return Math.max(0, med + lambda * (quantile(dist, q) - med));
}
export function calCdf(dist, usd, med, lambda) {
  return cdfAt(dist, med + (usd - med) / lambda);
}
export function calCapFor(finalDist, target, med, lambda) {
  const raw = capFor(finalDist, target);
  return raw === null ? null : med + lambda * (raw - med);
}

// ───────────────────────── plan optimisation ─────────────────────────

/** All ways to cut a shot at its beat boundaries into contiguous segments. */
export function segmentations(beats) {
  const m = beats.length;
  const out = [];
  for (let mask = 0; mask < 1 << (m - 1); mask++) {
    const segs = [];
    let start = 0;
    for (let i = 0; i < m; i++) {
      const cut = i === m - 1 || mask & (1 << i);
      if (cut) {
        const group = beats.slice(start, i + 1);
        segs.push({
          t0: group[0].t0,
          t1: group[group.length - 1].t1,
          duration: group[group.length - 1].t1 - group[0].t0,
          chars: Math.max(...group.map((b) => b.chars)),
          motion: Math.max(...group.map((b) => b.motion)),
          beats: group.map((b) => b.id),
        });
        start = i + 1;
      }
    }
    out.push(segs);
  }
  return out;
}

export function shotOptions(shot, versions = ['2.0', '2.5']) {
  const opts = [];
  for (const segs of segmentations(shot.beats)) {
    for (const version of versions) {
      const lim = LIMITS[version];
      if (segs.some((s) => s.duration < lim.min || s.duration > lim.max)) continue;
      opts.push({ version, segments: segs.map((s) => ({ ...s, version })) });
    }
  }
  return opts;
}

/** Expected cost-to-final and P(resolved within K) of one shot option, averaged over draws. */
export function evalOption(shot, opt, fit, draws, K = 5) {
  let ecost = 0;
  let pres = 0;
  for (const d of draws) {
    let c = 0;
    let r = 1;
    for (const seg of opt.segments) {
      const job = { type: 'clip', seg: { ...seg, res: shot.res, hasRef: shot.hasRef } };
      const s = jobSuccess(job, d, fit.failRate);
      c += expectedTakes(s, K) * videoCost(seg.version, shot.res, seg.duration);
      r *= 1 - Math.pow(1 - s, K);
    }
    ecost += c / draws.length;
    pres += r / draws.length;
  }
  return { ecost, pres };
}

/**
 * Per-shot argmin of expected cost-to-final. A plan's expected cost is the
 * sum of per-shot expected costs (linearity of expectation), so choosing
 * each shot's best option independently is globally optimal for this
 * objective. `cutPenalty` (USD per extra cut) keeps the plan from cutting a
 * shot for pennies; `minResolved` rejects options that are cheap only
 * because they often fail to land within K takes.
 */
export function optimizeShots(shots, fit, draws, { K = 5, cutPenalty = 0.25, minResolvedRatio = 1, versions } = {}) {
  return shots.map((shot) => {
    const options = shotOptions(shot, versions).map((opt) => ({ ...opt, ...evalOption(shot, opt, fit, draws, K) }));
    const drafted = options.find((o) => o.version === '2.0' && o.segments.length === 1) || options[0];
    let best = drafted;
    let bestScore = drafted.ecost;
    for (const o of options) {
      if (o.pres < drafted.pres * minResolvedRatio - 1e-9) continue;
      const score = o.ecost + cutPenalty * (o.segments.length - 1);
      if (score < bestScore - 1e-9) {
        best = o;
        bestScore = score;
      }
    }
    return { shotId: shot.id, drafted, best, options };
  });
}

export function buildPlan(project, choice) {
  // choice: shotId -> option ({version, segments})
  return {
    anchors: project.anchors,
    shots: project.shots.map((s) => ({
      id: s.id,
      res: s.res,
      hasRef: s.hasRef,
      segments: choice[s.id].segments.map((g) => ({ duration: g.duration, chars: g.chars, motion: g.motion, version: g.version, t0: g.t0, t1: g.t1, beats: g.beats })),
    })),
  };
}

export function draftedChoice(project) {
  // The Auto default the agent drafts today: one Video 2.0 clip per shot.
  const c = {};
  for (const s of project.shots) {
    const one = segmentations(s.beats)[0];
    c[s.id] = { version: '2.0', segments: one.map((g) => ({ ...g, version: '2.0' })) };
  }
  return c;
}

/** PAI-Pro's current UI estimate shape: single-pass total x [0.9, 1.2]. */
export function singlePassRange(plan) {
  const t = planJobs(plan).reduce((s, j) => s + j.cost, 0);
  return { low: t * 0.9, high: t * 1.2, total: t };
}

// ───────────────────────── backtest helpers ─────────────────────────

export function actualSpend(takes) {
  return Math.round(takes.reduce((s, t) => s + t.cost, 0) * 100) / 100;
}

export function reliability(fit, takes, bins = 5) {
  const video = takes.filter((t) => t.kind === 'video' && t.outcome !== 'failed');
  const rows = video.map((t) => ({ p: passProb(fit.model.beta, t), y: t.outcome === 'pass' ? 1 : 0 }));
  rows.sort((a, b) => a.p - b.p);
  const out = [];
  for (let b = 0; b < bins; b++) {
    const part = rows.slice(Math.floor((b * rows.length) / bins), Math.floor(((b + 1) * rows.length) / bins));
    if (!part.length) continue;
    out.push({
      predicted: part.reduce((s, r) => s + r.p, 0) / part.length,
      observed: part.reduce((s, r) => s + r.y, 0) / part.length,
      n: part.length,
    });
  }
  let brier = 0;
  for (const r of rows) brier += (r.p - r.y) ** 2;
  return { bins: out, brier: brier / rows.length, n: rows.length };
}
