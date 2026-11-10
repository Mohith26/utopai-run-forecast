import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as F from '../core/forecast.mjs';
import { backtest, liveFit } from '../core/backtest.mjs';
import { generateHistory, PROJECT, TRUTH, trueLogit } from '../data/generate.mjs';

const HISTORY = JSON.parse(readFileSync(new URL('../data/history.json', import.meta.url)));
const PROJ = JSON.parse(readFileSync(new URL('../data/project.json', import.meta.url)));

test('generator reproduces the committed data', () => {
  const h = generateHistory();
  assert.equal(JSON.stringify(h), JSON.stringify(HISTORY));
  assert.equal(PROJ.shots.length, PROJECT.shots.length);
});

test('prices match the PAI-Pro model registry', () => {
  assert.equal(F.videoCost('2.0', '720p', 15), 3.45);
  assert.equal(F.videoCost('2.0', '480p', 4), 0.44);
  assert.equal(F.videoCost('2.0', '720p', 7.5), F.videoCost('2.0', '720p', 8)); // billed per started second
  assert.equal(F.videoCost('2.5', '720p', 20), 6.95);
  assert.equal(F.videoCost('2.5', '720p', 21), 10.42);
  assert.equal(F.videoCost('2.5', '1080p', 60), 30.64);
  assert.equal(F.anchorCost('character_sheet'), 0.45);
});

test('take distribution sums to one and E[takes] matches the closed form', () => {
  for (const s of [0.05, 0.3, 0.77, 1]) {
    for (const K of [1, 3, 5]) {
      const td = F.takeDistribution(s, K);
      const total = td.resolved.reduce((a, b) => a + b, 0) + td.unresolved;
      assert.ok(Math.abs(total - 1) < 1e-12);
      const closed = (1 - Math.pow(1 - s, K)) / s;
      assert.ok(Math.abs(F.expectedTakes(s, K) - closed) < 1e-12);
    }
  }
});

test('grid convolution equals brute-force enumeration', () => {
  const jobs = [
    { type: 'clip', cost: 1.15 },
    { type: 'anchor', cost: 0.45 },
    { type: 'clip', cost: 2.3 },
  ];
  const succ = [0.4, 0.8, 0.55];
  const K = { clip: 3, anchor: 2 };
  const { all, fin } = F.convolveJobs(jobs, succ, K);
  const bruteAll = new Map();
  const bruteFin = new Map();
  const outcomes = jobs.map((j, i) => {
    const k = j.type === 'anchor' ? K.anchor : K.clip;
    const td = F.takeDistribution(succ[i], k);
    const o = [];
    for (let n = 1; n <= k; n++) {
      o.push({ c: n * F.cents(j.cost), p: td.resolved[n - 1], ok: true });
      if (n === k) o.push({ c: n * F.cents(j.cost), p: td.unresolved, ok: false });
    }
    return o;
  });
  for (const a of outcomes[0]) for (const b of outcomes[1]) for (const c of outcomes[2]) {
    const x = a.c + b.c + c.c;
    const p = a.p * b.p * c.p;
    bruteAll.set(x, (bruteAll.get(x) || 0) + p);
    if (a.ok && b.ok && c.ok) bruteFin.set(x, (bruteFin.get(x) || 0) + p);
  }
  for (let x = 0; x < all.length; x++) {
    assert.ok(Math.abs(all[x] - (bruteAll.get(x) || 0)) < 1e-12, `all@${x}`);
    assert.ok(Math.abs(fin[x] - (bruteFin.get(x) || 0)) < 1e-12, `fin@${x}`);
  }
});

test('logistic MAP fit recovers known coefficients', () => {
  const rng = F.mulberry32(5);
  const X = [];
  const y = [];
  const truth = [1.2, -0.8, -0.1, -0.4, -0.5, 0.6, -0.2, 0.15, 0.4];
  for (let i = 0; i < 20000; i++) {
    const seg = {
      duration: 4 + Math.floor(rng() * 12),
      chars: Math.floor(rng() * 4),
      motion: Math.floor(rng() * 3),
      version: rng() < 0.3 ? '2.5' : '2.0',
      res: ['480p', '720p', '1080p'][Math.floor(rng() * 3)],
      hasRef: rng() < 0.7,
    };
    const x = F.featurize(seg);
    X.push(x);
    y.push(rng() < F.sigmoid(x.reduce((s, v, j) => s + v * truth[j], 0)) ? 1 : 0);
  }
  const fit = F.fitLogistic(X, y, { lambda: 0.01 });
  fit.beta.forEach((b, j) => {
    const se = Math.sqrt(fit.cov[j][j]);
    assert.ok(Math.abs(b - truth[j]) < 4 * se + 0.02, `coef ${F.FEATURE_NAMES[j]}: ${b} vs ${truth[j]} (se ${se})`);
  });
});

test('fitted history model points the right way on every driver', () => {
  const fit = F.fitHistory(HISTORY.projects.flatMap((p) => p.takes));
  const b = fit.model.beta;
  const longer = F.passProb(b, { duration: 15, chars: 1, motion: 1, version: '2.0', res: '720p', hasRef: true });
  const shorter = F.passProb(b, { duration: 6, chars: 1, motion: 1, version: '2.0', res: '720p', hasRef: true });
  assert.ok(shorter > longer);
  assert.ok(b[3] < 0 && b[4] < 0 && b[5] > 0 && b[8] > 0); // characters, motion hurt; 2.5 and refs help
  assert.ok(Math.abs(fit.failRate - TRUTH.failRate) < 0.01);
});

test('forecast mean equals the sum of per-job expected costs (linearity)', () => {
  const fit = F.fitHistory(HISTORY.projects.slice(0, 450).flatMap((p) => p.takes));
  const draws = F.posteriorDraws(fit, 8, 3);
  const plan = F.buildPlan(PROJ, F.draftedChoice(PROJ));
  const f = F.forecastPlan(plan, fit, draws, { cap: 40 });
  const sum = f.perJob.reduce((s, j) => s + j.cost, 0);
  assert.ok(Math.abs(f.mean - sum) < 0.01, `${f.mean} vs ${sum}`);
  assert.ok(f.median >= f.lo && f.median <= f.hi);
  assert.ok(f.pFinalUnderCap <= f.pAllResolved + 1e-12);
  assert.equal(f.singlePass, F.singlePassRange(plan).total.toFixed(2) * 1);
});

test('segmentations enumerate every cut at beat boundaries', () => {
  const shot = PROJ.shots.find((s) => s.beats.length === 3);
  const segs = F.segmentations(shot.beats);
  assert.equal(segs.length, 4);
  for (const s of segs) assert.equal(s.reduce((a, g) => a + g.duration, 0), shot.beats[2].t1 - shot.beats[0].t0);
  for (const o of F.shotOptions(shot)) for (const g of o.segments) {
    assert.ok(g.duration >= F.LIMITS[o.version].min && g.duration <= F.LIMITS[o.version].max);
  }
});

test('per-shot optimisation is globally optimal for expected cost (brute force)', () => {
  const fit = F.fitHistory(HISTORY.projects.slice(0, 450).flatMap((p) => p.takes));
  const draws = F.posteriorDraws(fit, 4, 9);
  const shots = PROJ.shots.slice(0, 3);
  const res = F.optimizeShots(shots, fit, draws, { cutPenalty: 0, minResolvedRatio: 0 });
  const greedy = res.reduce((s, r) => s + r.best.ecost, 0);
  let best = Infinity;
  const [a, b, c] = res.map((r) => r.options);
  for (const x of a) for (const y of b) for (const z of c) best = Math.min(best, x.ecost + y.ecost + z.ecost);
  assert.ok(Math.abs(greedy - best) < 1e-9);
  for (const r of res) assert.ok(r.best.ecost <= r.drafted.ecost + 1e-9);
});

test('conformal stretch factor gives finite-sample coverage', () => {
  const rng = F.mulberry32(42);
  let covered = 0;
  let total = 0;
  for (let trial = 0; trial < 300; trial++) {
    const make = () => {
      const med = 20 + rng() * 20;
      const y = med * Math.exp(F.normal(rng) * 0.25); // forecast is too narrow on purpose
      return { med, lo: med * 0.9, hi: med * 1.1, actual: y };
    };
    const cal = Array.from({ length: 40 }, make);
    const lam = F.conformalLambda(cal, 0.2);
    for (let i = 0; i < 20; i++) {
      const t = make();
      const w = F.widen(t, lam);
      total++;
      if (t.actual >= w.lo && t.actual <= w.hi) covered++;
    }
  }
  const cov = covered / total;
  assert.ok(cov >= 0.78 && cov <= 0.86, `coverage ${cov}`);
});

test('calibrated quantiles and cap are consistent with the stretch', () => {
  const fit = F.fitHistory(HISTORY.projects.slice(0, 450).flatMap((p) => p.takes));
  const draws = F.posteriorDraws(fit, 8, 3);
  const f = F.forecastPlan(F.buildPlan(PROJ, F.draftedChoice(PROJ)), fit, draws, { cap: 40 });
  assert.equal(F.calQuantile(f.dist, 0.9, f.median, 1), f.hi);
  const cap = F.calCapFor(f.finalDist, 0.5, f.median, 1.2);
  if (cap !== null) assert.ok(F.calCdf(f.finalDist, cap + 0.01, f.median, 1.2) >= 0.5 - 1e-9);
});

test('rotating backtest: conformal fixes the under-coverage of the raw model', () => {
  const bt = backtest(HISTORY);
  assert.ok(bt.coverageRaw < 0.78, `raw ${bt.coverageRaw}`);
  assert.ok(bt.coverageConformal >= 0.76 && bt.coverageConformal <= 0.86, `conformal ${bt.coverageConformal}`);
  assert.ok(bt.coverageSinglePass < 0.4);
  const live = liveFit(HISTORY);
  assert.ok(live.lambda > 1 && live.lambda < 2);
});

test('ground-truth helper and fitted model agree on the direction of length', () => {
  const short = trueLogit({ duration: 5, chars: 1, motion: 0, version: '2.0', res: '720p', hasRef: true }, 0);
  const long = trueLogit({ duration: 15, chars: 1, motion: 0, version: '2.0', res: '720p', hasRef: true }, 0);
  assert.ok(short > long);
});
