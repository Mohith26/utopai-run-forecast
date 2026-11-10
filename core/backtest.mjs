// Held-out evaluation of the forecaster on historical runs, and the fit
// used for the live forecast.
import { fitHistory, posteriorDraws, forecastPlan, conformalLambda, widen, actualSpend, reliability, singlePassRange, quantile } from './forecast.mjs';

export function historyPlan(p) {
  return {
    anchors: p.anchors,
    shots: p.clips.map((c) => ({
      id: c.shot,
      res: c.res,
      hasRef: c.hasRef,
      segments: [{ duration: c.duration, chars: c.chars, motion: c.motion, version: c.version }],
    })),
  };
}

export function intervalFor(p, fit, draws, alpha) {
  const plan = historyPlan(p);
  const f = forecastPlan(plan, fit, draws, { cap: 0, autoPass: false });
  return {
    id: p.id,
    lo: quantile(f.dist, alpha / 2),
    hi: quantile(f.dist, 1 - alpha / 2),
    med: f.median,
    actual: actualSpend(p.takes),
    single: singlePassRange(plan),
  };
}

/** Live configuration: fit on the first nTrain projects, conformalise on the rest. */
export function liveFit(history, { nTrain = 450, alpha = 0.2, S = 8, seed = 11 } = {}) {
  const P = history.projects;
  const fit = fitHistory(P.slice(0, nTrain).flatMap((p) => p.takes));
  const draws = posteriorDraws(fit, S, seed);
  const calib = P.slice(nTrain).map((p) => intervalFor(p, fit, draws, alpha));
  return { fit, lambda: conformalLambda(calib, alpha), alpha, nTrain, nCalib: calib.length, nTakes: fit.nTakes };
}

/**
 * Rotating split evaluation: the projects are cut into `folds` blocks. In
 * rotation r, block r is the test set, block r+1 the calibration set and
 * the rest the training set. Reports coverage of the nominal (1-alpha)
 * interval before and after conformal recalibration, and how often the
 * single-pass x[0.9, 1.2] range contains the actual cost to final.
 */
export function backtest(history, { folds = 4, alpha = 0.2, S = 8, seed = 11 } = {}) {
  const P = history.projects;
  const B = Math.floor(P.length / folds);
  const blocks = Array.from({ length: folds }, (_, i) => P.slice(i * B, (i + 1) * B));
  const rotations = [];
  let rel = null;
  for (let r = 0; r < folds; r++) {
    const test = blocks[r];
    const calib = blocks[(r + 1) % folds];
    const train = blocks.filter((_, i) => i !== r && i !== (r + 1) % folds).flat();
    const fit = fitHistory(train.flatMap((p) => p.takes));
    const draws = posteriorDraws(fit, S, seed + r);
    const cal = calib.map((p) => intervalFor(p, fit, draws, alpha));
    const tst = test.map((p) => intervalFor(p, fit, draws, alpha));
    const lambda = conformalLambda(cal, alpha);
    const inside = (row, lo, hi) => row.actual >= lo - 1e-9 && row.actual <= hi + 1e-9;
    rotations.push({
      lambda,
      n: tst.length,
      raw: tst.filter((t) => inside(t, t.lo, t.hi)).length / tst.length,
      conformal: tst.filter((t) => { const w = widen(t, lambda); return inside(t, w.lo, w.hi); }).length / tst.length,
      single: tst.filter((t) => inside(t, t.single.low, t.single.high)).length / tst.length,
      singleUnder: tst.filter((t) => t.actual > t.single.high).length / tst.length,
      ratio: tst.reduce((s, t) => s + t.actual / t.single.total, 0) / tst.length,
      test: tst,
    });
    if (r === 0) rel = reliability(fit, test.flatMap((p) => p.takes), 5);
  }
  const avg = (k) => rotations.reduce((s, x) => s + x[k], 0) / rotations.length;
  return {
    alpha,
    folds,
    rotations,
    coverageRaw: avg('raw'),
    coverageConformal: avg('conformal'),
    coverageSinglePass: avg('single'),
    singlePassUnder: avg('singleUnder'),
    meanActualOverSingle: avg('ratio'),
    reliability: rel,
    nTest: rotations.reduce((s, x) => s + x.n, 0),
  };
}
