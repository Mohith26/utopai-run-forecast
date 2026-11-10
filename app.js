import * as F from './core/forecast.mjs';
import { liveFit, backtest } from './core/backtest.mjs';

const $ = (id) => document.getElementById(id);
const usd = (v) => '$' + v.toFixed(2);
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const MOTION = ['static', 'moving', 'high motion'];

const S = {
  cap: 40,
  K: 5,
  cutPenalty: 0.25,
  replan: {},
  drawerTab: 'shots',
  bt: null,
};

async function load() {
  const [history, project] = await Promise.all([
    fetch('data/history.json').then((r) => r.json()),
    fetch('data/project.json').then((r) => r.json()),
  ]);
  S.history = history;
  S.project = project;
  const live = liveFit(history);
  S.fit = live.fit;
  S.lambda = live.lambda;
  S.live = live;
  S.draws = F.posteriorDraws(S.fit, 64, 7);
  reoptimize();
  for (const s of project.shots) S.replan[s.id] = false;
  wire();
  render();
  setTimeout(() => {
    S.bt = backtest(history);
    renderDrawer();
  }, 30);
}

function reoptimize() {
  S.opt = F.optimizeShots(S.project.shots, S.fit, S.draws, { K: S.K, cutPenalty: S.cutPenalty });
  S.optById = Object.fromEntries(S.opt.map((o) => [o.shotId, o]));
}

function choiceFor(replanMap) {
  const c = {};
  for (const s of S.project.shots) {
    const o = S.optById[s.id];
    c[s.id] = replanMap[s.id] ? o.best : o.drafted;
  }
  return c;
}

function forecast(replanMap) {
  const plan = F.buildPlan(S.project, choiceFor(replanMap));
  const f = F.forecastPlan(plan, S.fit, S.draws, { K: { clip: S.K, anchor: 3 }, cap: S.cap });
  const m = f.median;
  const L = S.lambda;
  return {
    plan,
    f,
    med: m,
    lo: F.calQuantile(f.dist, 0.1, m, L),
    hi: F.calQuantile(f.dist, 0.9, m, L),
    pCap: F.calCdf(f.finalDist, S.cap, m, L),
    cap80: F.calCapFor(f.finalDist, 0.8, m, L),
    pAll: f.pAllResolved,
  };
}

function render() {
  S.cur = forecast(S.replan);
  const all = Object.fromEntries(S.project.shots.map((s) => [s.id, S.optById[s.id].best.segments.length > 1 || S.optById[s.id].best.version !== '2.0']));
  S.full = forecast(all);
  S.none = forecast(Object.fromEntries(S.project.shots.map((s) => [s.id, false])));
  renderAuto();
  renderCanvas();
  renderTerminal();
  renderDrawer();
}

// ───────────────────────── Auto panel ─────────────────────────

function renderAuto() {
  const c = S.cur;
  const f = c.f;
  const range = F.singlePassRange(c.plan);
  const nClips = c.plan.shots.reduce((s, x) => s + x.segments.length, 0);
  const secs = S.project.shots.reduce((s, x) => s + (x.beats[x.beats.length - 1].t1 - x.beats[0].t0), 0);
  $('capText').textContent = `${usd(S.cap)} cap · own approval gate`;
  $('estLine').textContent = `${secs}s · ${nClips} clips · ${S.project.res} · ${usd(range.low)}-${usd(range.high)}`;

  const nSplit = S.project.shots.filter((s) => S.optById[s.id].best.segments.length > 1).length;
  const isAll = S.project.shots.every((s) => S.replan[s.id] === (S.optById[s.id].best !== S.optById[s.id].drafted));
  const isNone = S.project.shots.every((s) => !S.replan[s.id]);
  const cls = (p) => (p >= 0.8 ? 'good' : p >= 0.5 ? 'warn' : 'bad');

  // riskiest job in the current plan
  let worst = null;
  f.jobs.forEach((j, i) => {
    if (j.type !== 'clip') return;
    const pj = f.perJob[i];
    if (!worst || pj.s < worst.pj.s) worst = { j, pj };
  });
  const ws = S.project.shots.find((s) => s.id === worst.j.shotId);
  const wSeg = worst.j.seg;
  const unresolved = 1 - worst.pj.resolved;

  let rec;
  if (!isAll) {
    rec = `<div class="rec"><div class="t">Cut <b>${nSplit} shots</b> at beat boundaries: median <b>${usd(S.full.med)}</b> <span class="m">(was ${usd(c.med)})</span>, final within ${usd(S.cap)} <b>${pct(S.full.pCap)}</b>. Single-pass price unchanged at ${usd(S.full.f.singlePass)}.</div><button class="btn-light" id="applyBtn">Apply re-plan</button></div>`;
  } else {
    rec = `<div class="rec"><div class="t"><b>Re-planned:</b> ${nClips} clips from ${S.project.shots.length} shots. Same billed seconds; retakes get cheaper because shorter, simpler clips pass alignment more often. <span class="m">Was ${usd(S.none.med)} median.</span></div><button class="btn-light" id="detailsBtn2">Details</button></div>`;
  }

  $('fc').innerHTML = `
    <div class="fc-head">
      <div class="fc-title">Run forecast <span class="src">cost to final · ${S.history.projects.length} past runs</span></div>
      <div class="toggle" id="planToggle">
        <button data-v="none" class="${isNone ? 'on' : ''}">As drafted</button>
        <button data-v="all" class="${isAll ? 'on' : ''}">Re-planned</button>
        <button data-v="details" title="Per-shot plan, backtest and model">Details</button>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><div class="k">Cost to final (median)</div><div class="v">${usd(c.med)}</div><div class="s">80% range ${usd(c.lo)}–${usd(c.hi)}</div></div>
      <div class="stat"><div class="k">Final within ${usd(S.cap)}</div><div class="v ${cls(c.pCap)}">${pct(c.pCap)}</div><div class="s">every clip passes</div></div>
      <div class="stat"><div class="k">Cap for 80%</div><div class="v">${c.cap80 === null ? '—' : usd(Math.ceil(c.cap80))}</div><div class="s">${c.cap80 === null ? `${pct(1 - c.pAll)} need a rewrite` : `lands in ≤${S.K} takes: ${pct(c.pAll)}`}</div></div>
    </div>
    <div class="hist"><svg id="histSvg" viewBox="0 0 460 118" preserveAspectRatio="none"></svg>
      <div class="legend"><span><i style="background:#10b981"></i>final within cap</span><span><i style="background:#525252"></i>final, over cap</span><span><i style="background:repeating-linear-gradient(135deg,#7a3b2e 0 2px,#3a2420 2px 4px)"></i>a clip still failing after ${S.K} takes</span></div>
    </div>
    ${rec}
    <div class="why">Riskiest clip: <b>Shot ${ws.id} "${esc(ws.title)}"</b> ${wSeg.duration}s, ${wSeg.chars} character${wSeg.chars === 1 ? '' : 's'}, ${MOTION[wSeg.motion]}: passes alignment ${pct(worst.pj.s)} of takes, ${pct(unresolved)} chance it still fails after ${S.K}. Auto alone spends ~${usd(f.autoMean)} and flags ~${f.expFlaggedAfterAuto.toFixed(1)} of ${nClips} clips for retakes.</div>`;
  drawHist();
  $('planToggle').querySelectorAll('button').forEach((b) =>
    b.addEventListener('click', () => {
      if (b.dataset.v === 'details') return openDrawer('shots');
      for (const s of S.project.shots) S.replan[s.id] = b.dataset.v === 'all' && S.optById[s.id].best !== S.optById[s.id].drafted;
      render();
    }),
  );
  const ap = $('applyBtn');
  if (ap) ap.addEventListener('click', () => { for (const s of S.project.shots) S.replan[s.id] = S.optById[s.id].best !== S.optById[s.id].drafted; render(); });
  const d2 = $('detailsBtn2');
  if (d2) d2.addEventListener('click', () => openDrawer('shots'));
}

function drawHist() {
  const c = S.cur;
  const f = c.f;
  const m = c.med;
  const L = S.lambda;
  // calibrated distribution: Y' = m + L (Y - m), binned into $1 bins
  const q995 = (fc) => F.calQuantile(fc.f.dist, 0.995, fc.med, L);
  const xmax = Math.max(20, Math.ceil(Math.max(q995(S.none), q995(S.full), S.cap * 1.1)));
  const xmin = Math.max(0, Math.floor(Math.min(S.none.lo, S.full.lo, f.singlePass) * 0.85));
  const nb = xmax - xmin;
  const fin = new Float64Array(nb);
  const tot = new Float64Array(nb);
  const put = (arr, dist) => {
    for (let i = 0; i < dist.length; i++) {
      if (!dist[i]) continue;
      const y = m + L * (i / 100 - m);
      const b = Math.floor(y - xmin);
      if (b < 0 || b >= nb) continue; // tail beyond the 99.5th percentile is not drawn
      arr[b] += dist[i];
    }
  };
  put(fin, f.finalDist);
  put(tot, f.dist);
  const W = 460, H = 96, top = 6;
  const ymax = Math.max(...tot) * 1.08;
  const x = (v) => ((v - xmin) / (xmax - xmin)) * W;
  const y = (p) => top + H - (p / ymax) * H;
  let s = '';
  for (let g = 0; g <= 3; g++) s += `<line x1="0" x2="${W}" y1="${top + (H * g) / 3}" y2="${top + (H * g) / 3}" stroke="#262626" stroke-width="1"/>`;
  s += `<defs><pattern id="hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#3a2420"/><rect width="2" height="4" fill="#7a3b2e"/></pattern></defs>`;
  const bw = W / nb;
  for (let b = 0; b < nb; b++) {
    const lo = xmin + b;
    const pf = fin[b];
    const pu = Math.max(0, tot[b] - fin[b]);
    const x0 = b * bw + 0.6;
    const w = Math.max(0.8, bw - 1.2);
    const fillF = lo + 1 <= S.cap ? '#10b981' : '#525252';
    if (pf > 0) s += `<rect class="hb" data-b="${b}" x="${x0}" y="${y(pf)}" width="${w}" height="${top + H - y(pf)}" rx="1" fill="${fillF}" opacity="0.85"/>`;
    if (pu > 0) s += `<rect class="hb" data-b="${b}" x="${x0}" y="${y(pf + pu)}" width="${w}" height="${y(pf) - y(pf + pu)}" fill="url(#hatch)"/>`;
  }
  const lab = (v, t, col, anchor = 'start', dy = 0) => `<text x="${x(v) + (anchor === 'start' ? 4 : -4)}" y="${top + 9 + dy}" fill="${col}" font-size="10" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" text-anchor="${anchor}">${t}</text>`;
  s += `<line x1="${x(f.singlePass)}" x2="${x(f.singlePass)}" y1="${top}" y2="${top + H}" stroke="#a3a3a3" stroke-dasharray="3 3"/>`;
  s += lab(f.singlePass, `one pass ${usd(f.singlePass)}`, '#a3a3a3', 'start', 0);
  s += `<line x1="${x(S.cap)}" x2="${x(S.cap)}" y1="${top - 2}" y2="${top + H}" stroke="oklch(0.78 0.16 60)" stroke-width="1.5"/>`;
  s += lab(S.cap, `cap ${usd(S.cap)}`, 'oklch(0.82 0.14 75)', x(S.cap) > W - 70 ? 'end' : 'start', 12);
  s += `<line x1="0" x2="${W}" y1="${top + H}" y2="${top + H}" stroke="#404040"/>`;
  const step = xmax - xmin > 60 ? 20 : 10;
  for (let t = Math.ceil(xmin / step) * step; t <= xmax; t += step) {
    s += `<text x="${x(t)}" y="${top + H + 13}" fill="#737373" font-size="10" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">$${t}</text>`;
  }
  const svg = $('histSvg');
  svg.setAttribute('viewBox', `0 0 ${W} 118`);
  svg.innerHTML = s;
  const tip = $('tip');
  svg.querySelectorAll('.hb').forEach((r) => {
    r.addEventListener('mousemove', (e) => {
      const b = +r.dataset.b;
      tip.hidden = false;
      tip.textContent = `$${xmin + b}–$${xmin + b + 1}: ${pct(tot[b], 1)} (${pct(fin[b], 1)} final)`;
      tip.style.left = e.clientX + 12 + 'px';
      tip.style.top = e.clientY - 28 + 'px';
    });
    r.addEventListener('mouseleave', () => (tip.hidden = true));
  });
}

// ───────────────────────── canvas ─────────────────────────

function riskColor(s) {
  return s >= 0.6 ? 'oklch(0.65 0.12 150)' : s >= 0.4 ? 'oklch(0.78 0.16 60)' : 'oklch(0.62 0.18 25)';
}

function renderCanvas() {
  const vp = $('viewport');
  const f = S.cur.f;
  const nodes = [];
  const edges = [];
  const NOTE_W = 300, PAD_W = 290, X0 = 0, X1 = 370, X2 = 730;
  let yRow = 0;
  let noteIdx = 2;
  let tCursor = 0;
  let jobIdx = f.jobs.findIndex((j) => j.type === 'clip');
  const shotRows = [];
  for (const shot of S.project.shots) {
    const dur = shot.beats[shot.beats.length - 1].t1 - shot.beats[0].t0;
    const noteH = 92 + shot.beats.length * 44;
    const segs = S.cur.plan.shots.find((x) => x.id === shot.id).segments;
    const padH = 128;
    const stackH = segs.length * padH + (segs.length - 1) * 10;
    const rowH = Math.max(noteH, stackH);
    const noteY = yRow + (rowH - noteH) / 2;
    nodes.push(`<div class="node note" style="left:${X1}px;top:${noteY}px;width:${NOTE_W}px;height:${noteH}px"><div class="node-head"><span class="kind">@note_${noteIdx}</span></div><div class="node-body"><p>${esc(shot.heading)}</p>${shot.beats.map((b) => `<p>${esc(b.text)}</p>`).join('')}</div><div class="node-foot">Shot ${shot.id} (${tCursor}–${tCursor + dur}s)</div></div>`);
    edges.push({ from: [X0 + NOTE_W, 230], to: [X1, noteY + noteH / 2] });
    let py = yRow + (rowH - stackH) / 2;
    segs.forEach((seg) => {
      const j = f.jobs[jobIdx];
      const pj = f.perJob[jobIdx];
      jobIdx++;
      const beats = shot.beats.filter((b) => seg.beats.includes(b.id));
      const prompt = beats.map((b) => b.text).join(' ');
      const ver = seg.version === '2.5' ? ' · v2.5' : '';
      nodes.push(`<div class="node pending" style="left:${X2}px;top:${py}px;width:${PAD_W}px" data-shot="${shot.id}">
        <div class="node-head"><span class="kind">video · ${S.project.res} · ${seg.duration}s${ver}</span><span class="state-chip"><span class="dot"></span>draft</span></div>
        <div class="node-body">${esc(prompt)}</div>
        <div class="risk" title="Run forecast: pass rate per take from the alignment model"><span>pass <b>${pct(pj.s)}</b>/take</span><span class="bar"><i style="width:${(pj.s * 100).toFixed(1)}%;background:${riskColor(pj.s)}"></i></span><span>E ${pj.takes.toFixed(1)} takes · <b>${usd(pj.cost)}</b></span></div>
        <div class="node-foot"><button class="btn-cancel">Cancel</button><button class="btn-generate-primary">Generate · ${usd(j.cost)}</button></div>
      </div>`);
      edges.push({ from: [X1 + NOTE_W, noteY + noteH / 2], to: [X2, py + padH / 2] });
      py += padH + 10;
    });
    shotRows.push(rowH);
    yRow += rowH + 26;
    tCursor += dur;
    noteIdx++;
  }
  const totalH = yRow - 26;
  // script note
  const script = S.project.shots.map((s) => `<p>${esc(s.heading)}</p>${s.beats.map((b) => `<p>${esc(b.text)}</p>`).join('')}`).join('');
  const scriptH = Math.min(totalH, 560);
  nodes.unshift(`<div class="node note" style="left:${X0}px;top:${0}px;width:${NOTE_W}px;height:${scriptH}px"><div class="node-head"><span class="kind">@note_1</span></div><div class="node-body"><p>THE NIGHT FERRY</p>${script}</div><div class="node-foot">Script: The Night Ferry</div></div>`);
  edges.forEach((e) => {
    if (e.from[1] === 230) e.from[1] = scriptH / 2;
  });
  const W = X2 + PAD_W;
  const edgeSvg = `<svg class="edges" width="${W}" height="${totalH}">${edges
    .map((e) => {
      const dx = (e.to[0] - e.from[0]) * 0.5;
      return `<path d="M${e.from[0]},${e.from[1]} C${e.from[0] + dx},${e.from[1]} ${e.to[0] - dx},${e.to[1]} ${e.to[0]},${e.to[1]}"/>`;
    })
    .join('')}</svg>`;
  vp.innerHTML = edgeSvg + nodes.join('');
  // fit view
  const cv = $('canvas');
  const availW = cv.clientWidth - 90;
  const availH = cv.clientHeight - 80;
  const z = Math.min(0.86, availW / W);
  const tx = 74 + Math.max(0, (availW - W * z) / 2);
  const ty = 22 + Math.max(0, (availH - totalH * z) / 2);
  vp.style.transform = `translate(${tx}px, ${ty}px) scale(${z})`;
  $('zoomThumb').style.left = `${((z - 0.1) / (2 - 0.1)) * 100}%`;
  $('zoomTrack').style.background = `linear-gradient(to right, oklch(0.96 0.01 60) 0%, oklch(0.96 0.01 60) ${((z - 0.1) / 1.9) * 100}%, oklch(1 0 0 / 0.16) ${((z - 0.1) / 1.9) * 100}%)`;
}

// ───────────────────────── terminal ─────────────────────────

function renderTerminal() {
  const p = S.cur.plan;
  const range = F.singlePassRange(p);
  const nClips = p.shots.reduce((s, x) => s + x.segments.length, 0);
  const lines = [
    `<span class="g">●</span> Approval summary: <span class="b">The Night Ferry</span>`,
    ``,
    `  Runtime   78s requested, 78s planned`,
    `  Clips     ${nClips} from ${S.project.shots.length} shots, ${S.project.res}, Video 2.0`,
    `  Anchors   2 character sheets, 4 locations,`,
    `            3 voices (Ines, Teo, narrator VO)`,
    `  Dispatch  hybrid: 1 and 6 in parallel,`,
    `            2 → 3 → 4 → 5 sequential`,
    `  Estimate  ${usd(range.low)}–${usd(range.high)}, cap ${usd(S.cap)}`,
    `  Output    Timeline assignment only`,
    ``,
    `  Click <span class="b">Run Auto</span> to start. Every job is staged`,
    `  with <span class="c">--auto-run-id</span>; video mismatches are`,
    `  reported at the end, never auto-retried.`,
    ``,
    `<span class="d">✻ Planned for 41s</span>`,
  ];
  $('term').innerHTML = lines.join('\n') + `\n<div class="prompt">❯ <span style="color:#5a5a5a">waiting for Run Auto</span></div><span class="d">  ? for shortcuts · ← for agents</span>`;
}

// ───────────────────────── drawer ─────────────────────────

function openDrawer(tab) {
  S.drawerTab = tab || S.drawerTab;
  $('drawer').hidden = false;
  $('railForecast').classList.add('on');
  renderDrawer();
}

function renderDrawer() {
  if ($('drawer').hidden) return;
  document.querySelectorAll('#drawerTabs .seg-tab').forEach((b) => b.classList.toggle('on', b.dataset.tab === S.drawerTab));
  $('cntShots').textContent = S.project.shots.length;
  $('drawerMeta').textContent = `λ ${S.lambda.toFixed(2)} · K ${S.K}`;
  const body = $('drawerBody');
  if (S.drawerTab === 'shots') body.innerHTML = shotsTab();
  else if (S.drawerTab === 'backtest') body.innerHTML = backtestTab();
  else body.innerHTML = modelTab();
  body.querySelectorAll('input.chk[data-shot]').forEach((cb) =>
    cb.addEventListener('change', () => {
      S.replan[+cb.dataset.shot] = cb.checked;
      render();
    }),
  );
  const ks = $('kSel');
  if (ks) ks.addEventListener('change', () => { S.K = +ks.value; reoptimize(); render(); });
  const cp = $('cutSel');
  if (cp) cp.addEventListener('change', () => { S.cutPenalty = +cp.value; reoptimize(); render(); });
}

function segBar(segs, total, hasRef) {
  return `<div class="segs">${segs
    .map((g) => {
      const s = F.jobSuccess({ type: 'clip', seg: { ...g, res: S.project.res, hasRef } }, F.meanDraw(S.fit), S.fit.failRate);
      return `<i style="width:${(g.duration / total) * 100}%;background:${riskColor(s)}"></i>`;
    })
    .join('')}</div>`;
}

function shotsTab() {
  let rows = '';
  let tD = 0, tB = 0, tC = 0;
  for (const shot of S.project.shots) {
    const o = S.optById[shot.id];
    const dur = shot.beats[shot.beats.length - 1].t1 - shot.beats[0].t0;
    const changed = o.best !== o.drafted;
    const curOpt = S.replan[shot.id] ? o.best : o.drafted;
    const passes = curOpt.segments.map((g) => pct(F.jobSuccess({ type: 'clip', seg: { ...g, res: S.project.res, hasRef: shot.hasRef } }, F.meanDraw(S.fit), S.fit.failRate))).join(' · ');
    tD += o.drafted.ecost;
    tB += o.best.ecost;
    tC += curOpt.ecost;
    const maxChars = Math.max(...shot.beats.map((b) => b.chars));
    const maxMot = Math.max(...shot.beats.map((b) => b.motion));
    rows += `<tr>
      <td><input type="checkbox" class="chk" data-shot="${shot.id}" ${S.replan[shot.id] ? 'checked' : ''} ${changed ? '' : 'disabled title="No cheaper option"'}></td>
      <td><div style="color:var(--ink-0)">${shot.id}. ${esc(shot.title)}</div><div class="dim" style="font-size:10.5px;white-space:nowrap">${dur}s · ${maxChars} char · ${MOTION[maxMot]}</div></td>
      <td><div class="mono" style="font-size:11px">${o.drafted.segments.map((g) => g.duration).join('+')}s → ${changed ? `<span style="color:var(--ink-0)">${o.best.segments.map((g) => g.duration).join('+')}s${o.best.version === '2.5' ? ' v2.5' : ''}</span>` : '<span class="dim">keep</span>'}</div>${segBar(curOpt.segments, dur, shot.hasRef)}</td>
      <td class="num">${passes}</td>
      <td class="num">${usd(o.drafted.ecost)}</td>
      <td class="num" style="color:${changed ? 'var(--ink-0)' : 'var(--ink-3)'}">${usd(o.best.ecost)}</td>
      <td class="num">${pct(o.drafted.pres)} → ${pct(o.best.pres)}</td>
    </tr>`;
  }
  return `<div class="sec-label"><span>Per-shot plan</span><span class="cnt">${S.project.shots.length}</span><span class="rule"></span></div>
  <table class="shots"><thead><tr><th></th><th>Shot</th><th>Clips</th><th class="num">Pass/take</th><th class="num">E drafted</th><th class="num">E re-plan</th><th class="num">≤${S.K} takes</th></tr></thead>
  <tbody>${rows}<tr><td></td><td colspan="3" class="dim">Expected video spend to final (anchors excluded)</td><td class="num">${usd(tD)}</td><td class="num" style="color:var(--ink-0)">${usd(tB)}</td><td></td></tr></tbody></table>
  <p class="note-text">Each option is a way to cut the shot at its script beats, rendered on Video 2.0 or 2.5. Expected cost = Σ price × E[takes], with E[takes] = (1 − (1 − s)<sup>K</sup>) / s for per-take pass probability s, averaged over 64 posterior draws. Plan cost is a sum over shots, so the per-shot minimum is the global minimum. A cut has to save at least ${usd(S.cutPenalty)} in expectation to be suggested; tick a row to use it in the forecast above.</p>`;
}

function backtestTab() {
  const bt = S.bt;
  if (!bt) return `<div class="card dim">Running the rotating backtest on ${S.history.projects.length} historical runs…</div>`;
  const bar = (v, col, label) => `<div style="display:flex;justify-content:space-between;font-size:11.5px"><span>${label}</span></div><div class="covbar"><i style="width:${v * 100}%;background:${col}"></i><div class="nominal" style="left:${(1 - bt.alpha) * 100}%"></div><span>${pct(v, 1)}</span></div>`;
  const rel = bt.reliability;
  const W = 220, H = 164;
  let r = `<svg width="${W}" height="${H}" style="display:block"><rect x="30" y="6" width="${W - 40}" height="${H - 36}" fill="none" stroke="var(--line-1)"/>`;
  const px = (v) => 30 + v * (W - 40);
  const py = (v) => 6 + (1 - v) * (H - 36);
  r += `<line x1="${px(0)}" y1="${py(0)}" x2="${px(1)}" y2="${py(1)}" stroke="var(--ink-4)" stroke-dasharray="3 3"/>`;
  r += `<polyline fill="none" stroke="#10b981" stroke-width="1.5" points="${rel.bins.map((b) => `${px(b.predicted)},${py(b.observed)}`).join(' ')}"/>`;
  r += rel.bins.map((b) => `<circle cx="${px(b.predicted)}" cy="${py(b.observed)}" r="3" fill="#10b981"/>`).join('');
  for (const t of [0, 0.5, 1]) {
    r += `<text x="${px(t)}" y="${H - 16}" font-size="9.5" fill="var(--ink-3)" text-anchor="middle" font-family="ui-monospace,Menlo,monospace">${t}</text><text x="24" y="${py(t) + 3}" font-size="9.5" fill="var(--ink-3)" text-anchor="end" font-family="ui-monospace,Menlo,monospace">${t}</text>`;
  }
  r += `<text x="${px(0.5)}" y="${H - 3}" font-size="9.5" fill="var(--ink-3)" text-anchor="middle">predicted pass rate</text></svg>`;
  const rows = bt.rotations.map((x, i) => `<tr><td class="mono">${i + 1}</td><td class="num">${x.n}</td><td class="num">${x.lambda.toFixed(2)}</td><td class="num">${pct(x.raw, 1)}</td><td class="num">${pct(x.conformal, 1)}</td><td class="num">${pct(x.single, 1)}</td></tr>`).join('');
  return `<div class="sec-label"><span>Interval coverage on held-out runs</span><span class="cnt">${bt.nTest}</span><span class="rule"></span></div>
  <div class="card">${bar(bt.coverageRaw, '#525252', 'Model alone, nominal 80%')}${bar(bt.coverageConformal, '#10b981', 'After split-conformal stretch (what the panel shows)')}${bar(bt.coverageSinglePass, '#7a3b2e', 'Today’s single-pass ×[0.9, 1.2] range')}
  <p class="note-text" style="margin-top:0">The single-pass range prices one take per job, which is what an Auto run reserves; the actual cost to a final cut averaged <b style="color:var(--ink-1)">${bt.meanActualOverSingle.toFixed(2)}×</b> single pass and exceeded the top of the range in ${pct(bt.singlePassUnder)} of runs. The model alone under-covers because projects differ in difficulty, which it does not model; the conformal stretch fixes coverage without touching the model.</p></div>
  <div style="display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start">
    <div class="card" style="margin:0"><div class="dim" style="font-size:10px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">Take-level reliability</div>${r}<div class="dim mono" style="font-size:10px">Brier ${rel.brier.toFixed(3)} · ${rel.n} takes</div></div>
    <div><table class="shots"><thead><tr><th>Fold</th><th class="num">Runs</th><th class="num">λ</th><th class="num">Model</th><th class="num">Conformal</th><th class="num">Single</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note-text">Rotating split: in fold r, block r is the test set, block r+1 calibrates λ, the other two train the model. λ is the ⌈(n+1)(1−α)⌉-th smallest stretch factor needed to cover a calibration run.</p></div>
  </div>`;
}

function modelTab() {
  const b = S.fit.model.beta;
  const cov = S.fit.model.cov;
  const rows = F.FEATURE_NAMES.map((n, i) => {
    const se = Math.sqrt(cov[i][i]);
    return `<tr><td>${n}</td><td class="num">${b[i] >= 0 ? '+' : ''}${b[i].toFixed(2)} ± ${(1.96 * se).toFixed(2)}</td><td class="num">${i === 0 ? '' : '×' + Math.exp(b[i]).toFixed(2)}</td></tr>`;
  }).join('');
  const k = S.fit.kinds;
  const kr = (x) => pct(k[x].a / (k[x].a + k[x].b), 1);
  return `<div class="sec-label"><span>Alignment pass model</span><span class="cnt">${S.fit.nVideo} video takes</span><span class="rule"></span></div>
  <table class="shots"><thead><tr><th>Driver</th><th class="num">Log-odds (95%)</th><th class="num">Odds</th></tr></thead><tbody>${rows}</tbody></table>
  <p class="note-text">Bayesian logistic regression (Gaussian prior, Newton–Raphson MAP, Laplace posterior) on the alignment verdicts PAI-Pro already writes to <span class="mono">metadata.alignment</span>. Infra failures (content filter, transient, timeout) are modelled separately: they are billed, since Auto reservations are never refunded, but return no clip.</p>
  <div class="card kv" style="margin-top:10px">
    <span>Training runs / calibration runs</span><span class="mono">${S.live.nTrain} / ${S.live.nCalib}</span>
    <span>Infra failure rate per take</span><span class="mono">${pct(S.fit.failRate, 1)}</span>
    <span>Anchor pass rate: character sheet · location · voice</span><span class="mono">${kr('character_sheet')} · ${kr('location')} · ${kr('voice')}</span>
    <span>Conformal stretch λ (80% interval)</span><span class="mono">${S.lambda.toFixed(3)}</span>
    <span>Max takes per clip before a rewrite</span><span><select class="inp" id="kSel">${[3, 4, 5, 6].map((v) => `<option ${v === S.K ? 'selected' : ''}>${v}</option>`).join('')}</select></span>
    <span>Min expected saving per extra cut</span><span><select class="inp" id="cutSel">${[0, 0.25, 0.5, 1].map((v) => `<option value="${v}" ${v === S.cutPenalty ? 'selected' : ''}>${usd(v)}</option>`).join('')}</select></span>
  </div>`;
}

// ───────────────────────── wiring ─────────────────────────

function wire() {
  $('budget').addEventListener('input', () => {
    const v = parseFloat($('budget').value.replace(/[$,]/g, ''));
    if (Number.isFinite(v) && v > 0) {
      S.cap = v;
      render();
    }
  });
  $('drawerClose').addEventListener('click', () => {
    $('drawer').hidden = true;
    $('railForecast').classList.remove('on');
  });
  $('railForecast').addEventListener('click', () => ($('drawer').hidden ? openDrawer() : $('drawerClose').click()));
  $('drawerTabs').querySelectorAll('.seg-tab').forEach((b) => b.addEventListener('click', () => { S.drawerTab = b.dataset.tab; renderDrawer(); }));
  window.addEventListener('resize', renderCanvas);
  const params = new URLSearchParams(location.search);
  if (params.get('drawer')) openDrawer(params.get('drawer'));
  if (params.get('plan') === 'replanned') for (const s of S.project.shots) S.replan[s.id] = S.optById[s.id].best !== S.optById[s.id].drafted;
}

load().catch((e) => {
  document.body.insertAdjacentHTML('beforeend', `<pre style="position:fixed;bottom:0;left:0;color:#f87171">${esc(e.stack || e)}</pre>`);
});
