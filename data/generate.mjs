// Seeded synthetic data for the Run Forecast demo.
//   node data/generate.mjs   -> writes data/history.json and data/project.json
//
// history.json: 600 past PAI-Pro-style projects. Each has a drafted plan
// (anchors + clips) and the take log that actually happened: every take of
// every job until it passed the alignment check or hit 4 takes (3 for
// anchors), with infra failures (content filter, transient, timeout) that
// are billed but return nothing.
//
// The ground truth below is an ASSUMPTION about how alignment pass rates
// behave (longer clips, more characters and more motion fail more often;
// each project also has its own difficulty). It is not Utopai data. The
// forecaster never sees these coefficients; it only sees the take log.

import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mulberry32, normal, videoCost, anchorCost, sigmoid } from '../core/forecast.mjs';

export const SEED = 20261110;
export const TRUTH = {
  intercept: 2.34,
  perSecond: -0.092,
  perSecond2: -0.002,
  perCharacter: -0.45,
  perMotion: -0.45,
  v25: 0.5,
  res1080: -0.15,
  res480: 0.1,
  hasRef: 0.35,
  projectSd: 0.7,
  failRate: 0.03,
  anchorLogit: { character_sheet: 1.5, location: 2.2, voice: 3.5 },
};

export function trueLogit(seg, projectEffect) {
  const T = TRUTH;
  return (
    T.intercept +
    T.perSecond * seg.duration +
    T.perSecond2 * seg.duration * seg.duration +
    T.perCharacter * seg.chars +
    T.perMotion * seg.motion +
    (seg.version === '2.5' ? T.v25 : 0) +
    (seg.res === '1080p' ? T.res1080 : seg.res === '480p' ? T.res480 : 0) +
    (seg.hasRef ? T.hasRef : 0) +
    projectEffect
  );
}

function pick(rng, arr, w) {
  const t = w.reduce((a, b) => a + b, 0);
  let u = rng() * t;
  for (let i = 0; i < arr.length; i++) {
    u -= w[i];
    if (u <= 0) return arr[i];
  }
  return arr[arr.length - 1];
}

function runTakes(rng, job, pTrue, K) {
  const takes = [];
  for (let k = 1; k <= K; k++) {
    let outcome;
    if (rng() < TRUTH.failRate) outcome = 'failed';
    else outcome = rng() < pTrue ? 'pass' : 'mismatch';
    takes.push({ ...job, take: k, outcome });
    if (outcome === 'pass') break;
  }
  return takes;
}

export function generateHistory(seed = SEED, nProjects = 600) {
  const rng = mulberry32(seed);
  const projects = [];
  for (let p = 0; p < nProjects; p++) {
    const effect = normal(rng) * TRUTH.projectSd;
    const res = pick(rng, ['720p', '480p', '1080p'], [0.7, 0.18, 0.12]);
    const nShots = 4 + Math.floor(rng() * 6);
    const nChars = 1 + Math.floor(rng() * 3);
    const nLocs = 1 + Math.floor(rng() * 3);
    const anchors = [];
    for (let i = 0; i < nChars; i++) anchors.push({ kind: 'character_sheet', label: `char_${i + 1}` });
    for (let i = 0; i < nLocs * 2; i++) anchors.push({ kind: 'location', label: `loc_${i + 1}` });
    for (let i = 0; i < nChars + 1; i++) anchors.push({ kind: 'voice', label: `voice_${i + 1}` });
    const clips = [];
    for (let s = 0; s < nShots; s++) {
      const version = rng() < 0.2 ? '2.5' : '2.0';
      const duration = version === '2.5' ? 5 + Math.floor(rng() * 21) : 4 + Math.floor(rng() * 12);
      const chars = pick(rng, [0, 1, 2, 3], [0.2, 0.4, 0.3, 0.1]);
      const motion = pick(rng, [0, 1, 2], [0.4, 0.4, 0.2]);
      clips.push({ shot: s + 1, version, duration, chars: Math.min(chars, nChars), motion, res, hasRef: chars > 0 && rng() < 0.9 });
    }
    const takes = [];
    for (const a of anchors) {
      const pT = sigmoid(TRUTH.anchorLogit[a.kind] + effect * 0.5);
      takes.push(...runTakes(rng, { kind: a.kind, label: a.label, cost: anchorCost(a.kind) }, pT, 3));
    }
    for (const c of clips) {
      const pT = sigmoid(trueLogit(c, effect));
      takes.push(...runTakes(rng, { kind: 'video', ...c, cost: videoCost(c.version, c.res, c.duration) }, pT, 4));
    }
    projects.push({ id: `proj_${String(p + 1).padStart(3, '0')}`, anchors, clips, takes });
  }
  return { seed, generated_by: 'data/generate.mjs', projects };
}

// The project on the demo canvas: "The Night Ferry", 6 shots, 78 s.
// Beats come from the script breakdown; a shot can only be cut at a beat.
export const PROJECT = {
  title: 'The Night Ferry',
  res: '720p',
  anchors: [
    { kind: 'character_sheet', label: 'Ines, ferry captain (sheet)' },
    { kind: 'character_sheet', label: 'Teo, stowaway (sheet)' },
    { kind: 'location', label: 'Ferry deck, night' },
    { kind: 'location', label: 'Ferry deck, storm' },
    { kind: 'location', label: 'Engine room' },
    { kind: 'location', label: 'Lighthouse at dawn' },
    { kind: 'voice', label: 'Ines voice' },
    { kind: 'voice', label: 'Teo voice' },
    { kind: 'voice', label: 'Narrator VO' },
  ],
  shots: [
    {
      id: 1,
      title: 'The crossing',
      heading: 'EXT. FERRY DECK - NIGHT',
      hasRef: true,
      beats: [
        { id: '1a', t0: 0, t1: 6, chars: 0, motion: 1, text: 'Wide. The old ferry cuts across black water, one lamp swinging at the bow.' },
        { id: '1b', t0: 6, t1: 12, chars: 1, motion: 1, text: 'INES (50s, oilskin coat) walks the rail with a lantern, checking the ropes.' },
      ],
    },
    {
      id: 2,
      title: 'Someone below',
      heading: 'INT. ENGINE ROOM - NIGHT',
      hasRef: true,
      beats: [
        { id: '2a', t0: 0, t1: 5, chars: 1, motion: 0, text: 'Ines stops. A clang from behind the boiler. She lifts the lantern.' },
        { id: '2b', t0: 5, t1: 10, chars: 2, motion: 2, text: 'TEO (11, oversized hoodie) bolts from behind the pipes toward the ladder.' },
        { id: '2c', t0: 10, t1: 15, chars: 2, motion: 2, text: 'Ines catches his sleeve at the bottom rung. He twists, then gives up.' },
      ],
    },
    {
      id: 3,
      title: 'The deal',
      heading: 'INT. ENGINE ROOM - CONTINUOUS',
      hasRef: true,
      beats: [
        { id: '3a', t0: 0, t1: 5, chars: 2, motion: 0, text: 'TEO: "I just need to get to the lighthouse." Ines studies him.' },
        { id: '3b', t0: 5, t1: 10, chars: 2, motion: 0, text: 'INES: "Then you work the crossing like everyone else." She hands him gloves.' },
      ],
    },
    {
      id: 4,
      title: 'The squall',
      heading: 'EXT. FERRY DECK - STORM',
      hasRef: true,
      beats: [
        { id: '4a', t0: 0, t1: 6, chars: 0, motion: 2, text: 'A squall hits. Rain sheets sideways across the deck as the bow pitches into a wave.' },
        { id: '4b', t0: 6, t1: 15, chars: 2, motion: 2, text: 'Ines and Teo haul the bow line together, boots sliding, until the cleat holds.' },
      ],
    },
    {
      id: 5,
      title: 'The map',
      heading: 'INT. WHEELHOUSE - NIGHT',
      hasRef: true,
      beats: [
        { id: '5a', t0: 0, t1: 7, chars: 2, motion: 0, text: 'Teo unfolds a damp hand-drawn map. Ines leans in under the chart lamp.' },
        { id: '5b', t0: 7, t1: 14, chars: 0, motion: 0, text: 'Insert. The map: a lighthouse circled in red, a date written beside it.' },
      ],
    },
    {
      id: 6,
      title: 'Landfall',
      heading: 'EXT. LIGHTHOUSE - DAWN',
      hasRef: false,
      beats: [
        { id: '6a', t0: 0, t1: 12, chars: 0, motion: 1, text: 'Wide. The ferry rounds the point as the lighthouse beam fades into first light.' },
      ],
    },
  ],
};

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const here = dirname(fileURLToPath(import.meta.url));
  const project = { ...PROJECT, shots: PROJECT.shots.map((s) => ({ ...s, res: PROJECT.res })) };
  writeFileSync(join(here, 'history.json'), JSON.stringify(generateHistory()));
  writeFileSync(join(here, 'project.json'), JSON.stringify(project, null, 1));
  const h = generateHistory();
  const n = h.projects.reduce((s, p) => s + p.takes.length, 0);
  console.log(`wrote ${h.projects.length} projects, ${n} takes`);
}
