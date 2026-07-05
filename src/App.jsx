import React, { useState, useCallback, useEffect, useRef } from "react";
import {
  ComposedChart, Line, XAxis, YAxis, Tooltip, ReferenceLine,
  ResponsiveContainer, CartesianGrid,
} from "recharts";

/* ————————————————————————————————————————————————
   STRUCTURED PRODUCTS PRICER — v5
   Product family: TARF → { Vanilla TARF, Liability Knock Out TARF }
   Display convention: no minus sign anywhere — negatives shown as (1,234)
   Engine: BSM Monte Carlo, USD measure, S = EURUSD
   CIV: 1 CIV = 100 figures, 1 figure = 0.01 → target = CIV × 1.00
   ———————————————————————————————————————————————— */

// ---------- rng ----------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeNormals(nPaths, nSteps, seed) {
  const half = Math.ceil(nPaths / 2);
  const z = new Float64Array(nPaths * nSteps);
  const rnd = mulberry32(seed);
  for (let p = 0; p < half; p++) {
    for (let s = 0; s < nSteps; s++) {
      let u1 = rnd(); if (u1 < 1e-12) u1 = 1e-12;
      const u2 = rnd();
      const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      z[p * nSteps + s] = g;
      const q = p + half;
      if (q < nPaths) z[q * nSteps + s] = -g;
    }
  }
  return z;
}
function makeUniforms(nPaths, nSteps, seed) {
  const u = new Float64Array(nPaths * nSteps);
  const rnd = mulberry32(seed);
  for (let i = 0; i < u.length; i++) u[i] = rnd();
  return u;
}
// ---------- schedule ----------
function addMonthsClamped(d, m) {
  const day = d.getDate();
  const nd = new Date(d.getTime());
  nd.setDate(1); nd.setMonth(nd.getMonth() + m);
  const last = new Date(nd.getFullYear(), nd.getMonth() + 1, 0).getDate();
  nd.setDate(Math.min(day, last));
  return nd;
}
function buildSchedule(startISO, nFix, freq) {
  const start = new Date(startISO + "T00:00:00");
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const dates = [];
  for (let i = 1; i <= nFix; i++) {
    let d;
    if (freq === "Monthly") d = addMonthsClamped(start, i);
    else {
      const days = freq === "Weekly" ? 7 : 14;
      d = new Date(start.getTime() + i * days * 86400000);
    }
    dates.push(d);
  }
  const taus = dates.map(d => Math.max((d - today) / 86400000 / 365, 1 / 365));
  return { dates, taus, maturity: dates[nFix - 1] };
}
// ---------- closed form Garman Kohlhagen (for the DCD) ----------
function normCdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  const pr = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x > 0 ? 1 - pr : pr;
}
function gk(S, K, T, rd, rf, sig) {
  const sq = Math.max(sig * Math.sqrt(T), 1e-9);
  const d1 = (Math.log(S / K) + (rd - rf + 0.5 * sig * sig) * T) / sq;
  const d2 = d1 - sq;
  return {
    call: S * Math.exp(-rf * T) * normCdf(d1) - K * Math.exp(-rd * T) * normCdf(d2),
    put: K * Math.exp(-rd * T) * normCdf(-d2) - S * Math.exp(-rf * T) * normCdf(-d1),
    d1, d2,
  };
}
function accrualFrac(start, end, conv) {
  const days = Math.max((end - start) / 86400000, 1);
  if (conv === "ACT/360") return days / 360;
  if (conv === "ACT/ACT") return days / 365.25;
  if (conv === "30/360") {
    const d1 = Math.min(start.getDate(), 30), d2 = Math.min(end.getDate(), 30);
    return (360 * (end.getFullYear() - start.getFullYear())
      + 30 * (end.getMonth() - start.getMonth()) + (d2 - d1)) / 360;
  }
  return days / 365; // ACT/365
}
function dcdMatDate(startISO, term) {
  const s = new Date(startISO + "T00:00:00");
  if (term === "1W") return new Date(s.getTime() + 7 * 86400000);
  if (term === "2W") return new Date(s.getTime() + 14 * 86400000);
  const m = { "1M": 1, "2M": 2, "3M": 3, "6M": 6, "12M": 12 }[term] || 1;
  return addMonthsClamped(s, m);
}

// ---------- engine ----------
function priceEngine(params, z, u, nPaths, taus, collect) {
  const { S0, K, B, L, target, sigma, rd, rf, omega, amtEURperFix, koConv,
          lkoOn, H, lkoStyle, lkoVariant, ekiOn, E,
          pivotOn, kLow, kHigh, pivotL, pivotEkiOn, eLow, eHigh, payAtMat,
          accOn, koLevel, accStyle, countOn, targetCount } = params;
  const n = taus.length;
  const mu = rd - rf;
  const dt = new Float64Array(n), drift = new Float64Array(n),
        volS = new Float64Array(n), df = new Float64Array(n);
  let prev = 0;
  for (let i = 0; i < n; i++) {
    dt[i] = taus[i] - prev; prev = taus[i];
    drift[i] = (mu - 0.5 * sigma * sigma) * dt[i];
    volS[i] = sigma * Math.sqrt(Math.max(dt[i], 0));
    df[i] = Math.exp(-rd * taus[i]);
  }
  const amer = lkoOn && lkoStyle === "American";
  const accAmer = accOn && accStyle === "American";
  const dfn = Math.exp(-rd * taus[n - 1]); // single discount factor when everything pays at maturity (ZC)
  const stats = collect ? { alive: new Float64Array(n), cf: new Float64Array(n) } : null;
  let sum = 0, sumSq = 0, koCount = 0, lifeSum = 0, lkoCount = 0, ekiCount = 0;
  for (let p = 0; p < nPaths; p++) {
    let S = S0, Sp = S0, acc = 0, gains = 0, pv = 0, alive = true, life = n,
        lkoHit = false, lkoCounted = false, ekiTouched = false;
    const base = p * n;
    for (let i = 0; i < n; i++) {
      Sp = S;
      S = S * Math.exp(drift[i] + volS[i] * z[base + i]);
      if (!alive) continue;
      if (stats) stats.alive[i]++;
      let cash = 0;
      let terminate = false;
      // ---- accumulator KO barrier (gain side): cancels this and all remaining fixings ----
      if (accOn) {
        let hit = omega === 1 ? S >= koLevel : S <= koLevel;
        if (!hit && accAmer && volS[i] > 0) {
          const a1 = omega === 1 ? Math.log(koLevel / Sp) : Math.log(Sp / koLevel);
          const a2 = omega === 1 ? Math.log(koLevel / S) : Math.log(S / koLevel);
          if (a1 > 0 && a2 > 0) {
            const pHit = Math.exp(-2 * a1 * a2 / (volS[i] * volS[i]));
            if (u[base + i] < pHit) hit = true;
          }
        }
        if (hit) { alive = false; life = i + 1; koCount++; continue; }
      }
      // ---- LKO monitoring ----
      if (lkoOn && !lkoHit) {
        const beyondNow = omega === 1 ? S <= H : S >= H;
        if (beyondNow) lkoHit = true;
        else if (amer && volS[i] > 0) {
          const a1 = omega === 1 ? Math.log(Sp / H) : Math.log(H / Sp);
          const a2 = omega === 1 ? Math.log(S / H) : Math.log(H / S);
          if (a1 > 0 && a2 > 0) {
            const pHit = Math.exp(-2 * a1 * a2 / (volS[i] * volS[i]));
            if (u[base + i] < pHit) lkoHit = true;
          }
        }
        if (lkoHit) {
          if (!lkoCounted) { lkoCount++; lkoCounted = true; }
          if (lkoVariant === "Accelerated") {
            cash = amtEURperFix * Math.max(target - acc, 0);
            terminate = true;
          }
        }
      }
      if (!terminate) {
        // ---- fixing payoff ----
        const intr = pivotOn
          ? (S >= pivotL ? kHigh - S : S - kLow)
          : omega * (S - K);
        if (intr > 0) {
          if (countOn) {
            gains++;
            if (gains >= targetCount) {
              cash = amtEURperFix * (koConv === "none" ? 0 : intr);
              terminate = true; koCount++;
            } else cash = amtEURperFix * intr;
          } else {
            const newAcc = acc + intr;
            if (newAcc >= target) {
              let pay = intr;
              if (koConv === "capped") pay = Math.max(target - acc, 0);
              else if (koConv === "none") pay = 0;
              cash = amtEURperFix * pay;
              terminate = true; koCount++;
            } else { acc = newAcc; cash = amtEURperFix * intr; }
          }
        } else {
          if (pivotOn) {
            if (pivotEkiOn) {
              const knockedIn = S >= eHigh || S <= eLow;
              if (knockedIn) { cash = amtEURperFix * L * intr; ekiTouched = true; }
            } else {
              cash = amtEURperFix * L * intr;
            }
          } else if (lkoOn && lkoHit) {
            // leverage knocked out to 0x, no obligation
          } else if (ekiOn) {
            const knockedIn = omega === 1 ? S < E : S > E;
            if (knockedIn) { cash = amtEURperFix * L * intr; ekiTouched = true; }
          } else {
            const beyond = omega === 1 ? S < B : S > B;
            cash = amtEURperFix * (beyond ? L : 1) * intr;
          }
        }
      }
      const dfPay = payAtMat ? dfn : df[i];
      pv += dfPay * cash;
      if (stats) stats.cf[i] += dfPay * cash;
      if (terminate) { alive = false; life = i + 1; }
    }
    if (ekiTouched) ekiCount++;
    lifeSum += life; sum += pv; sumSq += pv * pv;
  }
  if (stats) for (let i = 0; i < n; i++) { stats.alive[i] /= nPaths; stats.cf[i] /= nPaths; }
  const pv = sum / nPaths;
  const varP = Math.max(sumSq / nPaths - pv * pv, 0);
  return { pv, se: Math.sqrt(varP / nPaths), koProb: koCount / nPaths,
           lkoProb: lkoCount / nPaths, ekiProb: ekiCount / nPaths,
           expLifeFix: lifeSum / nPaths, stats };
}
/* ---------- formatting: NO minus sign anywhere ----------
   negatives are rendered accounting style: (1,234)              */
const raw = (x, d) => Math.abs(x).toLocaleString("en-US",
  { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = (x, d = 0) =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${raw(x, d)})` : raw(x, d);
const fmtSigned = (x, d = 0) =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${raw(x, d)})` : "+" + raw(x, d);
const bigRaw = a =>
  a >= 1e9 ? (a / 1e9).toFixed(2) + "B" :
  a >= 1e6 ? (a / 1e6).toFixed(2) + "M" :
  a >= 1e4 ? (a / 1e3).toFixed(1) + "k" :
  a.toLocaleString("en-US", { maximumFractionDigits: a >= 100 ? 0 : 2 });
const fmtBig = x =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${bigRaw(Math.abs(x))})` : bigRaw(Math.abs(x));
const fmtBigSigned = x =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${bigRaw(Math.abs(x))})` : "+" + bigRaw(Math.abs(x));
const fmtRate = (x, d = 4) =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${Math.abs(x).toFixed(d)})` : x.toFixed(d);
const GLOBAL_CSS = `
@import url('https://cdnjs.cloudflare.com/ajax/libs/inter-ui/3.19.3/inter.css');
@keyframes spFade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.sp-fade { animation: spFade .4s cubic-bezier(.22,.8,.36,1); }
.sp-input { transition: border-color .16s ease, box-shadow .16s ease, background .16s ease; }
.sp-input:focus { border-color: #4A7DF0 !important; box-shadow: 0 0 0 3px rgba(74,125,240,.16); background: #1B2130 !important; }
select.sp-input option { background: #13161F; color: #E7EDF9; }
.sp-click { transition: border-color .18s ease, background .18s ease, transform .18s ease, box-shadow .18s ease; }
.sp-click:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(0,0,0,.45); border-color: rgba(74,125,240,.55) !important; }
.sp-btn { transition: filter .15s ease, transform .15s ease, box-shadow .15s ease; }
.sp-btn:hover:not(:disabled) { filter: brightness(1.12); transform: translateY(-1px); }
.sp-btn:active:not(:disabled) { transform: translateY(0); filter: brightness(.98); }
.sp-row { transition: background .12s ease; }
.sp-row:hover { background: rgba(74,125,240,.07); }
.sp-scroll { scrollbar-width: thin; scrollbar-color: #2A3140 transparent; }
.sp-scroll::-webkit-scrollbar { width: 8px; height: 8px; }
.sp-scroll::-webkit-scrollbar-thumb { background: #2A3140; border-radius: 8px; }
.sp-scroll::-webkit-scrollbar-thumb:hover { background: #39424F; }
.sp-scroll::-webkit-scrollbar-track { background: transparent; }
::selection { background: rgba(74,125,240,.35); }
`;
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const fmtDate = d => d ? `${String(d.getDate()).padStart(2,"0")} ${MONTHS[d.getMonth()]} ${d.getFullYear()}` : "…";

/* ———————— payoff diagram (marketing style) ———————— */
function PayoffDiagram({ res, C, mono, sans }) {
  if (!res) return null;
  const { K, B, L, omega, amtEURperFix: A, S0, lkoOn, H, ekiOn, E, accOn, koLevel,
          countOn, targetCount } = res;
  const t = accOn ? Math.abs(koLevel - K)
    : countOn ? Math.max(Math.abs(S0 - K) * 1.8, 0.03 * K)
    : res.targetFig / 100;
  const KO = K + omega * t;
  const W = 880, HH = 380, mL = 84, mR = 40, mT = 46, mB = 74;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const barrierS = lkoOn ? H : ekiOn ? E : null;
  const lossEnd = barrierS != null ? barrierS : K - omega * 0.55 * t;
  const lossDrawEnd = barrierS != null ? barrierS - omega * 0.22 * t : lossEnd;
  let x0 = Math.min(lossDrawEnd, B, K, KO, S0) - 0.06 * t;
  let x1 = Math.max(lossDrawEnd, B, K, KO, S0) + 0.30 * t;
  const pay = s => {
    const intr = omega * (s - K);
    if (intr > 0) return A * Math.min(intr, t);
    if (lkoOn) {
      const beyondLKO = omega === 1 ? s <= H : s >= H;
      if (beyondLKO) return 0;
    }
    if (ekiOn) {
      const knockedIn = omega === 1 ? s < E : s > E;
      return knockedIn ? A * L * intr : 0;
    }
    const beyond = omega === 1 ? s < B : s > B;
    return A * (beyond ? L : 1) * intr;
  };
  const cv = 1 / S0;
  const lossFloorRef = lkoOn ? H : ekiOn ? lossDrawEnd : (omega === 1 ? x0 : x1);
  const lossFloorVal = A * L * omega * (lossFloorRef - K);
  const yTop = A * t;
  const yBot = Math.min(lossFloorVal, 0);
  const yMax = yTop * 1.18, yMin = yBot * 1.12 - yTop * 0.05;
  const X = s => mL + ((s - x0) / (x1 - x0)) * iw;
  const Y = v => mT + ((yMax - v) / (yMax - yMin)) * ih;
  const Y0 = Y(0);
  const gainB = X(KO);
  const bx = X(B);
  const hasLevGap = Math.abs(B - K) > 1e-9;
  const lossEdgeS = lkoOn ? H : ekiOn ? lossDrawEnd : (omega === 1 ? x0 : x1);
  const dir = omega === 1 ? "Buyer (Long)" : "Seller (Short)";
  const tagW = 26;
  const midGain = (K + KO) / 2;
  const midLoss = ekiOn ? (E + lossDrawEnd) / 2 : (B + lossEdgeS) / 2;
  const midPart = ekiOn ? (K + E) / 2 : null;
  const title = countOn ? "Discrete TARF" : accOn ? "Accumulator"
    : lkoOn ? "Liability Knock Out TARF" : ekiOn ? "EKI TARF" : "Vanilla TARF";
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <text x={W / 2} y={20} textAnchor="middle" fontFamily={sans} fontSize="15" fontWeight="700" fill={C.text}>
        {title}
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        EURUSD {dir} · {fmtBig(A)} EUR per fixing
      </text>
      <line x1={mL} y1={Y0} x2={W - mR + 6} y2={Y0} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${W - mR + 6},${Y0} ${W - mR - 2},${Y0 - 4} ${W - mR - 2},${Y0 + 4}`} fill={C.mute} />
      <line x1={mL} y1={mT - 4} x2={mL} y2={HH - mB + 6} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${mL},${mT - 10} ${mL - 4},${mT - 2} ${mL + 4},${mT - 2}`} fill={C.mute} />
      <text x={mL - 10} y={mT + 4} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>PROFIT</text>
      <text x={mL - 10} y={HH - mB} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>(LOSS)</text>
      <text x={mL - 14} y={(mT + HH - mB) / 2} textAnchor="middle" fontFamily={sans} fontSize="10.5"
        fill={C.mute} transform={`rotate(-90 ${mL - 46} ${(mT + HH - mB) / 2})`} letterSpacing="0.12em">
        DERIVATIVE PAYOFF / FIXING
      </text>
      <text x={mL + iw / 2} y={HH - 16} textAnchor="middle" fontFamily={sans} fontSize="12"
        fontWeight="600" fill={C.mute}>
        Price of underlying at expiration
      </text>
      {!countOn && (<g>
        <line x1={mL} y1={Y(yTop)} x2={gainB} y2={Y(yTop)} stroke={C.line} strokeDasharray="3 4" />
        <text x={mL + 8} y={Y(yTop) - 6} textAnchor="start" fontFamily={mono} fontSize="10.5" fill={C.green}>
          +{fmtBig(yTop * cv)}
        </text>
      </g>)}
      {hasLevGap && !ekiOn && (
        <polyline points={`${X(K)},${Y0} ${bx},${Y(A * omega * (B - K))}`}
          fill="none" stroke={C.red} strokeWidth="3" opacity="0.75" />
      )}
      {!ekiOn && (
        <polyline
          points={`${bx},${Y(hasLevGap ? A * L * omega * (B - K) : 0)} ${X(lossEdgeS)},${Y(A * L * omega * (lossEdgeS - K))}`}
          fill="none" stroke={C.red} strokeWidth="3.4" strokeLinecap="round" />
      )}
      {ekiOn && (<g>
        <line x1={X(K)} y1={Y0} x2={X(E)} y2={Y0}
          stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
        <line x1={X(E)} y1={Y0} x2={X(E)} y2={Y(A * L * omega * (E - K))}
          stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
        <polyline
          points={`${X(E)},${Y(A * L * omega * (E - K))} ${X(lossDrawEnd)},${Y(A * L * omega * (lossDrawEnd - K))}`}
          fill="none" stroke={C.red} strokeWidth="3.4" strokeLinecap="round" />
      </g>)}
      {lkoOn && (<g>
        <line x1={X(H)} y1={Y(A * L * omega * (H - K))} x2={X(H)} y2={Y0}
          stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
        <line x1={X(H)} y1={Y0} x2={omega === 1 ? mL : W - mR} y2={Y0}
          stroke={C.green} strokeWidth="2.6" strokeDasharray="6 5" />
      </g>)}
      <line x1={X(K)} y1={Y0} x2={gainB} y2={Y(yTop)} stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      {!countOn && (<g>
        <line x1={gainB} y1={Y(yTop)} x2={gainB} y2={Y0} stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
        <line x1={gainB} y1={Y0} x2={omega === 1 ? W - mR : mL} y2={Y0}
          stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
      </g>)}
      <line x1={X(K)} y1={Y0} x2={X(K)} y2={Y0 + 26} stroke={C.text} strokeDasharray="3 3" strokeWidth="1" />
      <rect x={X(K) - tagW / 2} y={Y0 + 26} width={tagW} height={20} rx="3" fill="#060A14" stroke={C.text} strokeWidth="0.8" />
      <text x={X(K)} y={Y0 + 40} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill="#FFFFFF">K</text>
      <text x={X(K)} y={Y0 + 60} textAnchor="middle" fontFamily={mono} fontSize="11" fill={C.text}>{fmtRate(K)}</text>
      {hasLevGap && !ekiOn && (<g>
        <line x1={bx} y1={Y0} x2={bx} y2={Y0 + 18} stroke={C.mute} strokeDasharray="3 3" strokeWidth="1" />
        <text x={bx} y={Y0 + 32} textAnchor="middle" fontFamily={mono} fontSize="10" fill={C.mute}>B {fmtRate(B)}</text>
      </g>)}
      {!countOn && (
      <g>
        <path d={`M ${gainB - 17} ${Y0 + 30} h34 v18 h-34 z M ${gainB - 17} ${Y0 + 30} L ${gainB} ${Y0 + 18} L ${gainB + 17} ${Y0 + 30} z`}
          fill={C.red} />
        <text x={gainB} y={Y0 + 43} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill="#FFFFFF">KO</text>
        <text x={gainB} y={Y0 + 62} textAnchor="middle" fontFamily={sans} fontSize="9.5" fontWeight="700" fill={C.red}>{accOn ? "Barrier" : "Target"}</text>
        <text x={gainB} y={Y0 + 76} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(KO)}</text>
      </g>
      )}
      {lkoOn && (<g>
        <path d={`M ${X(H) - 19} ${Y0 + 30} h38 v18 h-38 z M ${X(H) - 19} ${Y0 + 30} L ${X(H)} ${Y0 + 18} L ${X(H) + 19} ${Y0 + 30} z`}
          fill="#6D5FC7" />
        <text x={X(H)} y={Y0 + 43} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill="#FFFFFF">LKO</text>
        <text x={X(H)} y={Y0 + 62} textAnchor="middle" fontFamily={sans} fontSize="9.5" fontWeight="700" fill="#8B7EDB">Barrier</text>
        <text x={X(H)} y={Y0 + 76} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(H)}</text>
      </g>)}
      {ekiOn && (<g>
        <line x1={X(E)} y1={Y0} x2={X(E)} y2={Y0 + 26} stroke="#C89A4B" strokeDasharray="3 3" strokeWidth="1" />
        <path d={`M ${X(E) - 17} ${Y0 + 30} h34 v18 h-34 z M ${X(E) - 17} ${Y0 + 30} L ${X(E)} ${Y0 + 18} L ${X(E) + 17} ${Y0 + 30} z`}
          fill="#C89A4B" />
        <text x={X(E)} y={Y0 + 43} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill="#131007">KI</text>
        <text x={X(E)} y={Y0 + 62} textAnchor="middle" fontFamily={sans} fontSize="9.5" fontWeight="700" fill="#C89A4B">Barrier</text>
        <text x={X(E)} y={Y0 + 76} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(E)}</text>
      </g>)}
      <circle cx={X(S0)} cy={Y0} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      <text x={X(S0)} y={pay(S0) > 0 ? Y0 + 18 : Y0 - 10} textAnchor="middle"
        fontFamily={mono} fontSize="10" fill={C.blue}>
        spot {fmtRate(S0)}
      </text>
      <g>
        <circle cx={X(midGain)} cy={Y(pay(midGain)) - 24} r="14" fill={C.green} />
        <text x={X(midGain)} y={Y(pay(midGain)) - 20} textAnchor="middle" fontFamily={sans}
          fontSize="11" fontWeight="800" fill="#08130C">1x</text>
      </g>
      <g>
        <circle cx={X(midLoss)} cy={Y(A * L * omega * (midLoss - K)) - 24} r="14" fill={C.red} />
        <text x={X(midLoss)} y={Y(A * L * omega * (midLoss - K)) - 20} textAnchor="middle" fontFamily={sans}
          fontSize="11" fontWeight="800" fill="#1A070C">{L}x</text>
      </g>
      {lkoOn && (
        <text x={omega === 1 ? X(H) - 8 : X(H) + 8} textAnchor={omega === 1 ? "end" : "start"}
          y={Y0 - 28} fontFamily={sans} fontSize="10.5" fontWeight="700" fill="#8B7EDB">
          0x, no obligation
        </text>
      )}
      {ekiOn && (
        <text x={X(midPart)} textAnchor="middle" y={Y0 - 26} fontFamily={sans}
          fontSize="10.5" fontWeight="700" fill={C.green}>
          participation at market
        </text>
      )}
      <text x={gainB > W - mR - 170 ? gainB - 10 : gainB + 8}
        textAnchor={gainB > W - mR - 170 ? "end" : "start"}
        y={Y(yTop) - 10} fontFamily={mono} fontSize="10.5" fill={C.green}>
        {countOn ? "stops after gain #" + targetCount
          : accOn ? "cancelled at KO" : "max +" + fmtBig(yTop * cv) + " EUR at KO"}
      </text>
    </svg>
  );
}

/* ———————— vanilla option payoff diagram ———————— */
function VanillaDiagram({ res, C, mono, sans }) {
  if (!res) return null;
  const { S0, K, side, type, eurN, premEUR, breakeven } = res;
  const sign = side === "Buy" ? 1 : -1;
  const om = type === "Call" ? 1 : -1;
  const cvE = 1 / S0;
  const net = s => sign * (eurN * Math.max(om * (s - K), 0) * cvE - premEUR);
  const W = 880, HH = 380, mL = 100, mR = 40, mT = 46, mB = 74;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const span = Math.max(Math.abs(breakeven - K) * 2.2, 0.06 * K);
  let x0 = Math.min(K, breakeven, S0) - 0.8 * span;
  let x1 = Math.max(K, breakeven, S0) + 0.8 * span;
  const vals = [net(x0), net(x1), net(K), 0];
  const yMax = Math.max(...vals) * 1.2 + premEUR * 0.2;
  const yMin = Math.min(...vals) * 1.2 - premEUR * 0.2;
  const X = s => mL + ((s - x0) / (x1 - x0)) * iw;
  const Y = v => mT + ((yMax - v) / (yMax - yMin)) * ih;
  const Y0 = Y(0);
  // polyline split at the breakeven into loss (red) and gain (green) parts
  const seg = (a, b) => {
    const pts = [];
    for (let i = 0; i <= 24; i++) { const s = a + ((b - a) * i) / 24; pts.push(`${X(s)},${Y(net(s))}`); }
    return pts.join(" ");
  };
  const leftNeg = net(x0) < 0;
  const tagW = 26;
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <text x={W / 2} y={20} textAnchor="middle" fontFamily={sans} fontSize="15" fontWeight="700" fill={C.text}>
        {side} EURUSD {type}
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        {type === "Call" ? "EUR call / USD put" : "EUR put / USD call"} · {fmtBig(eurN)} EUR notional ·
        payoff at expiry net of premium, in EUR
      </text>
      <line x1={mL} y1={Y0} x2={W - mR + 6} y2={Y0} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${W - mR + 6},${Y0} ${W - mR - 2},${Y0 - 4} ${W - mR - 2},${Y0 + 4}`} fill={C.mute} />
      <line x1={mL} y1={mT - 4} x2={mL} y2={HH - mB + 6} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${mL},${mT - 10} ${mL - 4},${mT - 2} ${mL + 4},${mT - 2}`} fill={C.mute} />
      <text x={mL - 10} y={mT + 4} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>PROFIT</text>
      <text x={mL - 10} y={HH - mB} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>(LOSS)</text>
      <text x={mL + iw / 2} y={HH - 16} textAnchor="middle" fontFamily={sans} fontSize="12"
        fontWeight="600" fill={C.mute}>
        EURUSD at expiry
      </text>
      {/* premium guide */}
      <line x1={mL} y1={Y(-sign * premEUR)} x2={W - mR} y2={Y(-sign * premEUR)}
        stroke={C.line} strokeDasharray="3 4" />
      <text x={mL - 8} y={Y(-sign * premEUR) + 4} textAnchor="end" fontFamily={mono} fontSize="10.5"
        fill={sign === 1 ? C.red : C.green}>
        {sign === 1 ? `(${bigRaw(premEUR)})` : "+" + bigRaw(premEUR)}
      </text>
      <text x={W - mR - 4} y={Y(-sign * premEUR) - 6} textAnchor="end" fontFamily={sans} fontSize="9.5" fill={C.faint}>
        premium {sign === 1 ? "paid" : "received"}
      </text>
      {/* payoff, colored by sign around breakeven */}
      <polyline points={seg(x0, breakeven)} fill="none" stroke={leftNeg ? C.red : C.green}
        strokeWidth="3.4" strokeLinecap="round" />
      <polyline points={seg(breakeven, x1)} fill="none" stroke={leftNeg ? C.green : C.red}
        strokeWidth="3.4" strokeLinecap="round" />
      {/* K tag */}
      <line x1={X(K)} y1={Y0} x2={X(K)} y2={Y0 + 26} stroke={C.text} strokeDasharray="3 3" strokeWidth="1" />
      <rect x={X(K) - tagW / 2} y={Y0 + 26} width={tagW} height={20} rx="3" fill="#060A14" stroke={C.text} strokeWidth="0.8" />
      <text x={X(K)} y={Y0 + 40} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill="#FFFFFF">K</text>
      <text x={X(K)} y={Y0 + 60} textAnchor="middle" fontFamily={mono} fontSize="11" fill={C.text}>{fmtRate(K)}</text>
      {/* breakeven */}
      <line x1={X(breakeven)} y1={mT + 22} x2={X(breakeven)} y2={Y0} stroke={C.amber} strokeDasharray="4 3" strokeWidth="1.4" />
      <text x={X(breakeven)} y={mT + 16} textAnchor="middle" fontFamily={mono} fontSize="10" fill={C.amber}>
        breakeven {fmtRate(breakeven)}
      </text>
      {/* spot */}
      <circle cx={X(S0)} cy={Y0} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      <text x={X(S0)} y={net(S0) > 0 ? Y0 + 18 : Y0 - 10} textAnchor="middle"
        fontFamily={mono} fontSize="10" fill={C.blue}>
        spot {fmtRate(S0)}
      </text>
    </svg>
  );
}

/* ———————— DCD redemption diagram ———————— */
function DCDDiagram({ res, C, mono, sans }) {
  if (!res) return null;
  const { S0, K, depCcy, N, breakeven } = res;
  const conv = depCcy === "EUR"; // conversion happens above K for a EUR deposit
  const alt = depCcy === "EUR" ? "USD" : "EUR";
  const W = 880, HH = 380, mL = 100, mR = 40, mT = 46, mB = 74;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const red = N; // nominal only, coupon excluded
  const redeem = s => conv ? (s <= K ? red : red * K / s) : (s >= K ? red : red * s / K);
  const span = 0.16 * K;
  let x0 = conv ? K - 0.7 * span : K - 1.6 * span;
  let x1 = conv ? K + 1.6 * span : K + 0.7 * span;
  x0 = Math.min(x0, S0 - 0.15 * span, breakeven - 0.15 * span);
  x1 = Math.max(x1, S0 + 0.15 * span, breakeven + 0.15 * span);
  const edge = conv ? x1 : x0;
  const yMax = red * 1.05;
  const yMin = Math.min(redeem(edge), N) - (yMax - red) * 3;
  const X = s => mL + ((s - x0) / (x1 - x0)) * iw;
  const Y = v => mT + ((yMax - v) / (yMax - yMin)) * ih;
  const Ybase = HH - mB;
  // sampled converted branch
  const pts = [];
  const from = conv ? K : x0, to = conv ? x1 : K;
  for (let i = 0; i <= 40; i++) {
    const s = from + ((to - from) * i) / 40;
    pts.push(`${X(s)},${Y(redeem(s))}`);
  }
  const tagW = 26;
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <text x={W / 2} y={20} textAnchor="middle" fontFamily={sans} fontSize="15" fontWeight="700" fill={C.text}>
        Dual Currency Deposit
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        {depCcy} deposit · {alt} alternative · final redemption of nominal, excluding coupon, in {depCcy}
      </text>
      {/* axes */}
      <line x1={mL} y1={Ybase} x2={W - mR + 6} y2={Ybase} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${W - mR + 6},${Ybase} ${W - mR - 2},${Ybase - 4} ${W - mR - 2},${Ybase + 4}`} fill={C.mute} />
      <line x1={mL} y1={mT - 4} x2={mL} y2={Ybase + 6} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${mL},${mT - 10} ${mL - 4},${mT - 2} ${mL + 4},${mT - 2}`} fill={C.mute} />
      <text x={mL - 14} y={(mT + Ybase) / 2} textAnchor="middle" fontFamily={sans} fontSize="10.5"
        fill={C.mute} transform={`rotate(-90 ${mL - 56} ${(mT + Ybase) / 2})`} letterSpacing="0.12em">
        REDEMPTION IN {depCcy}
      </text>
      <text x={mL + iw / 2} y={HH - 16} textAnchor="middle" fontFamily={sans} fontSize="12"
        fontWeight="600" fill={C.mute}>
        EURUSD at maturity
      </text>
      {/* nominal guide */}
      <line x1={mL} y1={Y(red)} x2={W - mR} y2={Y(red)} stroke={C.line} strokeDasharray="3 4" />
      <text x={mL + 8} y={Y(red) - 6} textAnchor="start" fontFamily={mono} fontSize="10.5" fill={C.green}>
        {fmtBig(red)}
      </text>
      <text x={W - mR - 4} y={Y(red) - 6} textAnchor="end" fontFamily={sans} fontSize="9.5" fill={C.faint}>
        nominal
      </text>
      {/* protected flat leg (green) */}
      <line x1={conv ? X(x0) : X(K)} y1={Y(red)} x2={conv ? X(K) : X(x1)} y2={Y(red)}
        stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      {/* converted leg (red curve) */}
      <polyline points={pts.join(" ")} fill="none" stroke={C.red} strokeWidth="3.4" strokeLinecap="round" />
      {/* K tag */}
      <line x1={X(K)} y1={Ybase} x2={X(K)} y2={Ybase + 26} stroke={C.text} strokeDasharray="3 3" strokeWidth="1" />
      <rect x={X(K) - tagW / 2} y={Ybase + 26} width={tagW} height={20} rx="3" fill="#060A14" stroke={C.text} strokeWidth="0.8" />
      <text x={X(K)} y={Ybase + 40} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill="#FFFFFF">K</text>
      <text x={X(K)} y={Ybase + 60} textAnchor="middle" fontFamily={mono} fontSize="11" fill={C.text}>{fmtRate(K)}</text>
      {/* breakeven (incl. coupon) */}
      <line x1={X(breakeven)} y1={mT + 22} x2={X(breakeven)} y2={Ybase} stroke={C.amber} strokeDasharray="4 3" strokeWidth="1.4" />
      <text x={X(breakeven)} y={mT + 16} textAnchor="middle" fontFamily={mono} fontSize="10" fill={C.amber}>
        breakeven incl. coupon {fmtRate(breakeven)}
      </text>
      {/* spot */}
      <circle cx={X(S0)} cy={Ybase} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      <text x={X(S0)} y={Ybase - 10} textAnchor="middle" fontFamily={mono} fontSize="10" fill={C.blue}>
        spot {fmtRate(S0)}
      </text>
      {/* region labels */}
      <text x={conv ? X((x0 + K) / 2) : X((K + x1) / 2)} y={Y(red) - 12} textAnchor="middle"
        fontFamily={sans} fontSize="10.5" fontWeight="700" fill={C.green}>
        nominal repaid in {depCcy}
      </text>
      <text x={conv ? X((K + x1) / 2) : X((x0 + K) / 2)} y={Y(redeem(conv ? (K + x1) / 2 : (x0 + K) / 2)) - 14}
        textAnchor="middle" fontFamily={sans} fontSize="10.5" fontWeight="700" fill={C.red}>
        converted into {alt} at K
      </text>
    </svg>
  );
}

/* ———————— pivot payoff diagram ———————— */
function PivotPayoffDiagram({ res, C, mono, sans }) {
  if (!res) return null;
  const { kLow: kL, kHigh: kH, pivotL: P, L, amtEURperFix: A, S0,
          pivotEkiOn: ekip, eLow: eL, eHigh: eH } = res;
  const t = res.targetFig / 100;
  const W = 880, HH = 418, mL = 84, mR = 40, mT = 46, mB = 106;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const span = kH - kL;
  let x0 = Math.min(ekip ? eL - 0.18 * span : kL - 0.30 * span, S0 - 0.05 * span);
  let x1 = Math.max(ekip ? eH + 0.18 * span : kH + 0.30 * span, S0 + 0.05 * span);
  const cv = 1 / S0;
  const gLow = P - kL, gHigh = kH - P;
  const pay = s => {
    const intr = s >= P ? kH - s : s - kL;
    if (intr > 0) return A * Math.min(intr, t);
    if (ekip) {
      const knockedIn = s >= eH || s <= eL;
      return knockedIn ? A * L * intr : 0;
    }
    return A * L * intr;
  };
  const capped = t < Math.max(gLow, gHigh);
  const yTop = A * Math.min(Math.max(gLow, gHigh), t);
  const yBot = Math.min(pay(x0 + 0.02 * span), pay(x1 - 0.02 * span));
  const yMax = yTop * 1.22, yMin = yBot * 1.12 - yTop * 0.05;
  const X = s => mL + ((s - x0) / (x1 - x0)) * iw;
  const Y = v => mT + ((yMax - v) / (yMax - yMin)) * ih;
  const Y0 = Y(0);
  const capY = Y(A * t);
  const lowClip = t < gLow;   // low branch reaches the cap
  const highClip = t < gHigh; // high branch reaches the cap
  const hLowTop = A * Math.min(gLow, t);
  const hHighTop = A * Math.min(gHigh, t);
  const wingEndLow = x0 + 0.02 * span, wingEndHigh = x1 - 0.02 * span;
  const wingStartLow = ekip ? eL : kL, wingStartHigh = ekip ? eH : kH;
  const tagW = 30;
  const blackTag = (x, txt, val) => (<g key={txt}>
    <line x1={X(x)} y1={Y0} x2={X(x)} y2={Y0 + 26} stroke={C.text} strokeDasharray="3 3" strokeWidth="1" />
    <rect x={X(x) - tagW / 2} y={Y0 + 26} width={tagW} height={20} rx="3" fill="#060A14" stroke={C.text} strokeWidth="0.8" />
    <text x={X(x)} y={Y0 + 40} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill="#FFFFFF">{txt}</text>
    <text x={X(x)} y={Y0 + 60} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(val)}</text>
  </g>);
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <text x={W / 2} y={20} textAnchor="middle" fontFamily={sans} fontSize="15" fontWeight="700" fill={C.text}>
        {ekip ? "EKI Pivot TARF" : "Pivot TARF"}
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        EURUSD two sided · {fmtBig(A)} EUR per fixing · sells at {fmtRate(kH)} above pivot, buys at {fmtRate(kL)} below
      </text>
      {/* axes */}
      <line x1={mL} y1={Y0} x2={W - mR + 6} y2={Y0} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${W - mR + 6},${Y0} ${W - mR - 2},${Y0 - 4} ${W - mR - 2},${Y0 + 4}`} fill={C.mute} />
      <line x1={mL} y1={mT - 4} x2={mL} y2={HH - mB + 6} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${mL},${mT - 10} ${mL - 4},${mT - 2} ${mL + 4},${mT - 2}`} fill={C.mute} />
      <text x={mL - 10} y={mT + 4} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>PROFIT</text>
      <text x={mL - 10} y={HH - mB} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>(LOSS)</text>
      <text x={mL - 14} y={(mT + HH - mB) / 2} textAnchor="middle" fontFamily={sans} fontSize="10.5"
        fill={C.mute} transform={`rotate(-90 ${mL - 46} ${(mT + HH - mB) / 2})`} letterSpacing="0.12em">
        DERIVATIVE PAYOFF / FIXING
      </text>
      <text x={mL + iw / 2} y={HH - 16} textAnchor="middle" fontFamily={sans} fontSize="12"
        fontWeight="600" fill={C.mute}>
        Price of underlying at expiration
      </text>
      {/* KO target cap guide */}
      {capped && (<g>
        <line x1={mL} y1={capY} x2={W - mR} y2={capY} stroke={C.red} strokeDasharray="6 5" strokeWidth="1.6" />
        <text x={W - mR - 4} y={capY - 6} textAnchor="end" fontFamily={mono} fontSize="10.5" fill={C.red}>
          KO Target · +{fmtBig(A * t * cv)} EUR ({fmt(res.targetFig, 0)} fig)
        </text>
      </g>)}
      {!capped && (<g>
        <line x1={mL} y1={Y(yTop)} x2={X(P)} y2={Y(yTop)} stroke={C.line} strokeDasharray="3 4" />
        <text x={mL + 8} y={Y(yTop) - 6} textAnchor="start" fontFamily={mono} fontSize="10.5" fill={C.green}>
          +{fmtBig(yTop * cv)}
        </text>
      </g>)}
      {/* red wings (from the strikes, or from the KI barriers if EKI pair) */}
      {ekip && (<g>
        <line x1={X(kL)} y1={Y0} x2={X(eL)} y2={Y0}
          stroke={C.green} strokeWidth="2.8" strokeDasharray="6 5" />
        <line x1={X(kH)} y1={Y0} x2={X(eH)} y2={Y0}
          stroke={C.green} strokeWidth="2.8" strokeDasharray="6 5" />
        <line x1={X(eL)} y1={Y0} x2={X(eL)} y2={Y(A * L * (eL - kL))}
          stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
        <line x1={X(eH)} y1={Y0} x2={X(eH)} y2={Y(A * L * (kH - eH))}
          stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
      </g>)}
      <polyline points={`${X(wingStartLow)},${ekip ? Y(A * L * (eL - kL)) : Y0} ${X(wingEndLow)},${Y(A * L * (wingEndLow - kL))}`}
        fill="none" stroke={C.red} strokeWidth="3.4" strokeLinecap="round" />
      <polyline points={`${X(wingStartHigh)},${ekip ? Y(A * L * (kH - eH)) : Y0} ${X(wingEndHigh)},${Y(A * L * (kH - wingEndHigh))}`}
        fill="none" stroke={C.red} strokeWidth="3.4" strokeLinecap="round" />
      {/* green tent, clipped at the cap */}
      {lowClip ? (<g>
        <line x1={X(kL)} y1={Y0} x2={X(kL + t)} y2={capY} stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
        <line x1={X(kL + t)} y1={capY} x2={X(P)} y2={capY} stroke={C.green} strokeWidth="3.4" strokeDasharray="6 5" />
      </g>) : (
        <line x1={X(kL)} y1={Y0} x2={X(P)} y2={Y(A * gLow)} stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      )}
      {highClip ? (<g>
        <line x1={X(P)} y1={capY} x2={X(kH - t)} y2={capY} stroke={C.green} strokeWidth="3.4" strokeDasharray="6 5" />
        <line x1={X(kH - t)} y1={capY} x2={X(kH)} y2={Y0} stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      </g>) : (
        <line x1={X(P)} y1={Y(A * gHigh)} x2={X(kH)} y2={Y0} stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      )}
      {/* pivot connector if branch heights differ */}
      {Math.abs(hLowTop - hHighTop) > 1e-9 && (
        <line x1={X(P)} y1={Y(hLowTop)} x2={X(P)} y2={Y(hHighTop)}
          stroke={C.green} strokeWidth="2" strokeDasharray="3 3" opacity="0.7" />
      )}
      {/* tags */}
      {blackTag(kL, "KL", kL)}
      {blackTag(kH, "KH", kH)}
      <g>
        <line x1={X(P)} y1={Y0} x2={X(P)} y2={Y0 + 48} stroke={C.blue} strokeDasharray="3 3" strokeWidth="1" />
        <rect x={X(P) - 27} y={Y0 + 48} width={54} height={20} rx="10" fill={C.blue} />
        <text x={X(P)} y={Y0 + 62} textAnchor="middle" fontFamily={mono} fontSize="10" fontWeight="700" fill="#FFFFFF">PIVOT</text>
        <text x={X(P)} y={Y0 + 82} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(P)}</text>
      </g>
      {ekip && (<g>
        {[["KI", eL], ["KI", eH]].map(([txt, xv], i) => (
          <g key={i}>
            <line x1={X(xv)} y1={Y0} x2={X(xv)} y2={Y0 + 62} stroke="#C89A4B" strokeDasharray="3 3" strokeWidth="1" />
            <path d={`M ${X(xv) - 17} ${Y0 + 66} h34 v18 h-34 z M ${X(xv) - 17} ${Y0 + 66} L ${X(xv)} ${Y0 + 54} L ${X(xv) + 17} ${Y0 + 66} z`}
              fill="#C89A4B" />
            <text x={X(xv)} y={Y0 + 79} textAnchor="middle" fontFamily={mono} fontSize="11" fontWeight="700" fill="#131007">{txt}</text>
            <text x={X(xv)} y={Y0 + 98} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(xv)}</text>
          </g>
        ))}
        <text x={X((kL + eL) / 2)} textAnchor="middle" y={Y0 - 26} fontFamily={sans}
          fontSize="9.5" fontWeight="700" fill={C.green}>participation</text>
        <text x={X((kH + eH) / 2)} textAnchor="middle" y={Y0 - 26} fontFamily={sans}
          fontSize="9.5" fontWeight="700" fill={C.green}>participation</text>
      </g>)}
      {/* spot */}
      <circle cx={X(S0)} cy={Y0} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      <text x={X(S0)} y={pay(S0) > 0 ? Y0 + 18 : Y0 - 12} textAnchor="middle"
        fontFamily={mono} fontSize="10" fill={C.blue}>
        spot {fmtRate(S0)}
      </text>
      {/* badges */}
      <g>
        <circle cx={X((kL + Math.min(kL + t, P)) / 2)} cy={Y(pay((kL + Math.min(kL + t, P)) / 2)) - 22} r="14" fill={C.green} />
        <text x={X((kL + Math.min(kL + t, P)) / 2)} y={Y(pay((kL + Math.min(kL + t, P)) / 2)) - 18}
          textAnchor="middle" fontFamily={sans} fontSize="11" fontWeight="800" fill="#08130C">1x</text>
      </g>
      <g>
        <circle cx={X((kH + Math.max(kH - t, P)) / 2)} cy={Y(pay((kH + Math.max(kH - t, P)) / 2)) - 22} r="14" fill={C.green} />
        <text x={X((kH + Math.max(kH - t, P)) / 2)} y={Y(pay((kH + Math.max(kH - t, P)) / 2)) - 18}
          textAnchor="middle" fontFamily={sans} fontSize="11" fontWeight="800" fill="#08130C">1x</text>
      </g>
      <g>
        <circle cx={X((wingStartLow + wingEndLow) / 2)} cy={Y(pay((wingStartLow + wingEndLow) / 2)) - 22} r="14" fill={C.red} />
        <text x={X((wingStartLow + wingEndLow) / 2)} y={Y(pay((wingStartLow + wingEndLow) / 2)) - 18}
          textAnchor="middle" fontFamily={sans} fontSize="11" fontWeight="800" fill="#1A070C">{L}x</text>
      </g>
      <g>
        <circle cx={X((wingStartHigh + wingEndHigh) / 2)} cy={Y(pay((wingStartHigh + wingEndHigh) / 2)) - 22} r="14" fill={C.red} />
        <text x={X((wingStartHigh + wingEndHigh) / 2)} y={Y(pay((wingStartHigh + wingEndHigh) / 2)) - 18}
          textAnchor="middle" fontFamily={sans} fontSize="11" fontWeight="800" fill="#1A070C">{L}x</text>
      </g>
      <text x={X(P)} y={Y(Math.max(hLowTop, hHighTop)) - 12} textAnchor="middle"
        fontFamily={sans} fontSize="10.5" fontWeight="700" fill={C.green}>
        buys at KL below pivot · sells at KH above
      </text>
      <text x={W / 2} y={HH - 34} textAnchor="middle" fontFamily={sans} fontSize="10" fill={C.faint}>
        {ekip ? "EKI Pivot TARF: obligations knock in only beyond the KI barriers" : ""}
      </text>
    </svg>
  );
}

/* ———————————————— main app ———————————————— */
export default function StructuredPricer() {
  const [page, setPage] = useState("setup");
  const [product, setProduct] = useState("TARF");
  const [tarfType, setTarfType] = useState("vanilla"); // 'vanilla' | 'lko'

  const [spot, setSpot] = useState(1.0850);
  const [buyCcy, setBuyCcy] = useState("EUR");
  const [notional, setNotional] = useState(12000000);
  const [notionalCcy, setNotionalCcy] = useState("EUR");
  const [strike, setStrike] = useState(1.0600);
  const [levBarSameAsStrike, setLevBarSameAsStrike] = useState(true);
  const [levBar, setLevBar] = useState(1.0600);
  const [leverage, setLeverage] = useState(2);
  const [civ, setCiv] = useState(0.30);
  const [countTarget, setCountTarget] = useState(5);
  const [koConv, setKoConv] = useState("full");
  const [payTiming, setPayTiming] = useState("Rolling");
  const [accKO, setAccKO] = useState(1.1200);
  const [accKoStyle, setAccKoStyle] = useState("European");
  const [vanType, setVanType] = useState("Call");
  const [vanSide, setVanSide] = useState("Buy");
  const [vanStrike, setVanStrike] = useState(1.0850);
  const [vanTerm, setVanTerm] = useState("3M");
  const [lkoBar, setLkoBar] = useState(1.0000);
  const [ekiBar, setEkiBar] = useState(1.0200);
  const [kLow, setKLow] = useState(1.0400);
  const [pivotLvl, setPivotLvl] = useState(1.0850);
  const [kHigh, setKHigh] = useState(1.1300);
  const [eLowBar, setELowBar] = useState(1.0100);
  const [eHighBar, setEHighBar] = useState(1.1600);
  const [lkoStyle, setLkoStyle] = useState("European");
  const [lkoVariant, setLkoVariant] = useState("Standard");
  const todayISO = new Date().toISOString().slice(0, 10);
  const [startDate, setStartDate] = useState(todayISO);
  const [nFix, setNFix] = useState(12);
  const [freq, setFreq] = useState("Monthly");
  const [sigma, setSigma] = useState(8.0);
  const [rUSD, setRUSD] = useState(3.65);
  const [rEUR, setREUR] = useState(2.45);
  const [nPaths, setNPaths] = useState(20000);

  const [busy, setBusy] = useState(false);
  const [spotLive, setSpotLive] = useState(undefined); // undefined = fetching, null = failed, {rate,src} = live
  const appliedLive = useRef(false);
  const [showCivHelp, setShowCivHelp] = useState(false);
  const [notionalMode, setNotionalMode] = useState("Total");
  const [depCcy, setDepCcy] = useState("EUR");
  const [dcdStrike, setDcdStrike] = useState(1.0850);
  const [dcdTerm, setDcdTerm] = useState("3M");
  const [dcdMargin, setDcdMargin] = useState(0);
  const [dcdDayCount, setDcdDayCount] = useState("ACT/365");
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const [selGreek, setSelGreek] = useState(null);

  const omega = buyCcy === "EUR" ? 1 : -1;
  const effLevBar = levBarSameAsStrike ? strike : levBar;
  const { maturity } = buildSchedule(startDate, Math.max(1, nFix || 1), freq);
  const isLKO = product === "TARF" && tarfType === "lko";
  const isEKI = product === "TARF" && tarfType === "eki";
  const isPivot = product === "TARF" && tarfType === "pivot";
  const isPivotEKI = product === "TARF" && tarfType === "ekipivot";
  const isCount = product === "TARF" && tarfType === "count";
  const isPivotFam = isPivot || isPivotEKI;
  const tarfName = tarfType === "lko" ? "Liability Knock Out TARF"
    : tarfType === "eki" ? "EKI TARF"
    : tarfType === "pivot" ? "Pivot TARF"
    : tarfType === "ekipivot" ? "EKI Pivot TARF"
    : tarfType === "count" ? "Discrete TARF" : "Vanilla TARF";

  const fetchSpot = useCallback(async (applyLevels) => {
    const sources = [
      ["https://api.frankfurter.app/latest?from=EUR&to=USD", d => d && d.rates && d.rates.USD, "ECB · frankfurter"],
      ["https://open.er-api.com/v6/latest/EUR", d => d && d.rates && d.rates.USD, "open.er-api"],
    ];
    for (const [url, pick, src] of sources) {
      try {
        const r = await fetch(url);
        if (!r.ok) continue;
        const d = await r.json();
        const v = pick(d);
        if (v && isFinite(v)) {
          const s = +(+v).toFixed(4); // EURUSD, 4 decimals
          setSpotLive({ rate: s, src });
          setSpot(s);
          if (applyLevels && !appliedLive.current) {
            appliedLive.current = true;
            const rnd = x => +x.toFixed(4);
            setStrike(rnd(s - 0.0250));
            setLevBar(rnd(s - 0.0250));
            setLkoBar(rnd(s - 0.0850));
            setEkiBar(rnd(s - 0.0650));
            setKLow(rnd(s - 0.0450));
            setPivotLvl(s);
            setKHigh(rnd(s + 0.0450));
            setELowBar(rnd(s - 0.0750));
            setEHighBar(rnd(s + 0.0750));
            setAccKO(rnd(s + 0.0350));
            setDcdStrike(s);
            setVanStrike(s);
          }
          return;
        }
      } catch (e) { /* try next source */ }
    }
    setSpotLive(null);
  }, []);
  useEffect(() => { fetchSpot(true); }, [fetchSpot]);

  const runPricing = useCallback(() => {
    setBusy(true); setErr("");
    setTimeout(() => {
      try {
        if (product === "VAN") {
          const S0v = +spot, Kv = +vanStrike, sig = +sigma / 100, rd = +rUSD / 100, rf = +rEUR / 100;
          const Nv = +notional;
          if (!(S0v > 0 && Kv > 0 && Nv > 0)) throw new Error("Check spot, strike and notional.");
          const eDate = dcdMatDate(startDate, vanTerm);
          const today = new Date(); today.setHours(0, 0, 0, 0);
          const T = Math.max((eDate - today) / 86400000 / 365, 1 / 365);
          const eurN = notionalCcy === "EUR" ? Nv : Nv / Kv;
          const sign = vanSide === "Buy" ? 1 : -1;
          const cv0 = 1 / S0v;
          const posUSD = (s, sg = sig, rdd = rd, rff = rf, TT = T) => {
            const g = gk(s, Kv, TT, rdd, rff, sg);
            return sign * eurN * (vanType === "Call" ? g.call : g.put);
          };
          const g0 = gk(S0v, Kv, T, rd, rf, sig);
          const pxPerEUR = vanType === "Call" ? g0.call : g0.put; // USD per 1 EUR notional
          const premUSD = eurN * pxPerEUR;
          const premEUR = premUSD * cv0;
          const probITM = vanType === "Call" ? normCdf(g0.d2) : normCdf(-g0.d2);
          const breakeven = vanType === "Call" ? Kv + pxPerEUR : Kv - pxPerEUR;
          const fwdV = S0v * Math.exp((rd - rf) * T);
          const hS = S0v * 0.001, hV = 0.005, hR = 0.001;
          const b0 = posUSD(S0v);
          const delta = (posUSD(S0v + hS) - posUSD(S0v - hS)) / (2 * hS);
          const gamma = (posUSD(S0v + hS) - 2 * b0 + posUSD(S0v - hS)) / (hS * hS);
          const vega = (posUSD(S0v, sig + hV) - posUSD(S0v, sig - hV)) / (2 * hV) * 0.01;
          const theta = posUSD(S0v, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - b0;
          const rhoUSD = (posUSD(S0v, sig, rd + hR) - posUSD(S0v, sig, rd - hR)) / (2 * hR) * 0.0001;
          const rhoEUR = (posUSD(S0v, sig, rd, rf + hR) - posUSD(S0v, sig, rd, rf - hR)) / (2 * hR) * 0.0001;
          const NS = 25, sLo = S0v * 0.90, sHi = S0v * 1.10;
          const prof = { delta: [], gamma: [], vega: [], theta: [], rho: [], pv: [] };
          for (let i = 0; i < NS; i++) {
            const s = +(sLo + ((sHi - sLo) * i) / (NS - 1)).toFixed(4);
            prof.pv.push({ s, v: posUSD(s) * cv0 });
            prof.delta.push({ s, v: (posUSD(s + hS) - posUSD(s - hS)) / (2 * hS) });
            prof.gamma.push({ s, v: (posUSD(s + hS) - 2 * posUSD(s) + posUSD(s - hS)) / (hS * hS) * 0.01 });
            prof.vega.push({ s, v: (posUSD(s, sig + hV) - posUSD(s, sig - hV)) / (2 * hV) * 0.01 * cv0 });
            prof.theta.push({ s, v: (posUSD(s, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - posUSD(s)) * cv0 });
            prof.rho.push({ s, vUSD: (posUSD(s, sig, rd + hR) - posUSD(s, sig, rd - hR)) / (2 * hR) * 0.0001 * cv0,
                               vEUR: (posUSD(s, sig, rd, rf + hR) - posUSD(s, sig, rd, rf - hR)) / (2 * hR) * 0.0001 * cv0 });
          }
          setRes({
            kind: "VAN", name: vanSide + " EURUSD " + vanType,
            S0: S0v, K: Kv, side: vanSide, type: vanType, eurN, T,
            expiry: fmtDate(eDate), term: vanTerm,
            premEUR, premUSD, premPct: (premEUR / eurN) * 100, pips: pxPerEUR * 10000,
            probITM: probITM * 100, breakeven, fwd: fwdV,
            delta, deltaPct: (delta / eurN) * 100, gammaFig: gamma * 0.01,
            vega, theta, rhoUSD, rhoEUR,
            prof, pivotOn: false, lkoOn: false, ekiOn: false, pivotEkiOn: false, accOn: false,
          });
          setSelGreek(null); setPage("results"); setBusy(false);
          return;
        }
        if (product === "DCD") {
          const S0d = +spot, Kd = +dcdStrike, sig = +sigma / 100, rd = +rUSD / 100, rf = +rEUR / 100;
          const Nd = +notional;
          if (!(S0d > 0 && Kd > 0 && Nd > 0)) throw new Error("Check spot, conversion strike and notional.");
          const mDate = dcdMatDate(startDate, dcdTerm);
          const today = new Date(); today.setHours(0, 0, 0, 0);
          const T = Math.max((mDate - today) / 86400000 / 365, 1 / 365); // option expiry, ACT/365
          const sDate = new Date(startDate + "T00:00:00");
          const tauAcc = accrualFrac(sDate, mDate, dcdDayCount);           // deposit accrual
          const marg = (+dcdMargin || 0) / 100;
          const cv0 = 1 / S0d;
          // client's embedded short option, valued in USD
          const posUSD = (s, sg = sig, rdd = rd, rff = rf, TT = T) => {
            const g = gk(s, Kd, TT, rdd, rff, sg);
            return depCcy === "EUR" ? -Nd * g.call : -(Nd / Kd) * g.put;
          };
          const g0 = gk(S0d, Kd, T, rd, rf, sig);
          const premUSD = depCcy === "EUR" ? Nd * g0.call : (Nd / Kd) * g0.put;
          const premDep = depCcy === "EUR" ? premUSD * cv0 : premUSD;
          const rBase = depCcy === "EUR" ? rf : rd;
          const premNet = Math.max(premDep - Nd * marg * tauAcc, 0);
          const rEnh = rBase + premNet * Math.exp(rBase * T) / (Nd * tauAcc);
          const prob = depCcy === "EUR" ? normCdf(g0.d2) : normCdf(-g0.d2);
          const cpn = rEnh * tauAcc;
          const breakeven = depCcy === "EUR" ? Kd * (1 + cpn) : Kd / (1 + cpn);
          const hS = S0d * 0.001, hV = 0.005, hR = 0.001;
          const b0 = posUSD(S0d);
          const delta = (posUSD(S0d + hS) - posUSD(S0d - hS)) / (2 * hS);
          const gamma = (posUSD(S0d + hS) - 2 * b0 + posUSD(S0d - hS)) / (hS * hS);
          const vega = (posUSD(S0d, sig + hV) - posUSD(S0d, sig - hV)) / (2 * hV) * 0.01;
          const theta = posUSD(S0d, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - b0;
          const rhoUSD = (posUSD(S0d, sig, rd + hR) - posUSD(S0d, sig, rd - hR)) / (2 * hR) * 0.0001;
          const rhoEUR = (posUSD(S0d, sig, rd, rf + hR) - posUSD(S0d, sig, rd, rf - hR)) / (2 * hR) * 0.0001;
          const eurEquivN = depCcy === "EUR" ? Nd : Nd / Kd;
          const NS = 25, sLo = S0d * 0.90, sHi = S0d * 1.10;
          const prof = { delta: [], gamma: [], vega: [], theta: [], rho: [], pv: [] };
          for (let i = 0; i < NS; i++) {
            const s = +(sLo + ((sHi - sLo) * i) / (NS - 1)).toFixed(4);
            prof.pv.push({ s, v: posUSD(s) * cv0 });
            prof.delta.push({ s, v: (posUSD(s + hS) - posUSD(s - hS)) / (2 * hS) });
            prof.gamma.push({ s, v: (posUSD(s + hS) - 2 * posUSD(s) + posUSD(s - hS)) / (hS * hS) * 0.01 });
            prof.vega.push({ s, v: (posUSD(s, sig + hV) - posUSD(s, sig - hV)) / (2 * hV) * 0.01 * cv0 });
            prof.theta.push({ s, v: (posUSD(s, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - posUSD(s)) * cv0 });
            prof.rho.push({ s, vUSD: (posUSD(s, sig, rd + hR) - posUSD(s, sig, rd - hR)) / (2 * hR) * 0.0001 * cv0,
                               vEUR: (posUSD(s, sig, rd, rf + hR) - posUSD(s, sig, rd, rf - hR)) / (2 * hR) * 0.0001 * cv0 });
          }
          setRes({
            kind: "DCD", name: "Dual Currency Deposit",
            S0: S0d, K: Kd, depCcy, N: Nd, T, cpn, maturity: fmtDate(mDate), term: dcdTerm,
            dayCount: dcdDayCount,
            rEnh: rEnh * 100, rBase: rBase * 100, pickup: (rEnh - rBase) * 100,
            premDep, premPct: (premDep / Nd) * 100, prob: prob * 100, breakeven,
            fwd: S0d * Math.exp((rd - rf) * T),
            optName: depCcy === "EUR" ? "short EUR call / USD put" : "short EUR put / USD call",
            delta, deltaPct: (delta / eurEquivN) * 100, gammaFig: gamma * 0.01,
            vega, theta, rhoUSD, rhoEUR,
            prof, pivotOn: false, lkoOn: false, ekiOn: false, pivotEkiOn: false,
          });
          setSelGreek(null); setPage("results"); setBusy(false);
          return;
        }
        const S0 = +spot, L = +leverage;
        const isACC = product === "ACCU";
        const KL = +kLow, KH = +kHigh, PL = +pivotLvl;
        const K = isPivotFam ? PL : +strike;
        const ELB = +eLowBar, EHB = +eHighBar;
        const KOL = +accKO;
        const B = (levBarSameAsStrike || isEKI || isPivotFam || isACC) ? K : +levBar;
        if (!(S0 > 0 && K > 0 && L >= 1)) throw new Error("Check spot, strike, leverage.");
        if (isPivotFam) {
          if (!(KL < PL && PL < KH)) throw new Error("Pivot TARF requires Low Strike < Pivot < High Strike.");
          if (isPivotEKI) {
            if (!(ELB < KL)) throw new Error("Low KI barrier must be below the Low Strike.");
            if (!(EHB > KH)) throw new Error("High KI barrier must be above the High Strike.");
          }
        } else {
          if (omega === 1 && B > K + 1e-12) throw new Error("Buying EUR: leverage barrier must be at or below the strike.");
          if (omega === -1 && B < K - 1e-12) throw new Error("Buying USD: leverage barrier must be at or above the strike.");
        }
        const H = +lkoBar;
        const EB = +ekiBar;
        if (isEKI) {
          if (omega === 1 && !(EB < K)) throw new Error("Buying EUR: KI barrier must be below the strike.");
          if (omega === -1 && !(EB > K)) throw new Error("Buying USD: KI barrier must be above the strike.");
        }
        if (isLKO) {
          if (omega === 1 && !(H < Math.min(K, B))) throw new Error("Buying EUR: LKO barrier must be below the strike and leverage barrier.");
          if (omega === -1 && !(H > Math.max(K, B))) throw new Error("Buying USD: LKO barrier must be above the strike and leverage barrier.");
        }
        if (isACC) {
          if (omega === 1 && !(KOL > K)) throw new Error("Buying EUR: KO barrier must be above the strike.");
          if (omega === -1 && !(KOL < K)) throw new Error("Buying USD: KO barrier must be below the strike.");
        }
        const target = (isACC || isCount) ? 1e18 : +civ * 1.0;
        if (!isACC && !isCount && !(target > 0)) throw new Error("CIV must be above 0.");
        const nGainTarget = Math.max(1, Math.round(+countTarget || 1));
        const n = Math.max(1, Math.min(60, Math.round(+nFix)));
        const { taus, maturity, dates } = buildSchedule(startDate, n, freq);
        const sig = +sigma / 100, rd = +rUSD / 100, rf = +rEUR / 100;
        const convRate = isPivotFam ? PL : K;
        const totN = notionalMode === "Per fixing" ? (+notional) * n : +notional;
        const amtPerFixCcy = totN / n;
        const amtEURperFix = notionalCcy === "EUR" ? amtPerFixCcy : amtPerFixCcy / convRate;
        const totalEUR = notionalCcy === "EUR" ? totN : totN / convRate;

        const NP = Math.max(2000, Math.min(60000, Math.round(+nPaths)));
        const z = makeNormals(NP, n, 12345);
        const u = makeUniforms(NP, n, 98765);
        const payAtMat = payTiming === "At maturity (ZC)";
        const base = { S0, K, B, L, target, sigma: sig, rd, rf, omega, amtEURperFix, koConv,
          lkoOn: isLKO, H, lkoStyle, lkoVariant, ekiOn: isEKI, E: EB,
          pivotOn: isPivotFam, kLow: KL, kHigh: KH, pivotL: PL,
          pivotEkiOn: isPivotEKI, eLow: ELB, eHigh: EHB, payAtMat,
          accOn: isACC, koLevel: KOL, accStyle: accKoStyle,
          countOn: isCount, targetCount: nGainTarget };
        const P = (over, np = NP, tt = taus) => priceEngine({ ...base, ...over }, z, u, np, tt).pv;
        const r0 = priceEngine(base, z, u, NP, taus, true);

        const hS = S0 * 0.001;
        const pvUp = P({ S0: S0 + hS }), pvDn = P({ S0: S0 - hS });
        const delta = (pvUp - pvDn) / (2 * hS);
        const gamma = (pvUp - 2 * r0.pv + pvDn) / (hS * hS);
        const hV = 0.005;
        const vega = (P({ sigma: sig + hV }) - P({ sigma: sig - hV })) / (2 * hV) * 0.01;
        const tauTheta = taus.map(t2 => Math.max(t2 - 1 / 365, 1e-6));
        const theta = P({}, NP, tauTheta) - r0.pv;
        const hR = 0.001;
        const rhoUSD = (P({ rd: rd + hR }) - P({ rd: rd - hR })) / (2 * hR) * 0.0001;
        const rhoEUR = (P({ rf: rf + hR }) - P({ rf: rf - hR })) / (2 * hR) * 0.0001;

        const npLad = Math.min(NP, 8000);
        const NS = 25, sLo = S0 * 0.90, sHi = S0 * 1.10;
        const grid = Array.from({ length: NS }, (_, i) => sLo + ((sHi - sLo) * i) / (NS - 1));
        const lad = o => grid.map(s => P({ S0: s, ...o }, npLad));
        const l0 = lad({});
        const lVu = lad({ sigma: sig + hV }), lVd = lad({ sigma: sig - hV });
        const lTh = grid.map(s => P({ S0: s }, npLad, tauTheta));
        const lRdU = lad({ rd: rd + hR }), lRdD = lad({ rd: rd - hR });
        const lRfU = lad({ rf: rf + hR }), lRfD = lad({ rf: rf - hR });
        const ds = grid[1] - grid[0];
        const cv = 1 / S0; // display conversion USD → EUR at spot
        const prof = { delta: [], gamma: [], vega: [], theta: [], rho: [], pv: [] };
        for (let i = 0; i < NS; i++) {
          const s = +grid[i].toFixed(4);
          prof.pv.push({ s, v: l0[i] * cv });
          const dl = i === 0 ? (l0[1] - l0[0]) / ds :
                     i === NS - 1 ? (l0[NS - 1] - l0[NS - 2]) / ds :
                     (l0[i + 1] - l0[i - 1]) / (2 * ds);
          prof.delta.push({ s, v: dl });
          if (i > 0 && i < NS - 1)
            prof.gamma.push({ s, v: (l0[i + 1] - 2 * l0[i] + l0[i - 1]) / (ds * ds) * 0.01 });
          prof.vega.push({ s, v: (lVu[i] - lVd[i]) / (2 * hV) * 0.01 * cv });
          prof.theta.push({ s, v: (lTh[i] - l0[i]) * cv });
          prof.rho.push({ s, vUSD: (lRdU[i] - lRdD[i]) / (2 * hR) * 0.0001 * cv,
                             vEUR: (lRfU[i] - lRfD[i]) / (2 * hR) * 0.0001 * cv });
        }
        const fwd = S0 * Math.exp((rd - rf) * taus[taus.length - 1]);
        let cum = 0;
        const tauN = taus[taus.length - 1];
        const fixings = taus.map((tau, i) => {
          const ecf = r0.stats.cf[i] * cv;
          cum += ecf;
          return {
            i: i + 1, date: fmtDate(dates[i]), tau,
            fwd: S0 * Math.exp((rd - rf) * tau),
            df: Math.exp(-rd * (payAtMat ? tauN : tau)),
            amt: amtEURperFix,
            alive: r0.stats.alive[i] * 100,
            ecf, cum,
          };
        });
        setRes({
          fixings,
          name: isACC ? "Accumulator · " + accKoStyle + " KO"
            : isCount ? "Discrete TARF"
            : isLKO ? "Liability Knock Out TARF" : isEKI ? "EKI TARF"
            : isPivotEKI ? "EKI Pivot TARF" : isPivot ? "Pivot TARF" : "Vanilla TARF",
          pvUSD: r0.pv, se: r0.se, pvEUR: r0.pv / S0,
          pvPct: (r0.pv / (totalEUR * S0)) * 100,
          delta, deltaPct: (delta / totalEUR) * 100,
          gammaFig: gamma * 0.01, vega, theta, rhoUSD, rhoEUR,
          koProb: r0.koProb * 100, lkoProb: r0.lkoProb * 100,
          ekiProb: r0.ekiProb * 100, expLife: r0.expLifeFix,
          prof, fwd, n,
          amtEURperFix, targetFig: isACC ? Math.abs(KOL - K) * 100 : target * 100, S0, K, B, L, omega,
          accOn: isACC, koLevel: KOL, accStyle: accKoStyle,
          countOn: isCount, targetCount: nGainTarget,
          lkoOn: isLKO, H, lkoStyle, lkoVariant, ekiOn: isEKI, E: EB,
          pivotOn: isPivotFam, kLow: KL, kHigh: KH, pivotL: PL,
          pivotEkiOn: isPivotEKI, eLow: ELB, eHigh: EHB, payAtMat,
          terms: {
            pair: "EUR/USD", buyCcy, notional: totN, notionalCcy,
            strike: K, leverage: L, civ: +civ, freq, nFix: n,
            sigma: +sigma, rUSD: +rUSD, rEUR: +rEUR, koConv,
            maturity: fmtDate(maturity),
          },
        });
        setSelGreek(null);
        setPage("results");
      } catch (e) { setErr(e.message || String(e)); }
      setBusy(false);
    }, 30);
  }, [spot, strike, levBar, levBarSameAsStrike, leverage, civ, koConv, startDate,
      nFix, freq, sigma, rUSD, rEUR, nPaths, notional, notionalCcy, omega, buyCcy,
      isLKO, lkoBar, lkoStyle, lkoVariant, isEKI, ekiBar,
      isPivotFam, isPivotEKI, kLow, kHigh, pivotLvl, eLowBar, eHighBar, notionalMode,
      product, depCcy, dcdStrike, dcdTerm, dcdMargin, dcdDayCount, payTiming, accKO, accKoStyle,
      vanType, vanSide, vanStrike, vanTerm, isCount, countTarget]);

  // ---------- design tokens ----------
  const C = {
    bg: "radial-gradient(1000px 540px at 75% -10%, rgba(96,118,168,0.10), transparent 62%), #0A0C11",
    card: "linear-gradient(180deg, #13161F 0%, #0F1218 100%)", card2: "#171B26", line: "#242A38",
    inp: "#161A24", inpLine: "#2A3140",
    text: "#E8EAEE", mute: "#98A1B3", faint: "#646E82",
    blue: "#4A7DF0", amber: "#D9A441", green: "#2FBF8F", red: "#E5484D", violet: "#8B7EDB",
  };
  const mono = "'SF Mono', ui-monospace, 'Cascadia Code', Consolas, 'Roboto Mono', Menlo, monospace";
  const sans = "'Inter', 'Inter UI', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";
  const label = { fontSize: 12.5, color: C.mute, fontFamily: sans, marginBottom: 7, display: "block" };
  const colHead = { fontSize: 10.5, letterSpacing: "0.14em", textTransform: "uppercase",
    color: C.faint, fontFamily: sans };
  const input = { width: "100%", boxSizing: "border-box", background: C.inp,
    border: `1px solid ${C.inpLine}`, borderRadius: 10, color: C.text, padding: "11px 13px",
    fontFamily: mono, fontSize: 14, outline: "none" };
  const card = { background: C.card, border: `1px solid ${C.line}`, borderRadius: 16, padding: 22,
    boxShadow: "inset 0 1px 0 rgba(255,255,255,0.03), 0 12px 36px rgba(0,0,0,0.38)" };
  const cardTitle = { fontFamily: sans, fontSize: 16.5, fontWeight: 700, color: C.text,
    marginBottom: 16, letterSpacing: "-0.01em" };

  const Num = ({ v, set, step = "any", min }) => (
    <input type="number" className="sp-input" value={v} step={step} min={min} style={input}
      onChange={e => set(e.target.value === "" ? "" : +e.target.value)} />
  );
  const Sel = ({ v, set, opts }) => (
    <div style={{ position: "relative" }}>
      <select value={v} className="sp-input"
        style={{ ...input, appearance: "none", WebkitAppearance: "none", MozAppearance: "none",
          paddingRight: 30, cursor: "pointer" }}
        onChange={e => set(e.target.value)}>
        {opts.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
      <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-56%)",
        pointerEvents: "none", color: C.mute, fontSize: 11 }}>▼</span>
    </div>
  );
  const NotionalInput = () => (
    <input type="text" inputMode="numeric" className="sp-input"
      value={notional === "" ? "" : fmt(notional, 0)}
      style={{ ...input, fontSize: 15, letterSpacing: "0.02em" }}
      onChange={e => {
        const raw2 = e.target.value.replace(/[^0-9]/g, "");
        setNotional(raw2 === "" ? "" : +raw2);
      }} />
  );
  const Field = ({ name, children, hint }) => (
    <div>
      <span style={label}>{name}</span>
      {children}
      {hint && <div style={{ fontSize: 11, color: C.faint, marginTop: 5, fontFamily: mono }}>{hint}</div>}
    </div>
  );

  const products = [
    { id: "TARF", name: "TARF", desc: "Target redemption forward family: strip of leveraged forwards knocked out on accumulated gains", ready: true },
    { id: "ACCU", name: "Accumulator", desc: "Accumulative forward with a knock out barrier: European or American observation, rolling or ZC settlement", ready: true },
    { id: "DCD", name: "Dual Currency Deposit", desc: "Yield enhanced deposit: the client implicitly sells an FX option and earns its premium as extra coupon", ready: true },
    { id: "VAN", name: "Vanilla Option", desc: "European EURUSD call or put, buy or sell, closed form Garman Kohlhagen", ready: true },
  ];
  const tarfTypes = [
    { id: "vanilla", name: "Vanilla TARF", desc: "Leverage applies on every losing fixing until maturity or target knock out" },
    { id: "lko", name: "Liability Knock Out TARF", desc: "A barrier on the loss side knocks the leverage down to 0x if triggered" },
    { id: "eki", name: "EKI TARF", desc: "European KI barrier: obligation only when the fixing lands beyond the barrier; in between, the client trades at market" },
    { id: "pivot", name: "Pivot TARF", desc: "Two sided: sells EUR at the high strike above the pivot, buys EUR at the low strike below it, leveraged beyond the strikes" },
    { id: "ekipivot", name: "EKI Pivot TARF", desc: "Pivot TARF with a pair of European KI barriers: leveraged obligations only knock in beyond them, participation in between" },
    { id: "count", name: "Discrete TARF", desc: "Knocks out after a fixed number of gaining fixings, regardless of their size, instead of an accumulated CIV amount" },
  ];

  const dirText = omega === 1 ? "Client buys EUR / sells USD at strike" : "Client buys USD / sells EUR at strike";
  const lossSide = omega === 1 ? "below" : "above";

  /* ————————————————— PAGE 1 : SETUP ————————————————— */
  if (page === "setup") {
    return (
      <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
        <style>{GLOBAL_CSS}</style>
        <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28,
            paddingBottom: 18, borderBottom: `1px solid ${C.line}` }}>
            <div style={{ width: 30, height: 30, borderRadius: 9, transform: "rotate(45deg)",
              background: "linear-gradient(135deg, #4A7DF0, #6E8EF7)",
              boxShadow: "0 4px 16px rgba(74,125,240,0.45)", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center" }}>
              <div style={{ width: 10, height: 10, borderRadius: 3, background: "rgba(255,255,255,0.9)" }} />
            </div>
            <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>
              FX Structured Products{" "}
              <span style={{ backgroundImage: "linear-gradient(90deg, #4A7DF0, #8B7EDB)",
                WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Pricer</span>
            </div>
          </div>

          {/* product family */}
          <div style={{ ...card, marginBottom: 20 }}>
            <div style={cardTitle}>Product</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
              {products.map(p => {
                const active = product === p.id;
                return (
                  <div key={p.id} onClick={() => p.ready && setProduct(p.id)}
                    className={p.ready ? "sp-click" : undefined}
                    style={{ background: active ? "rgba(74,125,240,0.10)" : C.card2,
                      border: `1.5px solid ${active ? C.blue : C.line}`, borderRadius: 12,
                      padding: "16px 16px 14px", cursor: p.ready ? "pointer" : "not-allowed",
                      opacity: p.ready ? 1 : 0.45, position: "relative" }}>
                    <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6,
                      color: active ? C.blue : C.text }}>{p.name}</div>
                    <div style={{ fontSize: 11.5, color: C.mute, lineHeight: 1.45 }}>{p.desc}</div>
                    {!p.ready && (
                      <div style={{ position: "absolute", top: 10, right: 10, fontSize: 9.5,
                        letterSpacing: "0.1em", textTransform: "uppercase", color: C.faint,
                        border: `1px solid ${C.line}`, borderRadius: 20, padding: "2px 8px" }}>soon</div>
                    )}
                    {active && (
                      <div style={{ position: "absolute", top: 10, right: 10, width: 18, height: 18,
                        borderRadius: 9, background: C.blue, color: "#fff", fontSize: 11,
                        display: "flex", alignItems: "center", justifyContent: "center" }}>✓</div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* mandatory sub selection for TARF */}
            {product === "TARF" && (
              <div style={{ marginTop: 18, paddingTop: 18, borderTop: `1px solid ${C.line}` }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: C.mute, marginBottom: 12 }}>
                  Which TARF? <span style={{ fontWeight: 400, color: C.faint }}>(required)</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
                  {tarfTypes.map(tt => {
                    const active = tarfType === tt.id;
                    return (
                      <div key={tt.id} onClick={() => setTarfType(tt.id)} className="sp-click"
                        style={{ display: "flex", gap: 12, alignItems: "flex-start",
                          background: active ? "rgba(74,125,240,0.10)" : C.card2,
                          border: `1.5px solid ${active ? C.blue : C.line}`, borderRadius: 12,
                          padding: "14px 16px", cursor: "pointer" }}>
                        <div style={{ width: 18, height: 18, borderRadius: 9, marginTop: 1, flexShrink: 0,
                          border: `2px solid ${active ? C.blue : C.faint}`,
                          background: active ? C.blue : "transparent",
                          display: "flex", alignItems: "center", justifyContent: "center" }}>
                          {active && <div style={{ width: 6, height: 6, borderRadius: 3, background: "#fff" }} />}
                        </div>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: 14, color: active ? C.blue : C.text }}>{tt.name}</div>
                          <div style={{ fontSize: 11.5, color: C.mute, marginTop: 4, lineHeight: 1.45 }}>{tt.desc}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.6fr", gap: 20, marginBottom: 20 }}>
            <div style={card}>
              <div style={cardTitle}>Market</div>
              <div style={{ display: "grid", gap: 16 }}>
                <Field name="Currency pair"><Sel v={"EUR/USD"} set={() => {}} opts={["EUR/USD"]} /></Field>
                <div>
                  <span style={{ ...label, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span>Spot</span>
                    <span onClick={() => fetchSpot(false)} title="Refresh live rate"
                      style={{ cursor: "pointer", color: C.blue, fontSize: 12, lineHeight: 1,
                        padding: "1px 6px", border: `1px solid ${C.line}`, borderRadius: 6 }}>↻</span>
                  </span>
                  <Num v={spot} set={setSpot} step="0.0001" />
                  <div style={{ fontSize: 11, marginTop: 5, fontFamily: mono,
                    color: spotLive && spotLive.rate ? C.green : C.faint }}>
                    {spotLive === undefined ? "fetching live EURUSD…"
                      : spotLive && spotLive.rate ? `live ${spotLive.rate.toFixed(4)} · ${spotLive.src}`
                      : "live feed unavailable · manual"}
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Field name="USD rate (%)"><Num v={rUSD} set={setRUSD} step="0.05" /></Field>
                  <Field name="EUR rate (%)"><Num v={rEUR} set={setREUR} step="0.05" /></Field>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Field name="Vol σ (%)"><Num v={sigma} set={setSigma} step="0.25" /></Field>
                  <Field name="MC paths"><Num v={nPaths} set={setNPaths} step="1000" min="2000" /></Field>
                </div>
              </div>
            </div>

            <div style={card}>
              <div style={cardTitle}>Trade · {product === "DCD" ? "Dual Currency Deposit"
                : product === "ACCU" ? "Accumulator"
                : product === "VAN" ? "Vanilla Option" : tarfName}</div>
              {product === "VAN" ? (<>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                  <Field name="Type" hint={vanType === "Call" ? "EUR call / USD put" : "EUR put / USD call"}>
                    <Sel v={vanType} set={setVanType} opts={["Call", "Put"]} /></Field>
                  <Field name="Position">
                    <Sel v={vanSide} set={setVanSide} opts={["Buy", "Sell"]} /></Field>
                  <Field name="Strike">
                    <Num v={vanStrike} set={setVanStrike} step="0.0001" /></Field>
                  <div style={{ gridColumn: "1 / span 2" }}>
                    <Field name="Notional"
                      hint={notionalCcy === "USD" ? `= ${fmtBig((+notional || 0) / (+vanStrike || 1))} EUR at the strike` : "EUR notional of the option"}>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 92px", gap: 10 }}>
                        <NotionalInput />
                        <Sel v={notionalCcy} set={setNotionalCcy} opts={["EUR", "USD"]} />
                      </div>
                    </Field>
                  </div>
                </div>
                <div style={{ fontSize: 12, color: C.faint, marginTop: 14, lineHeight: 1.55 }}>
                  European exercise, cash settled at expiry, priced closed form under Garman Kohlhagen with the
                  model rates and flat vol from the Market card. The client {vanSide === "Buy" ? "pays" : "receives"} the
                  premium today.
                </div>
              </>) : product === "ACCU" ? (<>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                  <Field name="Client buys"><Sel v={buyCcy} set={setBuyCcy} opts={["EUR", "USD"]} /></Field>
                  <Field name="Strike" hint={omega === 1 ? "below market, client accumulates here" : "above market"}>
                    <Num v={strike} set={setStrike} step="0.0001" /></Field>
                  <Field name="Leverage"><Num v={leverage} set={setLeverage} step="0.5" min="1" /></Field>
                  <div style={{ gridColumn: "1 / span 2" }}>
                    <Field name="Notional"
                      hint={notionalMode === "Total"
                        ? `= ${fmtBig((+notional || 0) / Math.max(1, Math.round(+nFix || 1)))} ${notionalCcy} per fixing`
                        : `= ${fmtBig((+notional || 0) * Math.max(1, Math.round(+nFix || 1)))} ${notionalCcy} total`}>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 148px 96px", gap: 10 }}>
                        <NotionalInput />
                        <Sel v={notionalMode} set={setNotionalMode} opts={["Total", "Per fixing"]} />
                        <Sel v={notionalCcy} set={setNotionalCcy} opts={["EUR", "USD"]} />
                      </div>
                    </Field>
                  </div>
                  <Field name="KO barrier" hint={omega === 1 ? "above strike, cancels the trade" : "below strike, cancels the trade"}>
                    <Num v={accKO} set={setAccKO} step="0.0001" /></Field>
                  <Field name="KO observation">
                    <Sel v={accKoStyle} set={setAccKoStyle} opts={["European", "American"]} /></Field>
                  <Field name="Payment timing"
                    hint={payTiming === "Rolling" ? "each fixing settles on its date" : "all flows accumulate and pay at maturity"}>
                    <Sel v={payTiming} set={setPayTiming} opts={["Rolling", "At maturity (ZC)"]} /></Field>
                </div>
                <div style={{ fontSize: 12, color: C.faint, marginTop: 14, lineHeight: 1.55 }}>
                  {omega === 1 ? "Client buys EUR / sells USD at the strike" : "Client buys USD / sells EUR at the strike"}{" "}
                  on every fixing: 100% of the per fixing notional on favourable fixings, ×{leverage || "…"} on
                  unfavourable ones. If the {accKoStyle === "European" ? "fixing" : "spot path"} crosses the KO
                  barrier, that fixing and all remaining fixings are cancelled.
                </div>
              </>) : product === "DCD" ? (<>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                  <Field name="Deposit currency"><Sel v={depCcy} set={setDepCcy} opts={["EUR", "USD"]} /></Field>
                  <Field name="Conversion strike"
                    hint={depCcy === "EUR" ? "converted into USD if fixing above" : "converted into EUR if fixing below"}>
                    <Num v={dcdStrike} set={setDcdStrike} step="0.0001" /></Field>
                  <Field name="Bank margin (% p.a.)" hint="deducted from the coupon">
                    <Num v={dcdMargin} set={setDcdMargin} step="0.05" min="0" /></Field>
                  <div style={{ gridColumn: "1 / span 2" }}>
                    <Field name={`Deposit notional (${depCcy})`} hint="placed for the full term">
                      <NotionalInput />
                    </Field>
                  </div>
                </div>
                <div style={{ fontSize: 12, color: C.faint, marginTop: 14, lineHeight: 1.55 }}>
                  The client places a {depCcy} deposit and implicitly sells the bank a{" "}
                  {depCcy === "EUR" ? "EUR call / USD put" : "EUR put / USD call"} struck at the conversion
                  strike; the premium is returned as an enhanced coupon. At maturity the bank repays the weaker
                  currency: {depCcy} plus coupon if there is no conversion, otherwise the alternative currency
                  converted at the strike.
                </div>
              </>) : (<>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                {!isPivotFam && (<>
                  <Field name="Client buys"><Sel v={buyCcy} set={setBuyCcy} opts={["EUR", "USD"]} /></Field>
                  <Field name="Strike"><Num v={strike} set={setStrike} step="0.0001" /></Field>
                </>)}
                {isPivotFam && (<>
                  <Field name="Low Strike (Call)" hint="client buys EUR here below pivot">
                    <Num v={kLow} set={setKLow} step="0.0001" /></Field>
                  <Field name="Pivot level">
                    <Num v={pivotLvl} set={setPivotLvl} step="0.0001" /></Field>
                  <Field name="High Strike (Put)" hint="client sells EUR here above pivot">
                    <Num v={kHigh} set={setKHigh} step="0.0001" /></Field>
                </>)}
                <Field name="Leverage"><Num v={leverage} set={setLeverage} step="0.5" min="1" /></Field>
                <div style={{ gridColumn: "1 / span 2" }}>
                  <Field name="Notional"
                    hint={notionalMode === "Total"
                      ? `= ${fmtBig((+notional || 0) / Math.max(1, Math.round(+nFix || 1)))} ${notionalCcy} per fixing`
                      : `= ${fmtBig((+notional || 0) * Math.max(1, Math.round(+nFix || 1)))} ${notionalCcy} total`}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 148px 96px", gap: 10 }}>
                      <NotionalInput />
                      <Sel v={notionalMode} set={setNotionalMode} opts={["Total", "Per fixing"]} />
                      <Sel v={notionalCcy} set={setNotionalCcy} opts={["EUR", "USD"]} />
                    </div>
                  </Field>
                </div>
                {isCount ? (
                  <Field name="Target # of gains" hint="trade stops after this many ITM fixings">
                    <Num v={countTarget} set={setCountTarget} step="1" min="1" />
                  </Field>
                ) : (
                <div>
                  <span style={label}>CIV target{" "}
                    <span onClick={() => setShowCivHelp(v => !v)}
                      style={{ display: "inline-flex", alignItems: "center", justifyContent: "center",
                        width: 15, height: 15, borderRadius: 8, marginLeft: 4, cursor: "pointer",
                        border: `1px solid ${showCivHelp ? C.blue : C.faint}`,
                        color: showCivHelp ? C.blue : C.faint, fontSize: 10, fontWeight: 700,
                        verticalAlign: "middle" }}>?</span>
                  </span>
                  <Num v={civ} set={setCiv} step="0.05" min="0" />
                  <div style={{ fontSize: 11, color: C.faint, marginTop: 5, fontFamily: mono }}>
                    = {fmt((+civ || 0) * 100, 0)} figures of accumulated gain
                  </div>
                  {showCivHelp && (
                    <div style={{ marginTop: 8, padding: "10px 12px", background: "rgba(74,125,240,0.07)",
                      border: `1px solid rgba(74,125,240,0.35)`, borderRadius: 10,
                      fontSize: 11.5, color: C.mute, lineHeight: 1.55 }}>
                      1 CIV = 100 figures. 1 figure = 0.0100 in EURUSD terms: a move from 1.1500 to 1.1600
                      is 1 figure. The trade knocks out once the sum of in the money fixings reaches
                      CIV × 100 figures of accumulated intrinsic value.
                    </div>
                  )}
                </div>
                )}
                <Field name="Settlement at knock out">
                  <Sel v={koConv} set={setKoConv} opts={["full", "capped", "none"]} /></Field>
                <Field name="Payment timing"
                  hint={payTiming === "Rolling" ? "each fixing settles on its date" : "all flows accumulate and pay at maturity"}>
                  <Sel v={payTiming} set={setPayTiming} opts={["Rolling", "At maturity (ZC)"]} /></Field>
                {!isEKI && !isPivotFam && (
                  <div>
                    <span style={label}>Leverage barrier</span>
                    <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5,
                      color: C.mute, cursor: "pointer", height: 41 }}>
                      <input type="checkbox" checked={levBarSameAsStrike}
                        onChange={e => setLevBarSameAsStrike(e.target.checked)} />
                      same as strike
                    </label>
                  </div>
                )}
                {!levBarSameAsStrike && !isEKI && !isPivotFam && (
                  <Field name="Leverage barrier level"><Num v={levBar} set={setLevBar} step="0.0001" /></Field>
                )}
              </div>

              {isPivotEKI && (
                <div style={{ marginTop: 18, padding: "16px 16px 14px", background: "rgba(200,154,75,0.06)",
                  border: `1px solid rgba(200,154,75,0.35)`, borderRadius: 12 }}>
                  <div style={{ fontWeight: 700, fontSize: 13.5, color: "#C89A4B", marginBottom: 12 }}>
                    European Knock In pair
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.4fr", gap: 16 }}>
                    <Field name="Low KI barrier" hint="below Low Strike">
                      <Num v={eLowBar} set={setELowBar} step="0.0001" /></Field>
                    <Field name="High KI barrier" hint="above High Strike">
                      <Num v={eHighBar} set={setEHighBar} step="0.0001" /></Field>
                    <div style={{ fontSize: 11.5, color: C.faint, lineHeight: 1.55, alignSelf: "end", paddingBottom: 6 }}>
                      Fixing between a strike and its KI barrier: no obligation, the client participates at market.
                      Fixing beyond either barrier: the obligation knocks in at that strike on ×{leverage || "…"}
                      of the notional. Observed per fixing, European style.
                    </div>
                  </div>
                </div>
              )}

              {isEKI && (
                <div style={{ marginTop: 18, padding: "16px 16px 14px", background: "rgba(200,154,75,0.06)",
                  border: `1px solid rgba(200,154,75,0.35)`, borderRadius: 12 }}>
                  <div style={{ fontWeight: 700, fontSize: 13.5, color: "#C89A4B", marginBottom: 12 }}>
                    European Knock In
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 16 }}>
                    <Field name="KI barrier" hint={omega === 1 ? "below strike" : "above strike"}>
                      <Num v={ekiBar} set={setEkiBar} step="0.0001" /></Field>
                    <div style={{ fontSize: 11.5, color: C.faint, lineHeight: 1.55, alignSelf: "end", paddingBottom: 6 }}>
                      Fixing between strike and KI barrier: no obligation on either party, the client
                      participates at market. Fixing beyond the barrier: the client trades at the strike
                      on ×{leverage || "…"} of the notional. Observed per fixing, European style.
                    </div>
                  </div>
                </div>
              )}

              {isLKO && (
                <div style={{ marginTop: 18, padding: "16px 16px 14px", background: "rgba(139,126,219,0.06)",
                  border: `1px solid rgba(139,126,219,0.35)`, borderRadius: 12 }}>
                  <div style={{ fontWeight: 700, fontSize: 13.5, color: C.violet, marginBottom: 12 }}>
                    Liability Knock Out
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                    <Field name="LKO barrier" hint={omega === 1 ? "below strike" : "above strike"}>
                      <Num v={lkoBar} set={setLkoBar} step="0.0001" /></Field>
                    <Field name="Observation">
                      <Sel v={lkoStyle} set={setLkoStyle} opts={["European", "American"]} /></Field>
                    <Field name="Variant">
                      <Sel v={lkoVariant} set={setLkoVariant} opts={["Standard", "Accelerated"]} /></Field>
                  </div>
                  <div style={{ fontSize: 11.5, color: C.faint, marginTop: 10, lineHeight: 1.55 }}>
                    {lkoVariant === "Standard"
                      ? "If triggered, loss side leverage drops to 0x for the remaining fixings: the client keeps the gain leg but faces no further obligations."
                      : "If triggered, the strategy terminates immediately and the outstanding target balance is paid upfront to the client."}
                    {" "}{lkoStyle === "European"
                      ? "Observed on fixing dates only."
                      : "Monitored continuously (Brownian bridge between fixings)."}
                  </div>
                </div>
              )}

              <div style={{ fontSize: 12, color: C.faint, marginTop: 14, lineHeight: 1.55 }}>
                {isPivotFam
                  ? <>Above the pivot the client sells EUR at the High Strike; below it the client buys EUR at
                    the Low Strike, on 100% of the per fixing notional.{" "}
                    {isPivotEKI
                      ? <>The leveraged obligation only knocks in beyond the KI barriers; between a strike and
                        its barrier the client participates at market.</>
                      : <>Beyond either strike the obligation runs on ×{leverage || "…"} of the notional.</>}{" "}
                    Gains accrue toward the target; losses do not count toward it. USD notionals convert at the pivot.</>
                  : <>{dirText}. Leverage ×{leverage || "…"} applies {lossSide}{" "}
                    <span style={{ fontFamily: mono }}>{fmtRate(+effLevBar || 0)}</span>.
                    Gains accrue toward the target; losses do not count toward it.</>}
              </div>
              </>)}
            </div>
          </div>

          <div style={{ ...card, marginBottom: 26 }}>
            <div style={cardTitle}>Schedule</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
              <Field name="Start date">
                <input type="date" className="sp-input" value={startDate} style={input}
                  onChange={e => setStartDate(e.target.value)} />
              </Field>
              {product === "VAN" ? (<>
                <Field name="Expiry term">
                  <Sel v={vanTerm} set={setVanTerm} opts={["1W", "2W", "1M", "2M", "3M", "6M", "12M"]} /></Field>
                <Field name="Expiry date (auto)">
                  <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                    color: C.blue, fontWeight: 600 }}>
                    {fmtDate(dcdMatDate(startDate, vanTerm))}
                  </div>
                </Field>
                <Field name="Day count">
                  <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                    color: C.mute }}>ACT/365</div>
                </Field>
              </>) : product === "DCD" ? (<>
                <Field name="Deposit term">
                  <Sel v={dcdTerm} set={setDcdTerm} opts={["2W", "1M", "2M", "3M", "6M", "12M"]} /></Field>
                <Field name="Maturity (auto)">
                  <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                    color: C.blue, fontWeight: 600 }}>
                    {fmtDate(dcdMatDate(startDate, dcdTerm))}
                  </div>
                </Field>
                <Field name="Day count" hint="deposit interest accrual">
                  <Sel v={dcdDayCount} set={setDcdDayCount} opts={["ACT/365", "ACT/360", "30/360", "ACT/ACT"]} />
                </Field>
              </>) : (<>
                <Field name="Frequency"><Sel v={freq} set={setFreq} opts={["Weekly", "Biweekly", "Monthly"]} /></Field>
                <Field name="Number of fixings"><Num v={nFix} set={setNFix} step="1" min="1" /></Field>
                <Field name="Maturity (auto)">
                  <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                    color: C.blue, fontWeight: 600 }}>
                    {fmtDate(maturity)}
                  </div>
                </Field>
              </>)}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
            <button onClick={runPricing} disabled={busy} className="sp-btn"
              style={{ padding: "16px 72px",
                background: busy ? C.card2 : "linear-gradient(135deg, #4A7DF0 0%, #6E8EF7 140%)",
                color: busy ? C.mute : "#fff", letterSpacing: "0.01em",
                fontFamily: sans, fontSize: 17, fontWeight: 700, border: "none", borderRadius: 14,
                cursor: busy ? "wait" : "pointer",
                boxShadow: busy ? "none" : "0 10px 34px rgba(74,125,240,0.40)" }}>
              {busy ? "Pricing…" : "Price Product →"}
            </button>
            {err && <div style={{ color: C.red, fontSize: 13 }}>{err}</div>}
          </div>
        </div>
      </div>
    );
  }

  /* ————————————————— PAGE 2 : RESULTS ————————————————— */
  const greekCards = res ? [
    { id: "delta", name: "Delta", unit: "EUR equivalent", color: C.blue, val: fmtBigSigned(res.delta) },
    { id: "delta", name: "Delta %", unit: "of EUR notional", color: C.text, val: fmtSigned(res.deltaPct, 1) + "%" },
    { id: "gamma", name: "Gamma", unit: "Δdelta per 1 figure", color: C.text, val: fmtBigSigned(res.gammaFig) },
    { id: "vega", name: "Vega", unit: "EUR per 1 vol pt", color: C.blue, val: fmtBigSigned(res.vega / res.S0) },
    { id: "theta", name: "Theta", unit: "EUR per day", color: C.text, val: fmtBigSigned(res.theta / res.S0) },
    { id: "rho", name: "Rho USD / EUR", unit: "EUR per 1 bp", color: C.text,
      val: `${fmtBigSigned(res.rhoUSD / res.S0)} / ${fmtBigSigned(res.rhoEUR / res.S0)}` },
  ] : [];
  const greekMeta = {
    delta: { title: "Delta vs spot", unit: "EUR equivalent", keys: [["v", C.blue, "Delta"]] },
    gamma: { title: "Gamma vs spot", unit: "Δdelta per 1 figure", keys: [["v", C.amber, "Gamma"]] },
    vega:  { title: "Vega vs spot", unit: "EUR per 1 vol pt", keys: [["v", C.blue, "Vega"]] },
    theta: { title: "Theta vs spot", unit: "EUR per day", keys: [["v", C.amber, "Theta"]] },
    rho:   { title: "Rho vs spot", unit: "EUR per 1 bp",
             keys: [["vUSD", C.blue, "Rho USD"], ["vEUR", C.red, "Rho EUR"]] },
  };
  const tickFmt = v => fmtBig(v);

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
      <style>{GLOBAL_CSS}</style>
      <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 18,
          paddingBottom: 16, borderBottom: `1px solid ${C.line}` }}>
          <button onClick={() => setPage("setup")} className="sp-btn"
            style={{ background: C.card2, border: `1px solid ${C.line}`, color: C.text,
              borderRadius: 10, padding: "9px 16px", fontSize: 13.5, cursor: "pointer", fontFamily: sans }}>
            ← Edit parameters
          </button>
          <div style={{ fontSize: 20, fontWeight: 800 }}>
            {res ? res.name : (product === "DCD" ? "Dual Currency Deposit"
              : product === "ACCU" ? "Accumulator"
              : product === "VAN" ? "Vanilla Option" : tarfName)}{" "}
            <span style={{ backgroundImage: "linear-gradient(90deg, #4A7DF0, #8B7EDB)",
              WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Pricing</span>
          </div>
          <button onClick={runPricing} disabled={busy} className="sp-btn"
            style={{ marginLeft: "auto",
              background: "linear-gradient(135deg, #4A7DF0 0%, #6E8EF7 140%)", border: "none", color: "#fff",
              borderRadius: 10, padding: "9px 22px", fontSize: 13.5, fontWeight: 700, cursor: "pointer",
              boxShadow: "0 6px 20px rgba(74,125,240,0.30)" }}>
            {busy ? "Pricing…" : "Reprice"}
          </button>
        </div>

        {res && res.kind === "VAN" && (
          <div style={{ ...card, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, marginBottom: 18, padding: 0 }}>
            <div style={{ padding: "20px 22px" }}>
              <div style={colHead}>Premium</div>
              <div style={{ fontFamily: mono, fontSize: 30, marginTop: 6, fontVariantNumeric: "tabular-nums",
                color: res.side === "Buy" ? C.red : C.green }}>
                {res.side === "Buy" ? `(${bigRaw(res.premEUR)})` : "+" + bigRaw(res.premEUR)}{" "}
                <span style={{ fontSize: 14, color: C.mute }}>EUR</span>
              </div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>
                client {res.side === "Buy" ? "pays" : "receives"} today</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Premium % / pips</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8 }}>{fmt(res.premPct, 3)}%</div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4, fontFamily: mono }}>
                of EUR notional · {fmt(res.pips, 1)} USD pips per EUR</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Premium in USD</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8 }}>{fmtBig(res.premUSD)}</div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>settlement currency</div>
            </div>
          </div>
        )}

        {res && res.kind === "VAN" && (
          <div style={{ ...card, marginBottom: 18 }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 14 }}>Option economics</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
              {[
                ["Contract", (res.type === "Call" ? "EUR call / USD put" : "EUR put / USD call") + " · European"],
                ["Expiry", res.term + " · " + res.expiry],
                ["Forward", fmtRate(res.fwd)],
                ["Moneyness K/F", fmt((res.K / res.fwd) * 100, 2) + "%"],
                ["P(ITM at expiry)", fmt(res.probITM, 1) + "%"],
                ["Breakeven at expiry", fmtRate(res.breakeven)],
              ].map(([k, v]) => (
                <div key={k} style={{ background: C.card2, border: `1px solid ${C.line}`,
                  borderRadius: 10, padding: "11px 14px" }}>
                  <div style={colHead}>{k}</div>
                  <div style={{ fontFamily: mono, fontSize: 14, marginTop: 6,
                    color: k === "P(ITM at expiry)" ? C.amber : C.text }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {res && res.kind === "DCD" && (
          <div style={{ ...card, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, marginBottom: 18, padding: 0 }}>
            <div style={{ padding: "20px 22px" }}>
              <div style={colHead}>Enhanced yield</div>
              <div style={{ fontFamily: mono, fontSize: 30, marginTop: 6, color: C.green,
                fontVariantNumeric: "tabular-nums" }}>
                {fmt(res.rEnh, 2)}% <span style={{ fontSize: 14, color: C.mute }}>p.a.</span>
              </div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>{res.depCcy} deposit · {res.dayCount}</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Base deposit rate</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8 }}>{fmt(res.rBase, 2)}%</div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>market {res.depCcy} rate</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Yield pickup</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8, color: C.green }}>
                +{fmt(res.pickup, 2)}%</div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4, fontFamily: mono }}>
                option premium {fmtBig(res.premDep)} {res.depCcy} ({fmt(res.premPct, 2)}% of notional)</div>
            </div>
          </div>
        )}

        {res && res.kind === "DCD" && (
          <div style={{ ...card, marginBottom: 18 }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 14 }}>Deposit economics</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
              {[
                ["Option sold", res.optName],
                ["Term", res.term + " · matures " + res.maturity],
                ["Forward to maturity", fmtRate(res.fwd)],
                ["Conversion probability", fmt(res.prob, 1) + "%"],
                ["Breakeven spot", fmtRate(res.breakeven)],
                ["Coupon at maturity", fmtBig(res.N * res.cpn) + " " + res.depCcy],
              ].map(([k, v]) => (
                <div key={k} style={{ background: C.card2, border: `1px solid ${C.line}`,
                  borderRadius: 10, padding: "11px 14px" }}>
                  <div style={colHead}>{k}</div>
                  <div style={{ fontFamily: mono, fontSize: 14, marginTop: 6,
                    color: k === "Conversion probability" ? C.amber : C.text }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {(!res || (res.kind !== "DCD" && res.kind !== "VAN")) && (
        <div style={{ ...card, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, marginBottom: 18, padding: 0 }}>
          <div style={{ padding: "20px 22px" }}>
            <div style={colHead}>PV (client side)</div>
            <div style={{ fontFamily: mono, fontSize: 30, marginTop: 6, fontVariantNumeric: "tabular-nums",
              color: res && res.pvEUR >= 0 ? C.green : C.red }}>
              {res ? fmtBigSigned(res.pvEUR) : "…"} <span style={{ fontSize: 14, color: C.mute }}>EUR</span>
            </div>
          </div>
          <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
            <div style={colHead}>PV % of notional</div>
            <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8,
              color: res && res.pvPct >= 0 ? C.green : C.red }}>
              {res ? fmtSigned(res.pvPct, 3) + "%" : "…"}</div>
            <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>of EUR notional</div>
          </div>
          <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
            <div style={colHead}>PV in USD</div>
            <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8,
              color: res && res.pvUSD >= 0 ? C.green : C.red }}>
              {res ? fmtBigSigned(res.pvUSD) : "…"}</div>
            <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>settlement currency</div>
          </div>
        </div>
        )}

        {res && res.fixings && (
          <div style={{ ...card, marginBottom: 18, padding: 0, overflow: "hidden" }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between",
              padding: "18px 22px 12px" }}>
              <div style={{ fontWeight: 700, fontSize: 15 }}>Fixings schedule
                <span style={{ color: C.faint, fontWeight: 400, fontSize: 12.5 }}> · forwards, discounting and expected flows</span>
              </div>
              <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
                {res.fixings.length} fixings · Σ E[CF] = PV
              </div>
            </div>
            <div className="sp-scroll" style={{ maxHeight: 338, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: mono, fontSize: 12.5 }}>
                <thead>
                  <tr>
                    {["#", "Fixing date", "τ (yrs)", "Forward", "DF", "Amount EUR", "P live",
                      ...(res.accOn ? ["P KO"] : []), "E[CF] EUR", "Σ E[CF]"].map((h, i) => (
                      <th key={h} style={{ position: "sticky", top: 0, background: C.card2, zIndex: 1,
                        padding: "9px 16px", textAlign: i < 2 ? "left" : "right",
                        fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase",
                        color: C.faint, fontFamily: sans, fontWeight: 600,
                        borderBottom: `1px solid ${C.line}` }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {res.fixings.map(f => (
                    <tr key={f.i} className="sp-row" style={{ borderBottom: `1px solid ${C.line}` }}>
                      <td style={{ padding: "8px 16px", color: C.faint }}>{f.i}</td>
                      <td style={{ padding: "8px 16px", color: C.text }}>{f.date}</td>
                      <td style={{ padding: "8px 16px", textAlign: "right", color: C.mute }}>{f.tau.toFixed(3)}</td>
                      <td style={{ padding: "8px 16px", textAlign: "right", color: C.amber }}>{fmtRate(f.fwd)}</td>
                      <td style={{ padding: "8px 16px", textAlign: "right", color: C.mute }}>{f.df.toFixed(4)}</td>
                      <td style={{ padding: "8px 16px", textAlign: "right", color: C.text }}>{fmtBig(f.amt)}</td>
                      <td style={{ padding: "8px 16px", textAlign: "right",
                        color: f.alive > 66 ? C.green : f.alive > 33 ? C.amber : C.red }}>{fmt(f.alive, 1)}%</td>
                      {res.accOn && (
                        <td style={{ padding: "8px 16px", textAlign: "right",
                          color: 100 - f.alive < 33 ? C.mute : 100 - f.alive < 66 ? C.amber : C.red }}>
                          {fmt(100 - f.alive, 1)}%</td>
                      )}
                      <td style={{ padding: "8px 16px", textAlign: "right",
                        color: f.ecf >= 0 ? C.green : C.red }}>{fmtSigned(f.ecf, 0)}</td>
                      <td style={{ padding: "8px 16px", textAlign: "right",
                        color: f.cum >= 0 ? C.text : C.red }}>{fmtSigned(f.cum, 0)}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr style={{ background: C.card2 }}>
                    <td colSpan={res.accOn ? 8 : 7} style={{ padding: "10px 16px", fontFamily: sans, fontSize: 11.5,
                      color: C.mute }}>Total expected discounted cash flow (equals PV)</td>
                    <td colSpan={2} style={{ padding: "10px 16px", textAlign: "right", fontWeight: 700,
                      color: res.pvEUR >= 0 ? C.green : C.red }}>{fmtSigned(res.pvEUR, 0)} EUR</td>
                  </tr>
                </tfoot>
              </table>
            </div>
            <div style={{ padding: "10px 22px 14px", fontSize: 11, color: C.faint }}>
              Forward = spot × e^((rUSD − rEUR)·τ). P live = probability the trade has not knocked out before the
              fixing. E[CF] = expected discounted client cash flow at that fixing under BSM, converted to EUR at spot.
              {res && res.accOn && " P KO = cumulative probability the barrier has cancelled the trade before the fixing (" + res.accStyle + " observation)."}
              {res && res.payAtMat && " ZC settlement: every flow is accumulated and paid at final maturity, so a single maturity discount factor applies to all fixings."}
            </div>
          </div>
        )}

        <div style={{ marginBottom: 18 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 12 }}>
            {greekCards.map((g, i) => {
              const active = selGreek === g.id;
              return (
                <div key={i} onClick={() => setSelGreek(active ? null : g.id)} className="sp-click"
                  style={{ background: active ? "rgba(74,125,240,0.10)" : C.card,
                    border: `1.5px solid ${active ? C.blue : C.line}`, borderRadius: 12,
                    padding: "13px 14px", cursor: "pointer", userSelect: "none" }}>
                  <div style={{ ...colHead, display: "flex", justifyContent: "space-between" }}>
                    <span>{g.name}</span>
                    <span style={{ color: active ? C.blue : C.faint }}>{active ? "▲" : "▾"}</span>
                  </div>
                  <div style={{ fontFamily: mono, fontSize: 16.5, color: g.color, marginTop: 7,
                    fontVariantNumeric: "tabular-nums" }}>{g.val}</div>
                  <div style={{ fontSize: 10.5, color: C.faint, marginTop: 3 }}>{g.unit}</div>
                </div>
              );
            })}
          </div>
          {!selGreek && res && (
            <div style={{ fontSize: 11.5, color: C.faint, marginTop: 8 }}>
              Click any Greek to display its profile vs spot. Negative values are shown in parentheses.
            </div>
          )}
          {selGreek && res && (
            <div style={{ ...card, borderTop: `2px solid ${C.blue}`, marginTop: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                <div style={{ fontWeight: 700, fontSize: 15 }}>
                  {greekMeta[selGreek].title}
                  <span style={{ color: C.faint, fontWeight: 400, fontSize: 12.5 }}> · {greekMeta[selGreek].unit}</span>
                </div>
                <div onClick={() => setSelGreek(null)}
                  style={{ cursor: "pointer", color: C.faint, fontSize: 12, fontFamily: mono }}>close ✕</div>
              </div>
              <div style={{ height: 240 }}>
                <ResponsiveContainer>
                  <ComposedChart data={res.prof[selGreek]} margin={{ top: 20, right: 16, bottom: 4, left: 8 }}>
                    <CartesianGrid stroke={C.line} strokeDasharray="2 4" vertical={false} />
                    <XAxis dataKey="s" type="number" domain={["dataMin", "dataMax"]}
                      tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                      tickFormatter={v => v.toFixed(4)} stroke={C.line} />
                    <YAxis tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                      tickFormatter={tickFmt} stroke={C.line} width={56} />
                    <Tooltip contentStyle={{ background: C.card2, border: `1px solid ${C.line}`,
                        fontFamily: mono, fontSize: 12, borderRadius: 8 }}
                      labelFormatter={v => "Spot " + (+v).toFixed(4)}
                      formatter={(v, nm) => [fmt(v, 0), nm]} />
                    <ReferenceLine y={0} stroke={C.mute} strokeWidth={1} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.S0} stroke={C.blue} strokeDasharray="4 3"
                      label={{ value: "spot " + res.S0.toFixed(4), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top" }} />
                    {!res.pivotOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.K} stroke={C.amber} strokeDasharray="4 3"
                        label={{ value: "K " + res.K.toFixed(4), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 12 }} />
                    )}
                    {res.pivotOn && (<>
                      <ReferenceLine ifOverflow="extendDomain" x={res.kLow} stroke={C.amber} strokeDasharray="4 3"
                        label={{ value: "KL " + res.kLow.toFixed(4), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 12 }} />
                      <ReferenceLine ifOverflow="extendDomain" x={res.pivotL} stroke={C.blue} strokeDasharray="4 3"
                        label={{ value: "P " + res.pivotL.toFixed(4), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                      <ReferenceLine ifOverflow="extendDomain" x={res.kHigh} stroke={C.amber} strokeDasharray="4 3"
                        label={{ value: "KH " + res.kHigh.toFixed(4), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 12 }} />
                    </>)}
                    {res.lkoOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.H} stroke={C.violet} strokeDasharray="4 3"
                        label={{ value: "LKO " + res.H.toFixed(4), fill: C.violet, fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                    )}
                    {res.accOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.koLevel} stroke={C.red} strokeDasharray="4 3"
                        label={{ value: "KO " + res.koLevel.toFixed(4), fill: C.red, fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                    )}
                    {res.ekiOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.E} stroke="#C89A4B" strokeDasharray="4 3"
                        label={{ value: "KI " + res.E.toFixed(4), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                    )}
                    {res.pivotEkiOn && (<>
                      <ReferenceLine ifOverflow="extendDomain" x={res.eLow} stroke="#C89A4B" strokeDasharray="4 3"
                        label={{ value: "KI " + res.eLow.toFixed(4), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                      <ReferenceLine ifOverflow="extendDomain" x={res.eHigh} stroke="#C89A4B" strokeDasharray="4 3"
                        label={{ value: "KI " + res.eHigh.toFixed(4), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                    </>)}
                    {greekMeta[selGreek].keys.map(([k, col, nm]) => (
                      <Line key={k} dataKey={k} name={nm} dot={false} stroke={col}
                        strokeWidth={2.2} type="monotone" connectNulls />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        <div style={{ ...card, marginBottom: 18, paddingBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>Payoff diagram
              <span style={{ color: C.faint, fontWeight: 400, fontSize: 12.5 }}>
                {res && res.kind === "DCD" ? " · final redemption at maturity"
                  : res && res.kind === "VAN" ? " · at expiry, net of premium"
                  : " · per fixing, client perspective"}</span>
            </div>
            {res && res.kind !== "DCD" && res.kind !== "VAN" && !res.accOn && !res.countOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              gain leg stops once {fmt(res.targetFig, 0)} figures accumulated
            </div>}
            {res && res.countOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              knocks out after {res.targetCount} gaining fixings
            </div>}
            {res && res.accOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              {res.accStyle} KO at {fmtRate(res.koLevel)} · {res.payAtMat ? "ZC settlement" : "rolling settlement"}
            </div>}
          </div>
          {res && res.kind === "VAN"
            ? <VanillaDiagram res={res} C={C} mono={mono} sans={sans} />
            : res && res.kind === "DCD"
            ? <DCDDiagram res={res} C={C} mono={mono} sans={sans} />
            : res && res.pivotOn
            ? <PivotPayoffDiagram res={res} C={C} mono={mono} sans={sans} />
            : <PayoffDiagram res={res} C={C} mono={mono} sans={sans} />}
          <div style={{ fontSize: 11, color: C.faint, marginTop: 2 }}>
            {res && res.kind === "VAN"
              ? `Net position at expiry: intrinsic value ${res.side === "Buy" ? "minus the premium paid" : "minus intrinsic, plus the premium received"}. Breakeven is the strike ${res.type === "Call" ? "plus" : "minus"} the premium in USD pips per EUR.`
              : res && res.kind === "DCD"
              ? `Nominal redemption only, coupon excluded (the coupon of ${fmtBig(res.N * res.cpn)} ${res.depCcy} is paid regardless). Flat leg: nominal repaid in ${res.depCcy}. Sloped leg: the bank repays the alternative currency converted at the strike, so the ${res.depCcy} value of the nominal falls with the fixing. The amber marker is the breakeven including the coupon.`
              : res && res.countOn
              ? `No KO level on the spot axis: the trade terminates after the ${res.targetCount}th gaining fixing, whatever the size of each gain. Every favourable fixing pays 1x in full; unfavourable fixings pay ×${res.L} and never count.`
              : res && res.accOn
              ? `KO marker at the barrier (${res.accStyle} observation): a fixing${res.accStyle === "American" ? " or any spot path" : ""} beyond it cancels that fixing and all remaining ones. Unlike the TARF there is no target: gains are unbounded until the barrier, losses are unbounded and leveraged.`
              : res && res.pivotOn
              ? `Tent shaped gain around the pivot, clipped at the KO Target (${fmt(res.targetFig, 0)} figures): near the pivot a single clean fixing can breach the full target.`
              : `KO marker at strike + target (${res ? fmt(res.targetFig, 0) : "…"} figures): the fixing level at which a single clean fixing breaches the full remaining target.`}
            {res && res.lkoOn
              ? " Beyond the LKO barrier the loss obligation is knocked out to 0x, and stays at 0x for all remaining fixings once triggered."
              : res && res.ekiOn
              ? " Between the strike and the KI barrier the client simply trades at market (no obligation); beyond the barrier the obligation knocks in at the strike on the leveraged notional, hence the jump."
              : res && res.pivotEkiOn
              ? " Between each strike and its KI barrier the client participates at market; beyond either barrier the leveraged obligation knocks in at that strike, hence the jumps. Losses never count toward the target."
              : res && res.pivotOn
              ? " Beyond either strike the client is obligated on the leveraged notional; losses never count toward the target."
              : " Losses are unbounded and never count toward the target."}
          </div>
        </div>

        <div style={card}>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>PV vs spot
            <span style={{ color: C.faint, fontWeight: 400, fontSize: 12.5 }}> · mark to market profile</span>
          </div>
          <div style={{ height: 240 }}>
            {res && (() => {
              const vals = res.prof.pv.map(d => d.v);
              const vMax = Math.max(...vals), vMin = Math.min(...vals);
              const zero = vMax <= 0 ? 0 : vMin >= 0 ? 1 : vMax / (vMax - vMin);
              return (
              <ResponsiveContainer>
                <ComposedChart data={res.prof.pv} margin={{ top: 20, right: 16, bottom: 4, left: 8 }}>
                  <defs>
                    <linearGradient id="pvSpotGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset={zero} stopColor={C.green} />
                      <stop offset={zero} stopColor={C.red} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={C.line} strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="s" type="number" domain={["dataMin", "dataMax"]}
                    tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                    tickFormatter={v => v.toFixed(4)} stroke={C.line} />
                  <YAxis tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                    tickFormatter={tickFmt} stroke={C.line} width={56} />
                  <Tooltip contentStyle={{ background: C.card2, border: `1px solid ${C.line}`,
                      fontFamily: mono, fontSize: 12, borderRadius: 8 }}
                    labelFormatter={v => "Spot " + (+v).toFixed(4)}
                    formatter={v => [fmt(v, 0) + " EUR", "PV"]} />
                  <ReferenceLine y={0} stroke={C.mute} strokeWidth={1} />
                  <ReferenceLine ifOverflow="extendDomain" x={res.S0} stroke={C.blue} strokeDasharray="4 3"
                    label={{ value: "spot " + res.S0.toFixed(4), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top" }} />
                  {!res.pivotOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.K} stroke={C.amber} strokeDasharray="4 3"
                      label={{ value: "K " + res.K.toFixed(4), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 12 }} />
                  )}
                  {res.pivotOn && (<>
                    <ReferenceLine ifOverflow="extendDomain" x={res.kLow} stroke={C.amber} strokeDasharray="4 3"
                      label={{ value: "KL " + res.kLow.toFixed(4), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 12 }} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.pivotL} stroke={C.blue} strokeDasharray="4 3"
                      label={{ value: "P " + res.pivotL.toFixed(4), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.kHigh} stroke={C.amber} strokeDasharray="4 3"
                      label={{ value: "KH " + res.kHigh.toFixed(4), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 12 }} />
                  </>)}
                  {res.lkoOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.H} stroke={C.violet} strokeDasharray="4 3"
                      label={{ value: "LKO " + res.H.toFixed(4), fill: C.violet, fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                  )}
                  {res.accOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.koLevel} stroke={C.red} strokeDasharray="4 3"
                      label={{ value: "KO " + res.koLevel.toFixed(4), fill: C.red, fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                  )}
                  {res.ekiOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.E} stroke="#C89A4B" strokeDasharray="4 3"
                      label={{ value: "KI " + res.E.toFixed(4), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                  )}
                  {res.pivotEkiOn && (<>
                    <ReferenceLine ifOverflow="extendDomain" x={res.eLow} stroke="#C89A4B" strokeDasharray="4 3"
                      label={{ value: "KI " + res.eLow.toFixed(4), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.eHigh} stroke="#C89A4B" strokeDasharray="4 3"
                      label={{ value: "KI " + res.eHigh.toFixed(4), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 24 }} />
                  </>)}
                  <Line dataKey="v" dot={false} stroke="url(#pvSpotGrad)" strokeWidth={2.4} type="monotone" />
                </ComposedChart>
              </ResponsiveContainer>
              );
            })()}
          </div>
          <div style={{ fontSize: 11, color: C.faint, marginTop: 8 }}>
            Forward to maturity: <span style={{ fontFamily: mono, color: C.text }}>
            {res ? fmtRate(res.fwd) : "…"}</span>.
            {res && res.kind === "VAN"
              ? ` Mark to market of the position in EUR before expiry: the smooth curve above the expiry hockey stick is time value, largest at the strike where gamma and vega ${res.side === "Buy" ? "peak" : "trough"}.`
              : res && res.kind === "DCD"
              ? " Mark to market of the embedded short option in EUR (the deposit leg itself is excluded): most negative where the option is deepest in the bank's favour, with vega concentrated near the strike."
              : res && res.countOn
              ? " Compared with a CIV TARF, the discrete version knocks out faster when spot drifts favourably (any small gain counts as one full unit), so its upside flattens earlier while the leveraged downside is identical."
              : res && res.accOn
              ? " PV rises with spot then collapses toward the KO barrier: the client loses the remaining accumulation exactly when it is most valuable, the accumulator's signature reversal and negative gamma near the barrier."
              : res && res.lkoOn
              ? " Note the PV floor forming near the LKO barrier: the client's downside is protected once the barrier knocks the leverage out."
              : res && res.ekiOn
              ? " The dead zone between strike and KI barrier softens the downside close to the strike; the leveraged liability only bites beyond the barrier."
              : res && res.pivotEkiOn
              ? " PV peaks near the pivot and the participation zones soften both flanks: the leveraged liabilities only bite beyond the KI barriers, so the profile is flatter around the strikes than the plain pivot."
              : res && res.pivotOn
              ? " PV peaks near the pivot where every fixing gains, and falls away on both sides: the client is effectively short a leveraged strangle with capped premium, so the position is short vol on both wings."
              : " Upside flattens as the target caps client gains; downside is the leveraged strip of forwards."}
          </div>
        </div>
      </div>
    </div>
  );
}
