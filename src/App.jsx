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
function digCallGK(S, K, T, rd, rf, s) {
  const sq = s * Math.sqrt(T);
  const d2 = (Math.log(S / K) + (rd - rf - s * s / 2) * T) / sq;
  return Math.exp(-rd * T) * normCdf(d2);
}
function digPutGK(S, K, T, rd, rf, s) {
  const sq = s * Math.sqrt(T);
  const d2 = (Math.log(S / K) + (rd - rf - s * s / 2) * T) / sq;
  return Math.exp(-rd * T) * normCdf(-d2);
}
// KO observed at maturity only: truncated vanilla (exact)
function koEuro(S, K, H, T, rd, rf, s, om) {
  if (om === 1) return Math.max(gk(S, K, T, rd, rf, s).call - gk(S, H, T, rd, rf, s).call - (H - K) * digCallGK(S, H, T, rd, rf, s), 0);
  return Math.max(gk(S, K, T, rd, rf, s).put - gk(S, H, T, rd, rf, s).put - (K - H) * digPutGK(S, H, T, rd, rf, s), 0);
}
// continuous KO: Reiner Rubinstein up-and-out call / down-and-out put (validated vs bridge MC)
function koAmer(S, K, H, T, rd, rf, s, om) {
  // knocked out already: spot at or beyond the barrier kills the option outright
  if (om === 1 && S >= H) return 0;
  if (om === -1 && S <= H) return 0;
  const b = rd - rf, r = rd, sq = s * Math.sqrt(T), mu = (b - s * s / 2) / (s * s);
  const phi = om === 1 ? 1 : -1, eta = om === 1 ? -1 : 1;
  const x1 = Math.log(S / K) / sq + (1 + mu) * sq, x2 = Math.log(S / H) / sq + (1 + mu) * sq;
  const y1 = Math.log(H * H / (S * K)) / sq + (1 + mu) * sq, y2 = Math.log(H / S) / sq + (1 + mu) * sq;
  const A = phi * S * Math.exp((b - r) * T) * normCdf(phi * x1) - phi * K * Math.exp(-r * T) * normCdf(phi * (x1 - sq));
  const Bf = phi * S * Math.exp((b - r) * T) * normCdf(phi * x2) - phi * K * Math.exp(-r * T) * normCdf(phi * (x2 - sq));
  const Cf = phi * S * Math.exp((b - r) * T) * Math.pow(H / S, 2 * (mu + 1)) * normCdf(eta * y1) - phi * K * Math.exp(-r * T) * Math.pow(H / S, 2 * mu) * normCdf(eta * (y1 - sq));
  const Df = phi * S * Math.exp((b - r) * T) * Math.pow(H / S, 2 * (mu + 1)) * normCdf(eta * y2) - phi * K * Math.exp(-r * T) * Math.pow(H / S, 2 * mu) * normCdf(eta * (y2 - sq));
  return Math.max(A - Bf + Cf - Df, 0);
}
// probability of touching the barrier before T (continuous)
function pHitBar(S, H, T, rd, rf, s, up) {
  const nu = rd - rf - s * s / 2, sq = s * Math.sqrt(T);
  if (up) { const a = Math.log(H / S); if (a <= 0) return 1;
    return Math.min(1, normCdf((-a + nu * T) / sq) + Math.exp(2 * nu * a / (s * s)) * normCdf((-a - nu * T) / sq)); }
  const a = Math.log(S / H); if (a <= 0) return 1;
  return Math.min(1, normCdf((-a - nu * T) / sq) + Math.exp(-2 * nu * a / (s * s)) * normCdf((-a + nu * T) / sq));
}
// light 3-point smoothing for Monte Carlo greek profiles
function smoothProf(a) {
  return a.map((pt, i, arr) =>
    i === 0 || i === arr.length - 1 ? pt : { ...pt, v: (arr[i - 1].v + 2 * pt.v + arr[i + 1].v) / 4 });
}
// one x-axis for the payoff diagram and every chart of a pricing
function axisDomain(levels, S0) {
  const ls = levels.filter(x => x != null && isFinite(x));
  const lo0 = Math.min(...ls, S0), hi0 = Math.max(...ls, S0);
  const pad = Math.max((hi0 - lo0) * 0.18, S0 * 0.035);
  return { lo: lo0 - pad, hi: hi0 + pad };
}
function interpCurve(curve, t) {
  if (!curve || !curve.length) return null;
  if (t <= curve[0].t) return curve[0].r;
  if (t >= curve[curve.length - 1].t) return curve[curve.length - 1].r;
  for (let i = 1; i < curve.length; i++) {
    if (t <= curve[i].t) {
      const a = curve[i - 1], b = curve[i];
      return a.r + ((b.r - a.r) * (t - a.t)) / (b.t - a.t);
    }
  }
  return curve[curve.length - 1].r;
}
// Malz smile: sigma(delta) = ATM - 2 RR (d - 1/2) + 16 BF (d - 1/2)^2, delta = forward call delta at ATM vol
function smileVol(atm, rr, bf, F, K, T) {
  const sq = Math.max(atm * Math.sqrt(Math.max(T, 1e-9)), 1e-9);
  const d1 = (Math.log(F / K) + 0.5 * atm * atm * T) / sq;
  const x = normCdf(d1) - 0.5;
  return Math.max(atm - 2 * rr * x + 16 * bf * x * x, 0.001);
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
          accOn, koLevel, accStyle, countOn, targetCount,
          capLossOn, targetS, koConvS, accelOn, accelFA } = params;
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
  let sum = 0, sumSq = 0, koCount = 0, lifeSum = 0, lkoCount = 0, ekiCount = 0, lossKoCount = 0;
  for (let p = 0; p < nPaths; p++) {
    let S = S0, Sp = S0, acc = 0, lAcc = 0, gains = 0, pv = 0, alive = true, life = n,
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
            // accelerator: the improved strike boosts every gaining fixing to d + FA·d²
            const gIntr = accelOn ? intr + accelFA * intr * intr : intr;
            const newAcc = acc + gIntr;
            if (newAcc >= target) {
              let pay = gIntr;
              if (koConv === "capped") pay = Math.max(target - acc, 0);
              else if (koConv === "none") pay = 0;
              cash = amtEURperFix * pay;
              terminate = true; koCount++;
            } else { acc = newAcc; cash = amtEURperFix * gIntr; }
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
            const lossCash = amtEURperFix * (beyond ? L : 1) * intr; // negative
            if (capLossOn) {
              const lossFig = (beyond ? L : 1) * (-intr); // positive, rate units
              const newL = lAcc + lossFig;
              if (newL >= targetS) {
                let payL = lossCash;
                if (koConvS === "capped") payL = -amtEURperFix * Math.max(targetS - lAcc, 0);
                else if (koConvS === "none") payL = 0;
                cash = payL; terminate = true; lossKoCount++;
              } else { lAcc = newL; cash = lossCash; }
            } else cash = lossCash;
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
           lossKoProb: lossKoCount / nPaths,
           expLifeFix: lifeSum / nPaths, stats };
}
/* ---------- formatting: NO minus sign anywhere ----------
   negatives are rendered accounting style: (1,234)              */
const raw = (x, d = 0) => {
  const s = Math.abs(x).toFixed(d);
  const [i, f] = s.split(".");
  const gi = i.replace(/\B(?=(\d{3})+(?!\d))/g, "."); // dot separates thousands, millions, …
  return f ? gi + "." + f : gi;
};
const fmt = (x, d = 0) =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${raw(x, d)})` : raw(x, d);
const fmtSigned = (x, d = 0) =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${raw(x, d)})` : "+" + raw(x, d);
const bigRaw = a => raw(a, a < 100 && a !== Math.round(a) ? 2 : 0);
const fmtBig = x =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${bigRaw(Math.abs(x))})` : bigRaw(Math.abs(x));
const fmtBigSigned = x =>
  x == null || !isFinite(x) ? "…" : x < 0 ? `(${bigRaw(Math.abs(x))})` : "+" + bigRaw(Math.abs(x));
const PAIRS = {
  "EUR/USD": { base: "EUR", quote: "USD", pip: 0.0001, dec: 4, vol: 8.0, spot0: 1.0850 },
  "GBP/USD": { base: "GBP", quote: "USD", pip: 0.0001, dec: 4, vol: 9.0, spot0: 1.2700 },
  "EUR/GBP": { base: "EUR", quote: "GBP", pip: 0.0001, dec: 4, vol: 7.0, spot0: 0.8550 },
  "USD/JPY": { base: "USD", quote: "JPY", pip: 0.01, dec: 3, vol: 10.0, spot0: 147.50 },
};
let RATE_DEC = 4; // display precision for the active pair
const fmtRate = (x, d = RATE_DEC) =>
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
          countOn, targetCount, capLossOn, accelOn, accelFA } = res;
  const tS = capLossOn ? (res.targetSRate != null ? res.targetSRate : res.targetSFig / 100) : 0;
  const capX = capLossOn ? K - omega * (tS / Math.max(L, 1)) : null;
  const tR = res.targetRate != null ? res.targetRate : res.targetFig / 100;
  const t = accOn ? Math.abs(koLevel - K)
    : countOn ? Math.max(Math.abs(S0 - K) * 1.8, 0.028 * K)
    : accelOn ? (accelFA > 1e-9 ? (Math.sqrt(1 + 4 * accelFA * tR) - 1) / (2 * accelFA) : tR)
    : tR;
  const KO = K + omega * t;
  const W = 880, HH = 380, mL = 84, mR = 40, mT = 46, mB = 74;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const barrierS = lkoOn ? H : ekiOn ? E : capLossOn ? capX : null;
  const lossEnd = barrierS != null ? barrierS : K - omega * 0.55 * t;
  const lossDrawEnd = barrierS != null ? barrierS - omega * 0.22 * t : lossEnd;
  let x0 = Math.min(lossDrawEnd, B, K, KO, S0) - 0.06 * t;
  let x1 = Math.max(lossDrawEnd, B, K, KO, S0) + 0.30 * t;
  if (res.axLo != null) { x0 = res.axLo; x1 = res.axHi; }
  const pay = s => {
    const intr = omega * (s - K);
    if (intr > 0) {
      const g = accelOn ? intr + accelFA * intr * intr : intr;
      return A * Math.min(g, accelOn ? tR : t);
    }
    if (lkoOn) {
      const beyondLKO = omega === 1 ? s <= H : s >= H;
      if (beyondLKO) return 0;
    }
    if (ekiOn) {
      const knockedIn = omega === 1 ? s < E : s > E;
      return knockedIn ? A * L * intr : 0;
    }
    const beyond = omega === 1 ? s < B : s > B;
    const raw2 = A * (beyond ? L : 1) * intr;
    if (capLossOn) return Math.max(raw2, -A * tS);
    return raw2;
  };
  const cv = 1 / S0;
  const lossFloorRef = lkoOn ? H : ekiOn ? lossDrawEnd : capLossOn ? capX : (omega === 1 ? x0 : x1);
  const lossFloorVal = capLossOn ? -A * tS : A * L * omega * (lossFloorRef - K);
  const yTop = A * (accelOn ? tR : t);
  const yBot = Math.min(lossFloorVal, 0);
  const yMax = yTop * 1.18, yMin = yBot * 1.12 - yTop * 0.05;
  const X = s => mL + ((s - x0) / (x1 - x0)) * iw;
  const Y = v => mT + ((yMax - v) / (yMax - yMin)) * ih;
  const Y0 = Y(0);
  const gainB = X(KO);
  const bx = X(B);
  const hasLevGap = Math.abs(B - K) > 1e-9;
  const lossEdgeS = lkoOn ? H : ekiOn ? lossDrawEnd : capLossOn ? capX : (omega === 1 ? x0 : x1);
  const dir = omega === 1 ? "Buyer (Long)" : "Seller (Short)";
  const tagW = 26;
  const midGain = (K + KO) / 2;
  const midLoss = ekiOn ? (E + lossDrawEnd) / 2 : (B + lossEdgeS) / 2;
  const midPart = ekiOn ? (K + E) / 2 : null;
  const title = accelOn ? "Accelerator TARF" : capLossOn ? "Cap Loss TARF" : countOn ? "Discrete TARF" : accOn ? "Accumulator"
    : lkoOn ? "Liability Knock Out TARF" : ekiOn ? "EKI TARF" : "Vanilla TARF";
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <text x={W / 2} y={20} textAnchor="middle" fontFamily={sans} fontSize="15" fontWeight="700" fill={C.text}>
        {title}
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        {res.pair} {dir} · {fmtBig(A)} {res.base} per fixing
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
      {capLossOn && (<g>
        <line x1={X(capX)} y1={Y(-A * tS)} x2={omega === 1 ? mL : W - mR} y2={Y(-A * tS)}
          stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
        <text x={omega === 1 ? mL + 8 : W - mR - 8} textAnchor={omega === 1 ? "start" : "end"}
          y={Y(-A * tS) - 8} fontFamily={mono} fontSize="10.5" fill={C.red}>
          Loss capped at ({fmtBig(A * tS * cv)}) {res.base}
        </text>
      </g>)}
      {accelOn ? (
        <polyline
          points={Array.from({ length: 25 }, (_, i) => {
            const s = K + ((omega === 1 ? 1 : -1) * t * i) / 24;
            return `${X(s)},${Y(pay(s))}`;
          }).join(" ")}
          fill="none" stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      ) : (
        <line x1={X(K)} y1={Y0} x2={gainB} y2={Y(yTop)} stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      )}
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
      {capLossOn && (<g>
        <line x1={X(capX)} y1={Y0} x2={X(capX)} y2={Y0 + 26} stroke={C.red} strokeDasharray="3 3" strokeWidth="1" />
        <path d={`M ${X(capX) - 17} ${Y0 + 30} h34 v18 h-34 z M ${X(capX) - 17} ${Y0 + 30} L ${X(capX)} ${Y0 + 18} L ${X(capX) + 17} ${Y0 + 30} z`}
          fill={C.red} />
        <text x={X(capX)} y={Y0 + 43} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill="#FFFFFF">CAP</text>
        <text x={X(capX)} y={Y0 + 62} textAnchor="middle" fontFamily={sans} fontSize="9.5" fontWeight="700" fill={C.red}>Loss</text>
        <text x={X(capX)} y={Y0 + 76} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(capX)}</text>
      </g>)}
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
      {(() => {
        const below = pay(S0) > 0;
        let sx = X(S0), anchor = "middle";
        if (below) {
          const stems = [X(K), ...(!countOn ? [gainB] : []), ...(lkoOn ? [X(H)] : []),
            ...(ekiOn ? [X(E)] : []), ...(capLossOn ? [X(capX)] : [])];
          const near = stems.find(tx => Math.abs(tx - sx) < 48);
          if (near !== undefined) { anchor = sx >= near ? "start" : "end"; sx = near + (sx >= near ? 24 : -24); }
        }
        return (
          <text x={sx} y={below ? Y0 + 18 : Y0 - 10} textAnchor={anchor}
            fontFamily={mono} fontSize="10" fill={C.blue}>
            Spot {fmtRate(S0)}
          </text>
        );
      })()}
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
          0x, No obligation
        </text>
      )}
      {ekiOn && (
        <text x={X(midPart)} textAnchor="middle" y={Y0 - 26} fontFamily={sans}
          fontSize="10.5" fontWeight="700" fill={C.green}>
          Participation at market
        </text>
      )}
      <text x={gainB > W - mR - 170 ? gainB - 10 : gainB + 8}
        textAnchor={gainB > W - mR - 170 ? "end" : "start"}
        y={Y(yTop) - 10} fontFamily={mono} fontSize="10.5" fill={C.green}>
        {countOn ? "Stops after gain #" + targetCount
          : accOn ? "Cancelled at KO" : "Max +" + fmtBig(yTop * cv) + " " + res.base + " at KO"}
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
  if (res.axLo != null) { x0 = res.axLo; x1 = res.axHi; }
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
        {side} {res.pair} {type}
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        {type === "Call" ? `${res.base} call / ${res.quote} put` : `${res.base} put / ${res.quote} call`} · {fmtBig(eurN)} {res.base} notional ·
        payoff at expiry net of premium, in {res.base}
      </text>
      <line x1={mL} y1={Y0} x2={W - mR + 6} y2={Y0} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${W - mR + 6},${Y0} ${W - mR - 2},${Y0 - 4} ${W - mR - 2},${Y0 + 4}`} fill={C.mute} />
      <line x1={mL} y1={mT - 4} x2={mL} y2={HH - mB + 6} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${mL},${mT - 10} ${mL - 4},${mT - 2} ${mL + 4},${mT - 2}`} fill={C.mute} />
      <text x={mL - 10} y={mT + 4} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>PROFIT</text>
      <text x={mL - 10} y={HH - mB} textAnchor="end" fontFamily={sans} fontSize="11" fill={C.mute}>(LOSS)</text>
      <text x={mL + iw / 2} y={HH - 16} textAnchor="middle" fontFamily={sans} fontSize="12"
        fontWeight="600" fill={C.mute}>
        {res.pair} at expiry
      </text>
      {/* premium guide */}
      <line x1={mL} y1={Y(-sign * premEUR)} x2={W - mR} y2={Y(-sign * premEUR)}
        stroke={C.line} strokeDasharray="3 4" />
      <text x={mL - 8} y={Y(-sign * premEUR) + 4} textAnchor="end" fontFamily={mono} fontSize="10.5"
        fill={sign === 1 ? C.red : C.green}>
        {sign === 1 ? `(${bigRaw(premEUR)})` : "+" + bigRaw(premEUR)}
      </text>
      <text x={W - mR - 4} y={Y(-sign * premEUR) - 6} textAnchor="end" fontFamily={sans} fontSize="9.5" fill={C.faint}>
        Premium {sign === 1 ? "paid" : "received"}
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
        Breakeven {fmtRate(breakeven)}
      </text>
      {/* spot */}
      <circle cx={X(S0)} cy={Y0} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      {(() => {
        const below = net(S0) > 0;
        let sx = X(S0), anchor = "middle";
        if (below && Math.abs(X(K) - sx) < 48) {
          anchor = sx >= X(K) ? "start" : "end";
          sx = X(K) + (sx >= X(K) ? 24 : -24);
        }
        return (
          <text x={sx} y={below ? Y0 + 18 : Y0 - 10} textAnchor={anchor}
            fontFamily={mono} fontSize="10" fill={C.blue}>
            Spot {fmtRate(S0)}
          </text>
        );
      })()}
    </svg>
  );
}

/* ———————— sharkfin redemption diagram ———————— */
function SharkfinDiagram({ res, C, mono, sans }) {
  if (!res) return null;
  const { S0, K, H, om, partPct, rebPct, maxCpnPct, depCcy } = res;
  const W = 880, HH = 400, mL = 100, mR = 40, mT = 46, mB = 78;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const span = Math.max(Math.abs(H - K) * 2.1, 0.06 * K);
  let x0 = Math.min(K, H, S0) - 0.55 * span;
  let x1 = Math.max(K, H, S0) + 0.55 * span;
  if (res.axLo != null) { x0 = res.axLo; x1 = res.axHi; }
  const yTopPct = 100 + Math.max(maxCpnPct, rebPct) * 1.28 + 1.5;
  const yBotPct = 100 - Math.max(maxCpnPct, 3) * 0.45;
  const X = s => mL + ((s - x0) / (x1 - x0)) * iw;
  const Y = v => mT + ((yTopPct - v) / (yTopPct - yBotPct)) * ih;
  const yFloor = Y(100), yPeak = Y(100 + maxCpnPct), yReb = Y(100 + rebPct);
  const tagW = 26;
  const finStart = om === 1 ? X(K) : X(H), finEnd = om === 1 ? X(H) : X(K);
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <text x={W / 2} y={20} textAnchor="middle" fontFamily={sans} fontSize="15" fontWeight="700" fill={C.text}>
        Sharkfin Note
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontFamily={sans} fontSize="12.5" fill={C.mute}>
        {depCcy} deposit · capital floor 100% ·{" "}
        {om === 1 ? `call up & out, bullish ${res.base}` : `put down & out, bearish ${res.base}`} ·
        redemption in % of nominal
      </text>
      {/* axes */}
      <line x1={mL} y1={HH - mB} x2={W - mR + 6} y2={HH - mB} stroke={C.mute} strokeWidth="1.2" />
      <polygon points={`${W - mR + 6},${HH - mB} ${W - mR - 2},${HH - mB - 4} ${W - mR - 2},${HH - mB + 4}`} fill={C.mute} />
      <line x1={mL} y1={mT - 4} x2={mL} y2={HH - mB + 6} stroke={C.mute} strokeWidth="1.2" />
      <text x={mL - 46} y={mT + ih / 2} transform={`rotate(-90 ${mL - 46} ${mT + ih / 2})`} textAnchor="middle"
        fontFamily={sans} fontSize="11" letterSpacing="0.14em" fill={C.faint}>REDEMPTION %</text>
      <text x={mL + iw / 2} y={HH - 14} textAnchor="middle" fontFamily={sans} fontSize="12" fontWeight="600" fill={C.mute}>
        {res.pair} at maturity
      </text>
      {/* capital floor guide */}
      <line x1={mL} y1={yFloor} x2={W - mR} y2={yFloor} stroke={C.line} strokeDasharray="3 4" />
      <text x={mL + 8} y={yFloor - 8} textAnchor="start" fontFamily={mono} fontSize="10.5" fill={C.mute}>
        Capital floor 100%
      </text>
      {/* pre-strike flat leg at 100 */}
      <line x1={om === 1 ? mL : finEnd} y1={yFloor} x2={om === 1 ? finStart : W - mR} y2={yFloor}
        stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      {/* fin */}
      <line x1={om === 1 ? finStart : X(K)} y1={yFloor} x2={om === 1 ? finEnd : X(H)} y2={yPeak}
        stroke={C.green} strokeWidth="3.4" strokeLinecap="round" />
      {/* KO drop + post-KO leg at 100 + rebate */}
      <line x1={X(H)} y1={yPeak} x2={X(H)} y2={yReb} stroke={C.red} strokeWidth="2.6" strokeDasharray="6 5" />
      <line x1={X(H)} y1={yReb} x2={om === 1 ? W - mR : mL} y2={yReb} stroke={C.red} strokeWidth="3.4" strokeLinecap="round" />
      {/* peak annotation */}
      <text x={X(H) + (om === 1 ? -8 : 8)} y={yPeak - 10} textAnchor={om === 1 ? "end" : "start"}
        fontFamily={mono} fontSize="10.5" fill={C.green}>
        Max coupon +{fmt(maxCpnPct, 2)}%
      </text>
      <text x={om === 1 ? W - mR - 4 : mL + 4} y={yReb - 8} textAnchor={om === 1 ? "end" : "start"}
        fontFamily={mono} fontSize="10.5" fill={C.red}>
        After KO: 100% {rebPct > 0 ? `+ ${fmt(rebPct, 2)}% rebate` : "only"}
      </text>
      {/* K tag */}
      <line x1={X(K)} y1={HH - mB} x2={X(K)} y2={HH - mB + 26} stroke={C.text} strokeDasharray="3 3" strokeWidth="1" />
      <rect x={X(K) - tagW / 2} y={HH - mB + 26} width={tagW} height={20} rx="3" fill="#060A14" stroke={C.text} strokeWidth="0.8" />
      <text x={X(K)} y={HH - mB + 40} textAnchor="middle" fontFamily={mono} fontSize="11.5" fontWeight="700" fill="#FFFFFF">K</text>
      <text x={X(K)} y={HH - mB + 60} textAnchor="middle" fontFamily={mono} fontSize="11" fill={C.text}>{fmtRate(K)}</text>
      {/* KO house tag */}
      <g>
        <line x1={X(H)} y1={HH - mB} x2={X(H)} y2={HH - mB + 26} stroke={C.red} strokeDasharray="3 3" strokeWidth="1" />
        <path d={`M ${X(H) - 17} ${HH - mB + 30} h34 v18 h-34 z M ${X(H) - 17} ${HH - mB + 30} L ${X(H)} ${HH - mB + 18} L ${X(H) + 17} ${HH - mB + 30} z`}
          fill={C.red} />
        <text x={X(H)} y={HH - mB + 43} textAnchor="middle" fontFamily={mono} fontSize="10.5" fontWeight="700" fill="#FFFFFF">KO</text>
        <text x={X(H)} y={HH - mB + 62} textAnchor="middle" fontFamily={sans} fontSize="9.5" fontWeight="700" fill={C.red}>Barrier</text>
        <text x={X(H)} y={HH - mB + 76} textAnchor="middle" fontFamily={mono} fontSize="10.5" fill={C.text}>{fmtRate(H)}</text>
      </g>
      {/* spot */}
      <circle cx={X(S0)} cy={HH - mB} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      {(() => {
        let sx = X(S0), anchor = "middle";
        const stems = [X(K), X(H)];
        const near = stems.find(tx => Math.abs(tx - sx) < 48);
        if (near !== undefined) { anchor = sx >= near ? "start" : "end"; sx = near + (sx >= near ? 24 : -24); }
        return (
          <text x={sx} y={HH - mB - 10} textAnchor={anchor} fontFamily={mono} fontSize="10" fill={C.blue}>
            Spot {fmtRate(S0)}
          </text>
        );
      })()}
    </svg>
  );
}

/* ———————— DCD redemption diagram ———————— */
function DCDDiagram({ res, C, mono, sans }) {
  if (!res) return null;
  const { S0, K, depCcy, N, breakeven } = res;
  const conv = depCcy === res.base; // conversion happens above K for a base-ccy deposit
  const alt = conv ? res.quote : res.base;
  const W = 880, HH = 380, mL = 100, mR = 40, mT = 46, mB = 74;
  const iw = W - mL - mR, ih = HH - mT - mB;
  const red = N; // nominal only, coupon excluded
  const redeem = s => conv ? (s <= K ? red : red * K / s) : (s >= K ? red : red * s / K);
  const span = 0.16 * K;
  let x0 = conv ? K - 0.7 * span : K - 1.6 * span;
  let x1 = conv ? K + 1.6 * span : K + 0.7 * span;
  x0 = Math.min(x0, S0 - 0.15 * span, breakeven - 0.15 * span);
  x1 = Math.max(x1, S0 + 0.15 * span, breakeven + 0.15 * span);
  if (res.axLo != null) { x0 = res.axLo; x1 = res.axHi; }
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
        {res.pair} at maturity
      </text>
      {/* nominal guide */}
      <line x1={mL} y1={Y(red)} x2={W - mR} y2={Y(red)} stroke={C.line} strokeDasharray="3 4" />
      <text x={mL + 8} y={Y(red) - 10} textAnchor="start" fontFamily={mono} fontSize="10.5" fill={C.green}>
        {fmtBig(red)}
      </text>
      <text x={W - mR - 4} y={Y(red) - 10} textAnchor="end" fontFamily={sans} fontSize="9.5" fill={C.faint}>
        Nominal
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
        Breakeven incl. coupon {fmtRate(breakeven)}
      </text>
      {/* spot */}
      <circle cx={X(S0)} cy={Ybase} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      {(() => {
        let sx = X(S0), anchor = "middle";
        if (Math.abs(X(breakeven) - sx) < 42) {
          anchor = sx >= X(breakeven) ? "start" : "end";
          sx = X(breakeven) + (sx >= X(breakeven) ? 10 : -10);
        }
        return (
          <text x={sx} y={Ybase - 10} textAnchor={anchor}
            fontFamily={mono} fontSize="10" fill={C.blue}>
            Spot {fmtRate(S0)}
          </text>
        );
      })()}
      {/* region labels: green sits under its flat leg, red sits under the tail of the curve */}
      {(() => {
        let gx = conv ? X((x0 + K) / 2) : X((K + x1) / 2);
        if (Math.abs(gx - X(breakeven)) < 70) gx += gx >= X(breakeven) ? 55 : -55;
        return (
          <text x={gx} y={Y(red) + 18} textAnchor="middle"
            fontFamily={sans} fontSize="10.5" fontWeight="700" fill={C.green}>
            Nominal repaid in {depCcy}
          </text>
        );
      })()}
      {(() => {
        // anchor beneath the low end of the curve: the region below its endpoint is always empty
        const endS = conv ? x1 : x0;
        const yLab = Math.min(Y(redeem(endS)) + 24, Ybase - 12);
        return (
          <text x={conv ? X(x1) - 2 : X(x0) + 2} y={yLab} textAnchor={conv ? "end" : "start"}
            fontFamily={sans} fontSize="10.5" fontWeight="700" fill={C.red}>
            Converted into {alt} at K
          </text>
        );
      })()}
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
  if (res.axLo != null) { x0 = res.axLo; x1 = res.axHi; }
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
        {res.pair} two sided · {fmtBig(A)} {res.base} per fixing · sells at {fmtRate(kH)} above pivot, buys at {fmtRate(kL)} below
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
          KO Target · +{fmtBig(A * t * cv)} {res.base} ({fmt(res.targetFig, 0)} fig)
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
          fontSize="9.5" fontWeight="700" fill={C.green}>Participation</text>
        <text x={X((kH + eH) / 2)} textAnchor="middle" y={Y0 - 26} fontFamily={sans}
          fontSize="9.5" fontWeight="700" fill={C.green}>Participation</text>
      </g>)}
      {/* spot */}
      <circle cx={X(S0)} cy={Y0} r="5" fill="none" stroke={C.blue} strokeWidth="2" />
      {(() => {
        let sx = X(S0), anchor = "middle";
        const legs = [X(kL), X(kH)];
        const near = legs.find(tx => Math.abs(tx - sx) < 44);
        if (near !== undefined) { anchor = sx >= near ? "start" : "end"; sx = near + (sx >= near ? 20 : -20); }
        return (
          <text x={sx} y={Y0 - 12} textAnchor={anchor}
            fontFamily={mono} fontSize="10" fill={C.blue}>
            Spot {fmtRate(S0)}
          </text>
        );
      })()}
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

// ---------- design tokens & shared inputs (module scope so fields keep focus) ----------
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

const Num = ({ v, set, step = "any", min }) => {
  const [txt, setTxt] = useState(null);
  return (
    <input type="number" className="sp-input" step={step} min={min} style={input}
      value={txt !== null ? txt : (v === "" || v == null ? "" : v)}
      onFocus={() => setTxt(v === "" || v == null ? "" : String(v))}
      onChange={e => {
        const t = e.target.value;
        setTxt(t);
        const n = parseFloat(t);
        set(t === "" ? "" : (isFinite(n) ? n : v));
      }}
      onBlur={() => setTxt(null)} />
  );
};
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
const NotionalInput = ({ v, set }) => (
    <input type="text" inputMode="numeric" className="sp-input"
      value={v === "" ? "" : fmt(v, 0)}
      style={{ ...input, fontSize: 15, letterSpacing: "0.02em" }}
      onChange={e => {
        const raw2 = e.target.value.replace(/[^0-9]/g, "");
        set(raw2 === "" ? "" : +raw2);
      }} />
  );
// rate input: always displays 4 decimals when not being edited
const RateNum = ({ v, set }) => {
  const [txt, setTxt] = useState(null);
  return (
    <input type="number" className="sp-input" step={(1 / 10 ** RATE_DEC).toFixed(RATE_DEC)} inputMode="decimal"
      value={txt !== null ? txt : (v === "" || !isFinite(+v) ? "" : (+v).toFixed(RATE_DEC))}
      style={input}
      onFocus={() => setTxt(v === "" || !isFinite(+v) ? "" : String(+v))}
      onChange={e => {
        setTxt(e.target.value);
        const n = parseFloat(e.target.value);
        set(e.target.value === "" ? "" : (isFinite(n) ? n : v));
      }}
      onBlur={() => setTxt(null)} />
  );
};
const Field = ({ name, children, hint }) => (
    <div>
      <span style={label}>{name}</span>
      {children}
      {hint && <div style={{ fontSize: 11, color: C.faint, marginTop: 5, fontFamily: mono }}>{hint}</div>}
    </div>
  );


// ---------- Ratex brand: glyph, nav pill, site header ----------
const Glyph = ({ size = 30 }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" style={{ flexShrink: 0, display: "block" }}>
    <defs>
      <linearGradient id="rxg" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#2C5FD8" />
        <stop offset="55%" stopColor="#5A8DF0" />
        <stop offset="100%" stopColor="#9DB9FA" />
      </linearGradient>
    </defs>
    {/* thin elegant ring */}
    <circle cx="24" cy="24" r="21.5" fill="none" stroke="url(#rxg)" strokeWidth="1.6"
      opacity="0.9" />
    {/* rising rate curve, drawn fine */}
    <path d="M 11 32 C 17 32, 19 22, 24 20 C 29 18, 31 13, 37 12"
      fill="none" stroke="url(#rxg)" strokeWidth="2.4" strokeLinecap="round" />
    {/* terminal point, softly glowing */}
    <circle cx="37" cy="12" r="2.6" fill="#9DB9FA" />
    <circle cx="37" cy="12" r="4.8" fill="#5A8DF0" opacity="0.22" />
  </svg>
);
const Wordmark = ({ size = 21 }) => (
  <span style={{ fontSize: size * 1.06, fontWeight: 600, letterSpacing: "0.045em",
    fontFamily: "Georgia, 'Times New Roman', serif", lineHeight: 1,
    backgroundImage: "linear-gradient(105deg, #FFFFFF 30%, #AFC6FA 100%)",
    WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>
    Ratex
  </span>
);
// real bank logos, embedded so no extra files are needed
const BANK_LOGOS = {
  "bank_of_america.png": "data:image/png;base64,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",
  "barclays.png": "data:image/png;base64,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",
  "bnp_paribas.png": "data:image/png;base64,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",
  "citi.png": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAeAAAACBCAIAAACXaf6JAAB9iklEQVR42u1dd4BdVdGfmXPuK7tvexohnZAQaghVpIMKotJFEEQQAQVBaSp+CnYBRSmCiCAI0nvvvXcIAUJCSO/ZXt6+e87M98e59777dt9udjebkOAb1/D27Xv3nnbnzJn5zW9QRKAkayZuDBEx+A0ABBAxfB9EAMT9j0GCjwC6P6J7ASAFlwMBAUAARABERAiu776KApL/Uniz8GolKUlJPg+CJQU9YIkPXaSOC/7MDCJIBEhAiCKRkpUClZx/geG/AjFNKyJOzbtrAgACIsVVdpc2IJb0dElKssFLSUEPSAIrOaYTnU5lEWEQQEVIJCIIIIACYjo6ciuW5ZYuNsuXmxUrubneNjbZ1jZpbweTZZ9BBBWh9iSZUuVpqqiiqiqvqloPHaZHjEiO2MirGUKehyAIIO7GbMU1wtnXwR9iW0fJoi5JSTZkKSnoPkrXcYqbq8IMwogESoEIIhrm3JLF2XmfdsyamfvoA3/ObLtkmW1YCa1tkOsEazBQ687FEdOiIiAgzlgGAEWotSTTWFlJdUP16NGJCRNTm2+ZGj8hNW6CV1NDoXEt1gIAUmBWx83z8EYlKUlJNjApKejVS95GhgLnsjADC2rlVKI1pnP5stb332t/9aXs+9P9T2bJyhXU0YGApBV6GrWHpEUBALmrCQBK+AKgq7HrzHQRYAZrxfji+2xZCCGToY1GJiZtkZw6NbPTzulNN0tUh8rasrBFpQJNXeh7KSnqkpRkA5KSgu5VBCR0J8TeFGFGRCQSRLa27aMPWl54pvWlF/z33oHlK8D4ytOYSKL2QClBABFgCbQtAETaHgvcz5j/Y+BCyQsiICISuKtZC7lO7vSNGEiVqVFjkjvunPnCrhVf3D09ciSIIABbCwhIKrwAQtd4ZklKUpL1WkoKurgIBI7mbq4MQaUQwAK2zvqw+YlH2x9/zJ/xHjQ1kvIonSLPEyJxZq/kNTCG/whGtrj7HaI3w9sIIAYYDfeNAkUO4tRroLIRLEtnlrMdhkgN3yi53Q4V+x1QudteqeHDUUTcdkL5iGIc+vFZD3NJSlKS3qSkoIuIdFfN1gIgKkLEzsaGhmeebL7r9tzrr0D9KuUlKJ0Gz4tgG+4riCFSDhEkULUAgIICgCgAKCFWAyWvuN0NIe8ACb8RXA8i5F7+N0RAQhDI5WxHOwvDmHFle+xTc/Dhme229xJJEQFrBRGJAJy1jlIKIJakJOu3lBR0gYSGc8yctRYQiJQgtn4yu/7OW1rvu1c+mUUEqqwcPC0iYCVQvs4WFgSMLGMQ91bwd+iqFmMWdLemADjdHhnSEje+A9xz3toHAEQgQgDIZm17h00lEtN2rDrsW9UHfCNVXS3CYhmQUGHcgRIAsktSkpKsZ1JS0E5EILBKgxwQAGFGECAlAC3vvbvyv9d2PHgfrlyp0uWYTgkARPZyBHxz5m6YRBJzbKyFFrt/JLx7uBuEd0RUCpltW6vxjZo4qeKIo+oO+1bZRiOFGZiFCBFLIcSSlGR9lpKCjoOGIVBvwsKMSgliy7vvrLj6io5HHlCtzSpTBQnPKTgIvxNYy+h0vPNKrFs1FzlLAjvcuUZCd4hSCALZDr+9A0aPrTrimLpjji0bsZFw0MegH4giUlLQJSnJeiUlBV2ozgKEBoGi1tkzl195afs9d6u2NlVZAVqLtRBosZgWltD5gLFwYLHbxJK5Xf5K9EfnXo7cF1F40JnGQTLKajoReFciI961J5xfIiSCbNa0tsjYTaq/e/zQb383WV3NEXq6m5SUdUlK8pnL/66Cdkq5O0iDtM6uXLnsmn80X/8vvaqeqipFa7A2D+oIlGkPSdUROCNQksEGgHmlG1rZcV9392uE13Hp4SJBW53yltX5ToJGFmppQEJFku00rS24+RZ1p5059OsHKS/BxgARxnDTJe1ckpKsD/I/qqDzKLrAfBZ33mfmlffdveLPf4RZsxKVFeJ5wgZiFmyod3u4KMYvHHN7xIxn6DLe3RNUCjHQCPn7xy1wkFDN96JMQ+s+3FCcNY1IStracrnOxF77jvjpr6q2mRocDkKPB5Q0dUlKsh7I/6KClnxsDSBmOLd8Mnvx73/lP/SgTiWxrMzprHyCB0IvwDQRAYwB4pxqDOzkGHIZHHg5aoiAdMtKcbG76EMiERteoJWdfyTaAArzEYvhQYKmOJs6cnoQom1q8sszVSecPOKUHyczmQJTei3DpQt3qtI+UJL1QqLYO3a3nD4L+Z9T0E5bBfwVAGAtKWWsWXbjfxou+iPVL9fVNezSTCDv0ehJSwV6z1maUuhIhhD05tzI1orvi/HFWGEBECEQUqCUkEbCIM7IFlnAGmDr1DARoqfBS6DWQASILsU89FZjF4dJUQs/glWHGOx8CJGszTU20rbThv/mgrqdviDWCARw6cD/M/iMSyJBI/KtK5nqJflsJToAQxggwnUf8O8m/1sKWmKYZGARtqS9toXzF/7q57kH70tmKiSZEGMA8i4KKa6eChJJIMobkVApEyCL5HLc2WmND0SSqcS6IXrERomNRqrhI9TQod6QYVRdpcvLOZlCTxOAsIjvS0fWtLbYhga7fKldscJftsQuWWiXr+D6VdjRTojkJTGVBK0FECRQ1mEwERCwx1ZHiA/EyPEOiKiUtLR2alV16k9GnfoTnUyyMTF3xyAbE4HfpTBM2pVhtSQlWccS0bdDxLWOn7mO/l9S0DFiUBHHc6RXPv7I0p+fqRYu1LU1XODTCFg4ul4jGLLQCRB+FBAACYnAGM5muTPLXhKHj9ATJ6W33jq91TaJseOSG22s64Zopfro3HXGPiOazg5/2bLc4kXZTz7ueOft3Psf+HM/gcaVSpDSaUymgFDYAsec38Eii8KSXboQD1SKiIDSxKazoUnv9aXRF/wlM25C5O4InNGDYUYXLDZ3YGAmcmx+vbqQSlKStSQFMFsEEMsMAIooz9/72cn/ioKOKO/BZaAQGbaL/nZR81//nPSUpMvAGIhlnEjPztzAnZG3ohGVArbS3s7ZnK3I6MlTynberWynncq33T45dKiDsAXugohxPxJEB90T6BY/DD4ASCpsFAiiNX720zktb77e8cpLHa+9xPPnkbE6nYZUCoKs9Dw+O2hiNzsgxHWE/ZHQlG5oNCOGDb/w0iFf2k+MD6RClzTKGqxWiUKygfEMlpkQyfGWREMEJR1dknUp+UO1uOMos1IUK4f0GYfK/ycUtBvigCPUWtK6Y+WKeef82Dxwb6KmhhGBOb9ZdpuQSBmHOLeQB4MUKIRcp21t5WTKm7JN+b77VOz75fLJm3vpdHhTFssgAOSCg131T5FSLEU7IADCAIJIQOT0OgPkGupb3369+ZFHOp5/hufO0YqovAK0ClAZAMGJoWfzNATwBVR7pDV0ZDuNqT775xv/6CckICJIKr9z9X+xSuT8EZeAyQKgler0/WvueWaXqZOnbjrGMitX4qDkjC7J2pdoTUaPn2VGREX0waeLnnh5+smH75PwvM+c/fHzr6Dz2llA2JD2mmfNXPDDE+C9d3RdnRgLEINqdJuM0GkLBfksREgo7e2mIwsjNkp/6SvVBx5Wuf2OTi8DgBhH9UkhJX+Qa5jHRcTJjgCguJoOD13uSxAY8BDgAi0CoVIAIojZFcubn32y8a7bc6+8rNpbVUUleF4ARMl/t3jAL6ajAUCACEVyjfWpI78z7k8X62RabN4l3S9fRxfaKRYRFqUQkZ54bcZv/vng8y/OePrmX+w5dZKxVitVUtAlWQcSpTK4lWmZAVAram5rv+KOZ35/1UNbjB364k2/oojL7LNbk/qzHqu1K4W2syHPq3/lhUUnf1+vXE51Q8T4+VmCIvMQeGgxRi2EiKSkvS2XzdKkSVWHH1l7yOFlo8aiCAuz8QEJEVErCFAVkVqXeKsgdr0QSNelDEoeuowRk3TeqwKgNAC42wJAasjQ9GFHDjn0W81vvVF/0386Hr4fVq1KVGQkkRRjw6wVKIpJidGIgAACsyAm6obmbrph9pLF46+4JlU3hI2PSgOEPvi+LVnXO9d/a5kIlVYfzVt8/tUP3P7wWwwIVZVesTzGkpRkLUnMqQHMbEU8pXxjbn3sjfP/ef9Hs5cDYLoqE5KmfcbyeVbQce3MxldeYtmD9y778SlJ35fKCjG5AJ4Qaa/Cb8eoPSFEO2jp6Mi1t9PmW9R+53tDDj48UV3tyk2JKw6rgvEMFRP0LcoQWcoRUxOwMAX1vLublXlbWACcbSsiYi0RVU/bvnq7HVp/eNrK669pves2XLnCq6wCpQp90139HRgmKOZ5q43RQ4fyc0/POeLgMVdfnxk/wenoMKVxNf2KD76z+LVWq5pa/nrT45fe8lxLQztVZTSBaW/nWL2akpRkHYhg3sPhEb04ffb5/7jviRc/BM/z6ir9pnZreM3vMijyuVXQ+XRsEWBWXmLxbTeuPPuMpFaQToExjkfZQc66MYDmdXZgWyuNxuQaV8nYCbXfP2nIEUclK6uFWYxBR3MRQ+WASB6x07PmyccEA/xyULfbsjhfmHM8F7sCQt4wDcj9EQC0DooTImQmTKz4zZ9avnvC8qv+3n7Hrbq1laqqJGCsjlJqCrel/O4QjInkfKqtlQ+nf/rtQ8dcc0PVlC0D+J0EveyLHe2cO5b5ugdf/P01j8yZvQIr01RdztaKYHRs+Zw72kqyHkh+xQqwiCKcu3Tl76958Mb7Xs3mRFVWsLDN+cC8/rjZPp+ny+AUAyH5kdaLb7hu5RmnpRIJ8bwQ6ezgEwDdtDMKIoorRwKIpDQ3N+ZEMj88fZMHHt34+z9MZCrZmMB6dV5mCbV56CwJQoI9S/it0H8iwizGilbKMs/4ZAEzY95x3TVDPBKRMLvcJZUohaSAmY2pGL/JhAv+OvrO+2mfL+caG6Ezi1qHV0N3zJMubULIQ6oRxTdYVe0tmD//6CMa33+PtBZrpDe/eVdhZkQ69cIbv3f2NXOXtqi6CiBky+HYrDePQkk+9xKuNRZRRDPnL/nCsX/61y0v+l6SKtLWWmEJ84DXF/kcKugCv7PxSetFN1636mdnJMvSrBCYHaAicC53+W5AuB+W1VYarcnV19Pu+4y+6/6x5/0uPXQYO0CeVqEWDjwOkteZq1E67nwloR0KACxsmYlQa/XOx/O+dvrlJ/7+BqUUc3ffdFfBoKIKBr5yt2EoQqWEWaypnrrdJv++aejlV5vhI/36VURKiCJXXOSL6XLNIIcQQXwfKyoSK5bOP/aIxvffJe05HR0QjaxOR7ux+GRhPZSVUUpb35dC716IwCtJSda+xJbeqsb2FQ0dVFvJIGw5Tzcgsv5o6M+bgu4WFUwsuf2mVT87M1meFlLAzg1RPM4lIpiP5yF6WloaO5Opmj9dvMn1N1dtPU2MLyz5gtkR/iHQkX1rYZTYAgAILGKZEVAr9fH8JSf89rpdj/vL44+/S14CQr282tWC4U9o/UqQB06EpNgaAhhx6Dcn3Pto+ujjci3NkMuh84c4DDb2rKMBAVGMD5kKb8Wyhccf0/LxTNSe82i7tvWuoh0CJpnwgJmdhYIlfVySz0IiBh4BANBKoSK2JgA6rTdKOS6fKwXdJSpI2lv+6IMrzv5xKp1kUsAcYSawqO2MoYIjIkR/5Src4Yvj73p44+NOICKxBpV27uYQDwGrt5bjd3B4ZgkQfSJgDSOAVmrRisazL7ltx2MvuPb2FzuVhqqybr6HPgmGBrWz7IMcE1JIxMakhwwd/+dLh/zjOr+ymhsaAneHBLa0dDOl88FAJPENZarUkkXzTjimY/FC1Jqt7a1aFkZcfgDgkhzDNsr6dIYsyf+sRA4+xPV2RX5+FLQExbCDbBTlJRpff3np6T9IKsVKA3MQ9MPuSOdIRaGIgFaYy3W2t2ZOP3uTm++omLwZ+y4SGBjOUTphn5WzBN5pgCASB2AtA4jWqrG17ff/fnDHY//053893pIDVVMhIGDsGq4X557IowdduiMCGDPi6weOv+8R3Gtfv34lEQHlR6X7Qg0zGAUQxfepokLNnjn3B8fnmpvIjWpwIuzBjJY+vVWSknzWsp6e6T4/ChqCTJIgV7B13tyFP/x+oqNDEolAO0uYwh2TMCQYnHxQa2lp7SwvH37VdeN+cZ72EmwtauVgDWEAcADJFFEsUqxlZtFaZX3/ijue3uGYP/7fxXcva+xQtRVCaI2V4siNft8vtKYh72xHQqWsMZkx4yZed1PmtLOyzc1gGJQSEZGoukCXa4WMpohiDFXXwisvzz3jVMsm1MwS8UUV63dJSlKSAcrnAWYXoWcCW1ipzubmBaeeqBYvwqoqMSafhNfN7ZznhgBArW1DvWy51djLr66avDn7PqqQiSI4sPdX3wSERC6KaC0DgFbKt/a2R1/703WPvjdjHqTSqraarbHGxpzJg2ZmRqx8YXuAlGJrSemx556/dPKUFT8/K9HRgekysT4AxSp4RVcoSHIX4+shQ3L33r1g7Lhxv/yt+D5qLVHAcP0BKJWkJBu+fB4UdF4EQISJFvzyHH7tZV03JMhGCVPkCozDwIWc187+qhXel7869m9XpuqGsDGgo6yTvmJ+i7YIw9xsImKQR19+/3fXPvTi67NFeVRTCczW+AChsZlPHhxMwRi/KCIAKREG39/o0CNSo8ct+uFxaulSVVnJxgfn8ui2Q7gDhPN4sO8n6mpbrrhs6aQpGx1xVAiOjvzOJR1dkpIMjnxuXBwhbEOpJf/6e8ct//Vq68T4AZEbCLozevwnJNoEBFSUW7UqdeR3xv/z+mRtHRuDWkVJFH3PbO7aptCsZBFCeufjed/4yeX7n3r5C2/MwcpyKvPYWOaAxTmGh1grIYsIaOJahYigtfX9mh13GvPfO+2kSaapEXVCgB1tXtfhkqBEohsIFkmWl6381bkNb79JWovlONlISUpSkkGRDV9BB7FBEGvJ8+rffK3hwj8ka6oZEIhAKVHU4w8RKCJSuZUN5Sf/aPxfr9DJpLB19mCIBe4PUqMHcXDmK29+6qH73tCV5VSRZsvModYMks0jr8Ja1HJBbktYhIu0ZuNXTp6yyX/vgmnb24ZVmEgAUW+Dpggc0DCZ8rKti39+RmdrK2olsqaBzZKUpCRdZMN2cYScEiKuGvfy5YtPO9FraOTycrHtffETIGJnR0fFj84c86vfgGOrIAUQ0MmvoWqOTvzuKiqVhKoytpzPo4tM5q6EzWvZSxAlC4Kg0mxMeqOR46+75dPvfdu8+Lwqz7h88dUMvgh5Hr744sJzzx5/6ZUYOwmUfBwlKcmgyIatoKNEDncqX3LnLaA07PLFIJl7dYKkpC1bud9+o875BVojiIj5WnyD7koVZsgnLHVx13bn0VuLEqW0BD1Vio1J1Q0Zf/UNC376Y7toASSSwKvni2EEhdQ24936V14asvMubG3A1FFyQ5ekJIMhG7aCDmjoBYCI2Y767gl04ilhge3VKLnI0UtKgTWAhOSyiaKaVoPfXABYj4DAYT43CKBSwjY5pG7iv24MOUD63CkBMUZAsEQcWpKSDKps2Ao6D+0SQSSdSg/kIsyAFBl9ETfc2mz0eiEOmx3wSgkgkjAjgsqXi+2zBF77gcNdSlKSknSXDd7kyReyG7BaIAqJoR1NxP+UfkEMHOCIEJSAGfC11txrX5KSlCQuG7gFDQCho6Onek69SxDRcilx/7NVpTGqqptPeeyvrA2vfUlK8j8unwcFDWFJEhkQT2BEdvk/qp0BAvs5LBAwMCd5yXYuSUkGXT4nChr6Wcy0JEWkpGJLUpL1TNaRgo4s26C8dbHPYJjkNvBjdkn+ZyRkvI6V842nYSIWLY+7XklETRWUpAx74V5jBPKPnhrphd11fRaJ/yfqYP6PEeaqZGR1k7WooCOuSwkIsQGiIoEA3ROaw5lyevp/1R28vkoPdc8/m2ZAGMyVqNgu5LlGIKjk5QhIAt32mbe8oP2OvivQuRgUJQ4IFQNjRgKGL4C830lilIrrRXf61uUu/JFxP5pzrUWTlvdRbkAdXKuyFhS0FFDthMEn6ebclOJfBQhrmsY/sEbztSYUEautXT2oY9fLjaCPvuHVF9waqJs5Ylz6bJ4dCZWuQ6rnSzl26Y5EafoxQHxYwWvd7/qFSwRjNkq4r0gYQQmaH/Uj+iVf9d3h9CFfW2Et9SnfwKJXXw1ep+ucxPfSIp/sakFLwMz+uVPTefMCi/wtP1axPw+OgnZbfuy0WaBhWURYJP+IQUCaHwBxA5pQN6uOLILyuW7duzYAzs9+hwGDwt49Jy7nDZpgWIt/arAUOK7e7dPnexVearUjEtWp+ayeGYmtXQmffWYJmxS0KCrCLvn6AyFuUgI7WtYYk9nXNsf+H9V671JbzD0VHKvdjgUTHVUDhuC5iIoqRIIxc2jwJiZ+9u3ujww9FQHPWHwo8+XcooL1jv6QJdQBQf+Cc0M4n04lUL4WWlTj3rUE+rBO12vJew6cVpNiH8if/fK9HRwFjSGxQ3BQE7DCzIIAShEhCUVnzOB5d7X4REQREREKOZRbpA6YLQsQIYVz6piK+zdTQdqxAICwGOa+f5sIEcPcwu63jT1vlrmrmQQFxoK1rLXqT3FUDCvDuBZjLxcP3kZQYSJfT9kiIhDSq6K4Qpl9n+JwIqLCNYOlD3q1xPIHXveEWysIoBUpot6SYgKaK2EWa5nIbfmB3YACa/sUgKG5AbFKYtYyCwOgIkREIhIRhX3KRYiUJjNbFkSgcP8JapvJGmAkC2/k7Ct2qzpab9GFBZwqjZKcglIU7k+x9FRXhNI94CKietsYgwljFuuypYggUllra0/tqfbaIN8sPPMhBJtVj0YfIlJhutfAFXT4oIdPPIoIsDCzEKJSRCiIwCwrGps/nr/sk0UrFi5rWlLfvKyhtbW1I5ftNMYIgNY6kUhUVpaNqM2MHlYzfkTd5hNHTth4aMrzSAQALVsQIBXyFIdmax8shpDpDgJ96ikqWpCw2Nckbx93ZwAKP+F+UUT58n3FxKk2TX3kEUUQ8TQCoFKKsE+2UX9s9eDErLXqC7GRdDX9pNiIrIkUOe/F7EIQAWtZRJQiV0s925n7eP7S9+csmbN45aIVjfXNHe2dBsRqravKU8PrKsYOrZ40btg2m44ZWl3p7FfLTjNCVHhnLajpLpMQaGd2tXmJlCKSQKV2+mbJqsZ5S1YtWNawrL55ZVNbfVN7c0c2m80ZKyBCGlOJVGWZrqvMDK+t3Hho1fiNh0wcPawmU+4GgUWMZUIkCmcnPIOuSQfCsp2ASERdd2IRweB2YYwTC6peODvRWkYAIqUAGHjRysbpHy+YOX/5guWNy+qbW9o7jbWEKp3U1RXJjeqqNx5SPXHU0M032WhYTZVWChGYmVmckZRv0WDrzajYcvjG4ONtI49csFUjKiqiKwrOqbH6IP1W0BL9N7KWEUDAhOVCFIGIfLJw2esfzXvpnU/e/mD+R4tX1Te2cdYEVEHuJ9JWzv3GAsIAAB4mM6lNNqrdfsqYfXbcbLepE8ePHCYCImxYlCIKbYXVnrgjRcIsStGTr71/+xNvZMpSLKuxHJ0KOPVbX5oydqRlVligjyQqsAKAgD7bK+98asmqdu2pnggsrEh5KvH6zEWQ9LgXEiJ3EGeBhJ6ztOHX197v29Vb3UiY6/Qnblx3/Dd2x9VoagFAFlZEM+YsvOr2J71kYrWanYg62rMH773dvjttZS2TwkElrCt2go4AGojuQVWKEKm+pfWpNz565IUZL74755OlDX57J/jsmpjnugYGR7GdVsPqKnecNGq/Xbf82q5bjt1omIhYZme8RiptsB7GsGxOULgnPAKKUkojAqJl/mTRshmzF705c8G7Hy2YuWDlwlXNbW1ZyVmwDBw6OAjDRoUV5lkAARSplDestnLLsUN23Gb8XttttuOUcRXladepwPIKU47W3DONiItXrFrV1KaV6nI4NNZuPKy2tjLDEs1T3tdnWQBAKwUA0z9Z8NCL7z/28gdvz17W0NACOQMCQOhqaAAACLuKlqAQkjSkpnKrcUN3337Sfrtssd3kcZ7WzqCOHw0HUUvHnkMpqGQ0uIGl8MjpSnYsXdm4vLHZKxxVECCFHZ25TTYeVlleJrFTar8VNMYOQc57ZiwrQq2UiMxZvPzhF6bf89z0tz9cuGpVC7CAVpDwQCeoMhmtIIhBb7CgtAkySKexH3yy4oMPF//n7lfrhmX2mjbxmK/u9KWdt0wnE8xsmANDxEUfe9EWoYZmYQX0+oyFV/3jUV1XaYztvY+kFLd0HLDbtCljR4ac/vHbxF6iWJbLb3h81szlmE5K78o3nYSkJ9zz9AdKRiChFixpPv/ie/vylJEibun44s6Tj/vG7rj6wQBmUQSzF6647OrHsbxcxPZu1pNWXN88etiQfXfaKqhKvhZIReOuWffMCwBbS4haq9kLl11zzwu3PvnW3LnLxSIkPUh4lCmj0JUZG0IEAAZhgeUNHQ88N+OBp6afN6zq4L22Pu1be241cQwzW2ZFGEVKBuORl8CdgSIsxjlVSCmEtmznex8vfOzl6Y+/NfujT5bWr2oRw6AJtAdagZfEBATul6LHq/C4yCDWypIVrUsWNzz+/AcXJB+fOGH4YXttc8wBO08aM4KFLVtFKoBCrQHbl4g4j9xfb3zsrzc8VVtTZWz+edGKGhqb//Wb4449YBdrrdY62hTcfGmtjOUnXpt+6a1PP/Xa7LbGDtAKkklIJ6kshd16iaGjgVlWNnU8/fonT7/00Z+ufXS7LcefeNAuB++1XWVZyloLiArzwJdBWXxdLxGerAbd/SUgLKxILW9oPuDUv328oD6Z1BzTA5qopbVjt23G3Pv3n3T5br8UdFiWLgQDWWYE9LQy1j7+yvT/3P/ywy/NbFjZDFpD0lMVZUAgDCwMwlxcK0YurphqQ8S0R+UJEVzVau545J07Hn976pTRpxy++1H77VyWShprXSAxWo6re8YQAJIpDbWVqqZCeve9CiiFuYTnebqXz0Qri1AylRVQm9WpRJiL181fDYKAlllY8rZekTZjvnCUVrqucvVTD0iEOc9LVVb01b0NAAAJT0NdpU6XSW87BoCA0pQDSKQSfbt8/wW7/ioCzgGqtVq0ouHCGx+9/r5Xmla2QTpNmQxhEHZmGx2Foge/sCtaUaKMEOvb/X/d9sJND7924qFf/MUJXxtSWWGsVYQRJG/NHkmJ4G/GWEXkKWKEdz+ed/sTbz784gfvzlrM2Rx4HiQSWFGuAnPKgcpYBLo9GRF0sNsfEoSpFGLaWv7o0+W/+/Chy2555ugDtj/n2P3GDB9ijCVCIOjdblntbLjbZi3anNTnrDV5cnClwObE5/zMuaMICwuD1vqdWfN+deV9Dzz7nljCdNpVqWcOtEB4D8wbaF3nSyOVdxp++c1PXnpt5l+mPHHeiQccuvd2zGCZiQhCNTQ4ahR7+n2QdHQUnWYAgnMvueOt6QuptqK1w+RtPEQxtiLhXXj2UelE0lpWKh9f6buCjlz/AAjMIixaq5xv7nnu7b/e9PiL73wKOYDyFNVWQHDA40DrYjglPYNLYjYpuPi2OyuhAqoqF4R3Zi79/nk3Xnbbs78+6YCD9pjmJkwRhQHS1Z99WASsNZat7dlmDIw3BGO5zycdYxkMG2t713WBbycMpvTwmbyJulpL3wkpAmut7dOHC6bTWGOsQFCtqudmAwSluWBg+LzVNSW4SfSrm1nLfPXdz/zqqgeXLmzAynJVW8GW8w6iAjMqwtAXhKsAgAPfM6iaTNbYv1371EPPv3/lL7699/abG8uE4qKHa0LCJxJ5vMDTuqOz88EX3rv23hefePMTvyULCQ9TCZVOiYOdRPPkYExFQr55J2I3U1EEIKguhkApD8tTzb79+43P3vHkOxf+6KDvfG1XaxlYiNB5idfkcEAuEEnoPPfBm4SWKH/VMEhOSAz2gusf+s0/H25vzVFVBYFYK9ZyfrUXfAvDWYxQC+LmCywDAJankXD67OWHnXH1kQe8dfEZ3xxRV22sDd0d0XOyZiJdXkv+CV1jiXChvrGe1rc+9vI197yi6yot29iQiiJlWlsu/vnRUyeNMcYG2jnsWh8VdAjScIazZUWEmp5+88Pzrrr/+ddmAyksL1NlYKxwXq1g/gDTg0UQ01MhjC7yKASOG3BLmtIJKE++9/HSQ07/x6H7b//n0w8dO2KIb6xSREEQYVBpQgd+pZ6XjRTqkR6/HgL3V9uINV9FfQxbrhsRAABj2dNq/tIVp15w8/1PvgdlaVVXFat6XvCFUAFhVxqRvOJD90drLADousqPFzbu98NL/nzW4ad9c19jrTs29x7j7aXF8QBqNuff9tirf7vlmXdnLBBAKC9TNZXCzMy2+14reXMyiMsEKyevuRwcOBYwDT4fxBJYwBokUDUVy5tyx5573fPvzbns7COTWlsWF1teo40HAPLJI8GAcoR9Cz9imbVSyxuaT/rt9fc89g5WZlRVWVdzIXw2Iyxh4OaM7Oi8xzPY65zRTekEYurme19/c8b8G//wvR02H+9bq4lCr/f6m84WQ6OJp/Wni5ef/Zc7IZnMI75QQEBpZepbvnPwTicctIcxxjH9Yiw20hcFLcGZAkBYLLOn9ZJVjf93xd3/ufcVY5WqyAgKWzYcBUXjhoy7SOEW2nUhxD4QReTiClfchAGlkwjpOx5646U3P7nkZ988bO/tjbUQIpb6vByx55YMirrq4SJRLZW+NE9CqGhfujMoUmToBkI+NWAREGOs5+nHXn3/uPOuX7ykSdVUsmVrTBRsJyQkdIakOJdRUASXUCG5CDVLCC6OECvB82CMoXTCsnf6b29euar1Nz84yBll0eM00OrAQkQrmlpO++PNzW2iqioA2Fq2hvPBp8jMxqAyZPAAGxbLYhmsBQ6hqs6+VgSaHKgQUdiGiL3AIEX3WFpj0SNdU/mvG59dsqLx5t9/vyKdsiKuONAgUHQ7uAYUoWhw2vnTJSsOOePv77y/QA+pNsba0IWICBgCnNgycOhOQwAkICQVDII7XgQ3i5UZYmYAq4dUfryw8asnX3L7307ec9pmvjFaKYlySAdTYi6ONbBdQhtTBIBFEOTsi29bsLRJVVdaY0IjDIjItmW3mjziL2ceIQDOgRPCzwPRq7uT2z4xHEPxtL7vubd/dOFt8+etpOpyQgx3SwwDgBE6DfukO7CHt+JVPZzSd2oaWFVXLG7KHn7mP392wrzf/eBgAHSHrGDHXy/Sj9a4CX2KhKw/1u+aisPSeZ6++q6nT/njrT56qiZj/cgfgIRkfcudneD0dUpn0smyhKcVWSstnbn2tpzNGQCEVIJSGtx5ORjF/LGMLSOiqq367d/v8zz85QkHBufKcH/uz8xheFU01o4ZVnfcwbtd8t9nEMWYsLaZhA0QIUJCssyS86XTB2bwdKosMbymfEh1pqoynU4nEgkPEX2fOzs7GxtalzW0LqpvNa05AIBUkpKe238KfGWIImCM9YbVPPjYe8fqa2+/8AcYArzWALUusWNcka9bllRCz1u68ms/vOSDT1fqumrjm/x8EVnD0pYFY0ARpXRlOpFOeohkrW3Pmdasz20+WAalIJ2ghAanx/PawzUeTc5X5YmVHf6hP77i/st/tMvWm7opG4QjQo8TuyZPVgC1d84GT6urbn/qzsfeVTUVce3sIMhJRVf88ughVRVBj6Br1Lp3BR0hHAOwi2H+2aW3X/TvxzmRVLVuN4jAg4EqH8wAf8yCjh/5rbGYIEqW/+mKh2cvWHHtr46tKEubyLm+vujokvRFgqLpLJLw9B+uue8Xl95PFRkisDkLCKRIAKSt03bmKmvLt9xy3C6bj5u62ZgJY4YNqcokPKWImCWbM0tXNX30yaJXZsx99t05c+csZ0DKpAGEOfQshc98AFSorfrVpfdvNKTmhIN2943VAUa6f4snBBMFR9JjDtjpX/e82BZE1SQ6rZIiK8AdOc7mvDJv9Oia7SaN3n7y2C02HTVh4yE1FelMWbIsmXQHQQjNqPbOzpa27JL65nc+nPf46x89/easpQsaIJ2kVCKIucescgD0fV8Prb77obfOH3f3b085NHJoDhy3Lr2ZkVpRfXPrN8/8xwefrtRV5U47IwIqzdmcbW+trCmfutPEXbceP3Xy6AmjhtVUlHma3BaVM7axpX3eklUzZi1+ccanb30wf+WyZkhoKk+J5ZCTI5o1tMaolKpvt9/66dXPXHPWhJHDrGWiwHYcnMc9X0l5jfRztCdaZk+rGXMW/uyKe6G8jK2Nb5NKaVPf8OszDtl1m8nuTFDUsd6jgu7uZlrR2HL8+f9+4Il3dHWVIFjfAMXOj3lP/1rQjfkjP0ZgYYughlbd8cAbK+tbb73w+8Oqq6y15CCxpbqlG4wEz2LC0+dddfdvLnuAamtEjBhBQqWUae0A9qduPvboA3bYb+fNNxu3cU85hJNGD99tm01PPBQbW9qefuOjq+5+/tEXPgKlqCzJxoYLNf8AWQbIZE6/8JZtJ4/absoEY6zSFKqk/iwecUkdYKzdbvMJ+31xyp2PvauqMtZYRECtuNPY9iyl9NQpI7+221Z77zB5m4mjqisycb+KBBFEjvzNBFiWTJanUsNrq6ZOHH3cN3ZbvKL+7qff+tstT8+etZyqMkJSCAoCADTGUG3lH/796K7bTfrKzls5B85An4eC+GTBryIAYJhP+u11r02fp2srnXYmIrDMTS3jJtR97xtfOnjvaVPGjqSecz63mzz2kD2nAcC8pSsffXnGVXe/+PZ7czGdooQOTj8S6WiwVlS5t2Bx86m/u/GhK34Swg4Gvv3g6v40gIvGgxKEKCBnXHhLY0MnVabZcnRdrZVpbP3GPlufdez+zKyI8gDCQimeZuroMdyatpa1UnOXrtrv1EseeGqGHlJjhIU5RJsXsiV2uUU++yxEgK/2p+BrhWPmUlowyvcX6xs9pOqZVz4+8Md/X9nUopRi5gAIuK5ojEoyEIktE+ci+82/7v3N5Q9SbbWwL1bIU2LYNDTtsNWo2y468ZXrfnrmUV/ZfPwoADDGWmZjrbXsQM3Bj2XLbIytypQdtOe0hy85/f5LT95ifB03tCjl5dlI8zEOIY3tnfKD3/63LduptXJAsP76HrscGk8+dA9QCAKkleQs1zcPr0qefNRuT191+iv//tmvTzxw96mTqzLlxrpeMIuwM/IRiVARKSJFrr68y5YWa9lYM2JIzSnf3PfNG/7v7BO+pDpaxTBSwFiRN6VFAIFBn3fFfVk/p4jCVLIBPw5FwKAuK+2SGx+94+G3dW1FoJ214vbOjOLzT/vqWzf+3/997xubj9tY8vPlMDjC7OKm7CbQWmaWsSOGnHTIni9fe85V5x+9UWWSW7L5wpixLdP6VtWUP/z8+3/97yNEFMKxJDy/r6HkZ5Gxj/7ZHsVYS0QXXvfgYy9+pKrKYtpZkNB0+GNGVv/1nKMC6DqGkYpu21hRBS0Y2sXOdp6zePkBp/z1rekLdW3G+H4QCsSYi6pYeMldIe9REsmnEfbyIz0frKLgSXAUQEA0vtG1mVfennv4mVc2tLQppfgz4fTpYm30tgOtbiVFKbN92sw2cBEZUl1x/X3Pn/e3e1VNFVsjAsrT3NQ2LOP9/Vfffv6fZx++7w6e0r4xzEIISpEmpYgoSAoMfpQiReS8ycYyM39t16kvXPPT4w7d2TY2ElKAeZZo/QBbVpmy16fP/fN1D0CQDl6Qxd83QZcYQETMssf2m+0+bRPb2CpNbeNGVPzhzEPevPHnV/7smN2mTtKKfONbZgFxTVWEhEgUlUPv9nwiUvBJJSK+sZl06sLTv/nfC75foUQMB5XUI1IDRLasylKvvjPnlodfQURrC5L0+zM1PXyJbUV5+tNFy8+/6iGoyljDDo3Aje3TJo946l9nnPf9g6rKy4y1zBzNi1YBrU7AhkZIFMwXEbKIb6yn1IkH7/Hcdefsvv0429iqFQXZxZGOJmBjIVN+4b8fn7d0hac1i/Rke/Z16grnMcgQHpC+jzKcLYun9avvz/ndPx+Bykw+Shf8Q2T8K35+xISRw3xjAoqbHlQWFblLmLXKlrVSC5avOvjHf//gk1W6usz4PridPfA2YHHVHNi5AhJa2dEmX1SzxJUOhq4Y6WWQIt4nAUDjG6+64plXPv7e+f/O5vyAF2cdS+QKRFjdDrQ6Ey06hvflZ32FGa1uuAAAhDlZlrr6nmdOv/A2qMxYNohApG198/57bvHK9T/94WF7KUXGWEDQDlCJBCG4JFrQiPmMCUAgIK2IiHxjK8rT1553/Hmnfp1bWkJaogIUo7UGKir+euOz73+ywNPKwb37fbbFoBXM7Cl97Ne/UFubvPDMQ9644ec//+5XRw6pcVYkAGillcMuAATUjYB5cAZKtITy2ODweSFCTykAyfnm8H13/O8fv5cUIxxXTpFOZfCS/7zjOZ+NUhRP0+xfp7pHqgWAKOf7P7341my7UUoJs/KUbWj9xl5bPHX1mdtNHu8bCyBuC4WQUFDypkm8T8GNCNHTChF9YzYZOeyRv//4wH23NI3tWuvwo8GUiYhKeEuXtfz1P4+Dw/2FmK+BrMGiwF+UAWChoyCzCBBCS0f2tAtubu20ATWCG0YWpbU0Np9xzF4H7D7NN1aHB4We1ltXBR0lfDgChLZs51E/u/q9mUt0ddrkgahOgfZgOId+/fCwGJE8AylSWiutKMKvAgR7bIAlCvWyu0heCxcZyPBfCcIjw6rvvu/V2x56EQEdt846M6LRdQPzKrPoT7gDdSkp0W0AXb9EerkUFu4FzixZN50dJHFBC8wxXHLTi82CQKSQgAHbWn99+tfvv/iUcSOH+sYSktKEhVo4eh1/4QRinIdak6MiOP/EA8//wddscwsFz0Ns7QooTzW1dF50/SPB9fOcjH1eAOELpQgAvrnv9u/f8euzjvlKTUW5MY6RgxRRPk4TMDUF2YRR22N7ewEaD0PcqoS6LOf7X99t6sVnHAKtbaH9nQenshUoS77ywcJnX/+QEPPZPf3VYvnPB9afZcaKsktvfur+l2dBZbk1VnnaNrZ9bZ+tbr3opKqyMmOsdueCgG1DMEyZ6SLhdTGmt0ErZaxNeYkbf3/i7jtOMC3tpCncz4OmsGWoKL/p4dc/nrdEa3diFuyL1YtdelO0swKR67ufY+X6xMxE9Pt/PfDaO5/qTFkeSiSiNNnmtl233+Q3PzxIRBTFiHJ7kCIuDsn7cPFHf/jvC699omsqTM6EBqCEu2Kx6YyM31A1I6LSCojEN9zSYeubbUMLt7ZJNou5Tujs5LZ229hq61u4JSvGOtIvdEtNIpOzZ6sTEUS055lVLV//6g6H7fcFESYVAbHXugigNRZyVnr58a3kDHB8nUnRa4XLTEBEckb8Xi8b/kDOWmM2JJdHOJ/CgglPAAnR+lzG5j+/P/ZXJ3xDBJjFPerBJ+NaONiSCl4EF8bABAJEFFQKnSl93kkHHXfwLtzUEnNuBrucNQYy6bueeO+NDz7VWgWBuv75bUO8HaCIlJelRtTVuERQrYmQwk9gqI0Dt2OXxvd2+WD7CTYlrZS1/IPD9z5g7y25pSMwk2NXUgjSae97fjoA8IBWRgxB5f4bAQbhnZlLsqRAmDTZlo7tttz4+l8fl/IShtltUSFcLKTlLtbHaOuJ70GAqIiM5Uw69a9fHTt0SAXnbD5TE8AZ0dpTK1a23vToqwAgzHEVPwgiKP25WPhUo4BYZq3VYy9Pv/A/T2N1hTHGdcrt+tbY6srUpT8/Kp1MMkto0fVir8VQHHnLQ8BY62n91xsf/fddL6m6KuObXhkkIHZszL9BRAhoO43NtkKSRg4bss0mwzafMHLCmGEb1VWWJT23+2U7zdKG5tlzl8/4eNEbcxYvW9pofYayJCW9AHATWO2FyXX5wIiQVqa1fatJw68+79iyVNLxz7ki32sn0Sh0YoK4aPuw2syy4dlUypElFbmjgCjC+tZcW0cOKPSzF/2guzhL0qOhQzIiq22/EFFbJlFXkxpUhOPalrwjWFhIE+dsRvNtF3x//y9O9Y1VzpuBUWkN6NdUOlegs4UQhQhB4OKzvvXOzIVvz1pK6SS7KHeYgKUVtTa0X3vP89tvPn7goxjqT2cYa61c7lSUlBf2YeBTFLItIwY7Fv3yewc8/uqsnC10dLiPJRMvvDm7pSNbkU5xdDJYg3tH6TaY0CIMCGy4JpO4+lffqa3MhEhekDWGvmlFvjGbjhnx25O/evJvb6GqjA1UUOBqt2wh5d3++FvnfGf/slQyRkPRq+QzIqN/uv95ACkqAX0cs2ilGltaf/Ln2xxptkAeCkmkuKXl4p8es+2ksb4xWqvQG9zbQOUVdPQpl5v46vRPfnnFA1BVbm24CcDqtHN0qEEkRbY9C7nc+PHDvrb7F766yxY7bjGhprI8DBkWXCeqm7CqqfXFd2bf+9x7D77w/rLFDZhOq1TCssnb5iDxhRJ0u9MOqUxe94fjh9dWuRNWcEpZOyZ0PO/BRXvuufg026sqNdbWVJad/Psb/nXbS1ST4SJZy+GlRUgRt2enThn75BWnd+Ys9cHRLCBaUeQw3DDcHaGVigrZmArFt1500v5f2DrnG0+HjjkZeGfy3g9AQjBsqzNlF/7ksK/86PIw2TaviA1bKE/e8fS7//f9+pFDa5l5ALfF4LwePZVhOvLgTke4ronQWrvTlhMP3n2rWx9+i6rK40hbFoGk98GCFfOXrtxi/CiX6zggljssfClhojYQETe1/uKsQ7edPC4/azjwCpDREQFENClmPvbrX7z+gVdffm8elafYhqQxjpI3lfhgzorn35r5lV22ZhalBgVm1++9OSDEBgEJeCLPvezODz5epmoz1jcQwGxAedo2NB938M7HHbi7ca7nUJv1fr9QQbtxAREGIszmcmdcfHtbh1EVKevGpcfEdym0KEVpzZ3GtrRtu9XYU4/Y/cA9t62rrHCXdyAbt3vlGxZmKyrCuqqKr+8+9Rt7bLtwRf1Nj7x65e3Pzf10JVWUg0a2HAONRKWPQQTQdP7r/BOnbTouyAENlsjaBNqFq9BZChVlKVfWqIjXRwARrGWFpMNE+2B19ERoBwAiWqnyZDrthefE3ldJHnkQ2Vhrre+DIrEMN2HQvn/9Bd+La+fADh0E3RboM0XkG7Pvjpsfd8BO19zxgqquCGLrYcVsSugVy5rveuqdU4/Y27JgVK+1n8ewfIG9iH1iUMU5cNwTwAIK4Mj9d7z10bdAIv0V+PdIUWdz2/RZC5yCzg9Gf1VQ/kWeqxoJuSM3bavRpx65j4h4Kpi1NWRoiiUxCFtJJRKnHrHny+9c27USAohCtDn/kVc++MouW7MIRbTLfb59LBBb6JyWvqaQhycjAARj2NPqnmfe+MftL2J1uTU2oPYGIa1se3bKpOF/OesIiBZG3wKbFN3JtcjZDlfe/vRLb8xSlem8du7piBkLzAKAVto2tQ6vTFxx3rdfuObs47+xe21FxjcO9QSIpDR5SnlaaaWUIqVIa6W18hS5cj7GWmPsxkNqzjlm/9dvOPe8Hx1QrphbOig6EUDeOtZKSVPrn3988IF7bBfsS9H+vrbP+l3ry4WFFQvFHbVdiYA+bxj5z3F0iV4lBvteA5tznUowg4gAbW2XnXPEwXvv4BvjaVcXKdj51rwf8bXrgmlnHPOlippy69suDycCgFL3PvMuCxNFZ6SBnMP66lleY3EehV22njBmVC3nDFI0ruHYsrw/ewlAENTo99gVyagLo9hI4Ps//vY+yYRnrAUcNNMg0vKu9seBe06dsulIyeaQCvB0IgKJxMtvf9LRmdWKQpL5/sxXlK844KN2sB2IsHhazV++6icX3iFeomBXQxTmhIK/n/vtmoqMsQEboPRtp8zjLx3bhtZq/pKVF/7nSSiP4o/5ulFFehj9CVERmsbmA/fZ6pUbfvqDQ/dMep4LYWvlEFIQWsuBNgviiHm6LkREp7hZxBhbW1F+/okHvXTd2XvuNIHrmx2SMtTRoJUyq5pP+tauZxyznzGWguhEEEtaw1WyWgmhUPFYe9FIdRAICb7T37uE31zNfboFx9d3CclipKXj9GP2PvnwvcJsVwgqKA1WRyJefgBCNNZuPn7kUV/eDtqyBdFCB4JOJ196/9OZ85YoUsIiUahrPZOwQYiIxvLQmqrtNhsNnT6G2XVBvWMRQJqzpH5gdwnrpRS8F1ydkDs6N5888uC9p4lwMJJrIQDiW1ueSh2y5zaQzVGAkizw4bw3d9ncJasQByVRJT7CfblaAPuDMAx77t/umLuwQaUTIfbRbfokTS3nf3//vaZN8U1Il4oxtdCrUHCjMKQOAH+//emlixtU0gtd770AwQO3BiKSgG1t++UpB9zx51PGDB/iG4uE2jGaAPagzAreAsgf510dDQDwjd1yk9GP/f2MX556ALRnxYhjLFOeMk3te+4y+W9nHykOBICDywPSx4ncUIJy65kIBkcQjUfutyM4nAIET8bguwUkxHsAAsB3vvaFRLln43UbEECAPNXe1P7cGzPBPf9rYlutbYmcCQIAMG3TjcHa/K4W+duIFq9qhvypqp89KtDR+bRqIoJs5xH7Ts2k066Mr1ORgx6Td83+0k5TdNqzERglaoamjubs9FmLINBdoSd2AJ3rp8QYTlyuNf3rnuf+++Cb2jk3guES0so2tX19763POf6r7mNBgLfPsxDA/p33Rms1f+nKa+57BTJl1tUPlBBt2713ISYsAN11dFzxy2//5qSDXeFYT1F+O4coA6Xr+MffQsQ42aw7q2tF1loE/M1JB998wfEVGjhrEqmEbctuNq72P787PpVIsHDICb1hHO9LAgAQnpo6DQOEAHLpyZm/ZhJejwit5V222WSXrcdBWzasihreUQSQnnxjFgQH7fVaRce7NnmTjSDgRYqGVlxSyarmdt8aIpR+qAV34RjiAeJhA2AjicrUgXttCwBrJdwRmlqEBCLTpoyZMGZIcERwTQp3XGB556MFEDrIw7z7PkkvuSh9j6VaFq3Vx/OX/PzyeyCdculIQbSfkLO50SMqLv3ZUVHt9l79xUWEInyc26Cuf/ClVcuaVEKBcM8+nViVSBRE4raOy39x5A8O2dM3lggdA1lAP9qfo3eBdR0esVxOb843h++7w0OXn7pRTTq3qqWy3PvPn74/elidb/I4h5J23hAlVp9jXRyB3D5+0O5bgTVh0inm/5RMvDlzQWNLm1YkQaGeDSCnfvSIIZDUIXdSmIYKAIQdHX42UG29ALGKDVQP2ouIJNu546RRU8aOZBEVJe4O3rzlyRwQjOWKsrKpm24MLkk4RtcjIqDoo/nLoc8xt8K7hB0tYjn2eK0gDxKDWgquIM8ZF926cmWbSqg44E8Awfcv/emR4zYa6htDA8pwDixoYfC0amxtu+2RNyCP5+31gUEEAUWam1rP/f5+Jx+6px8mEQX41j6ASHobvtBlAYiE5Gnl+2bXbTZ98NJTRm2cufScb+4wZbzvG8cSWaKvK0kfxS3Jfb6wRbqy3LriemEQRUTA04uXNn44dzG4oh4A67kV7Wyx2sry8mQirLgWR16jMSYX0jQPLORZZAB9f+dtJya8AjfR2nn8gjDldpNHQYR9DFNaQACU+mRpvbEm9FD3s2c4wEFxXzGGieiymx598Jn3VWW5dYxIggCgtZLG5tOO2uOgvSL8Qgz832cJeuVgBs+9OfP92UspnWCOwY27tyystUlK2aa2g/bZ6rc/ONAaGyQRhRkcg1AdN8qeAgEErZW1duqk0W/d+H9H7LuDCGutNiTkb0l6lrWuAqNTM6GITBw1fNuJI6Czk1SYSQwCAkpRtr1zxqdLIYqzr7/KOewXQNJTaU8Bc3gWiP4swrFMwjVxcYTCAqDVTlPGQPCQF2LUBrlzQYx2yqajQJNIAWBBRECp+vq2FY2tQRnGPoxVvpnSi4ujZwkDpZY54ek3Z376f1c+CBUZyzby05Em09z5xe0m/v7UQ5glKJXi4t/9y7gCis/obY+/CQxR5KE4bXWY9Y6I3OmPHF5+8U+PIiQI66oF3AKDOEkueiQBUkoEhtRUJTwdbH/rb1mykqxHgiE8BBCM5VTCm7bVWMiZGB4/hNizvP/xAshnJPUj7rSuJXxAiZSndZHjboiJjr3TzxzmruOIYmy6IrXVpNFQ4J4abIlSLxABYMLIIYkyL1YiJ7yvwra2jvqmNuhj5E2KvlXYi16qigZ4N3DRkqyfO+vCW5vbDHkRI5UgIvu2JqMuPfeoTColwm4bk4gltD9CDkOrtVpW3/T8O3MgmQypS3tal0F/kAg6On5x4lfHjxiS8w0pchk1a8OYRQyQUohIYXpIkCpW0tAl6ZtE68St0B02GwsKY7XbA+8iaPXx3GUCQnkP9dqyokMC2pC7ph+ktAWSp/mD7goninitsX/f+RF9s/GQzOiNavNDuVaGJg9EB4CqirKhlWkwHPo4Qh8OUVs219DaDtC3IetbmKM3DzQCQFA4+/fXPPDMa7N1RRk7zg3XIiRo7bjwJ4dMmzTW942D/wa0E/0fLQIAR7H4+odz5y+ux6QOMe495w0KICG3Zbffetz3DtxDRKLc3LU3YQ6r57gNECmCqpS0c0n6K25r33rTUbo8IRGTOoRuaK3nLGvM5nIBS9zaCVuGyUXi6gaKuB/HZAEoAd9umIjUU09i15Me7yRh6kDfkqXiQb8unRcXths3tDrleRx5VNaKRHgHBJGKstTQqgxYE6RT5JlmIefbtvYc9MWC7uLq6dlOLvaXPK2ptay1eu7Njy687gmozBhrwnYCKbKtHUcfuPMJh+zpiJMcBcuAiYEpUnAvvD0LDFMUDS0ylRHiw82Tf+q39kp6nrEcUDINwILv54yFUNZgPyq5nkvST5Ew/AwTNh66UV0l+Ba7uDW1WtTQuryhNfh1jQ3ogFS0S/KnRGQF7v/usYJANwdqO+Io65o5WkR6gidif82m8KxeLPsBEcHajYdVExJ3ZTQZVIltEyxSnkpUV5QBR4DIWHvYdnTmoB+ZutG8RHfo7qTvavhFC0FYtFbNbW2nXnhzzpDj4YIwJ419W1uX/sUJ+8dHTNZgjEhEFBELv/rep6B1r7tQGP0j5Kw/adMRB++1rYgoR060riRAz663bsGSDEjWwXRGShgBmLm8LDlhWE2goCH25CvqaMsuWdEA/X/mC28XqGWIcebms0sBRFzOAJuwIpSxbCxbKywcpvADhMFyCF9CMS3dy1ES8/9ZA4smH1uTIUOqIRqdteMBimHgQAQ0qfLKNFiIz1XQH4b2nBngbXruZLxT+aSPEI58/hX3Tv9gsSp3DmGMyuyhVvX1bW/OmAsA4lBAfczp7kFIRJCwobntw/mrIOFJxO1dBL4RINVdHtFhe06tLC+zliOrdt1k1WGU1FCSkvRH4lmKLKJQjRlRA2xi+gsAABFtp1myshn6k/HVRUImg0gVowCwiCul6CgQHDsCEWmlPKW0UuELcpWuAIBZ3OeNzfOyFAEtYQ+vo8YU/UPvw1XsHi4SVFOZhnUBb4l85wIAqVQiqIAFXW4uxnD/Ltzj7aSogwNDh7i17Gn1wPNvX3LzM1iVscZAF3QzAgBdctMTOeMrpQSiQN4AR0u7uftw7tIVDS3gebFkpB62ZARrGNLe1/fcBiK08josX1KSkgxMIuhopLJGblQD3NX2RARhWdHQ6r7T/9s4k8kZXIFzwzIjgiLSRBHrYHNb+8qm1obm9raOzmzOGrYI4CmVTKjKTFl1Jl1XWZ4pS0ePJCJayyyswlJSkndHxJF1PWE5+pcrEEYqC4bANb28LBVeeZ0YZgIAUJZM5PsSd3SEZGSDd7uusy7hqUVrtbS+6cw/38kqSSRiY5932SuWMZN6fcbCu55441v7fcEaqxWF6OeBjJRmEQXw8bxlnPUpmWCb54Es+gVC4mxu6wnDp4wfCWF+dklKsv5LoSkjADByWHX3xxEBgbmhpR36q5/DZGoMmQ4ts8NWa4WIsKq5deb8ZW/OmPvexwtnLlq1aEVTW3NrW9bP5sRYDsxDJE9DOpkoS3kVlemNh1dvMrx2601HTd1s7KTRQ4fVVHpKAwAzWxvL4oMe3TEFELt+nbcLuCqi7wmApBKJdTVpeUkoDVJsugbqZAnResWGJPZGNJssQgA//dvtH3+6TNVUWBOyb2PcH+NA0uqKO549ZN/tNSm3x/WqU3sT7S46d+kq4Jhbt2f/BiqCnD91i9FV5WVBDYUBVWwsSUk+A8lnOyMADK+thAidlhcBgea2jv5eW8IHUUSYWQBcmduVTS2Pvjzj4Rfff/W9Tz9d0mA7fRAEReApIAKlwENM5L0QvojvS3Nn59L69lmzlz/jeKsT3ujhlVMnj9lru4n7f3GLSWNHeppE2BirAjxgH5/+visJ7OnzOl/acd1JQenzQqU6cCo7iV8umoICGhb3rrHsaXXjQy/+575XVbXTzt288IgAyJapPPX8m3Mfe/n9r+22rbEhg92ARLt+LlzR1OWc12OPBABx6qQxXXtXkpJsABIGuhEAoCKdBK2lSMYsNrdng5d9sT4kLKgjIgBsrVIKEd+ZNf/6e1+87el3Fy9aBUyQ8jCRVKlUnnU3pBKLZS0KAAABEqEmSCVciIctL1jZumDRe/c/8da51WW7bTXuqP12/PruU+uqMuAiVz1koCAMUJH2Zu4VmHJrnz4lwrsEv6+9G8buFEN2sIin1ZyFy8/6292QSjFzpJTd7AT8ohLmASGAwGU3P/O13bbtKSGzj6JdLtCylc0QEpUWGYB8AUBgEfBwszHDws8OIL+8JCVZLySdSCgiG3hbYzYUYlvWh74dnqMQHII4MjOt9UdzF19w/SO3Pv5WR3MnpFNYmSFXfoHZhmyUYfwozGCU2Juu9jIACEMUtk9oSnqCkPX58ZdmPf7Ch+MnPHrSQV845Zt7JT2NkVLp9gRHCBBXBKiPJ+3C0qk9xKM+K8l30O2uAzFRi1VUCfM8utxMBADOvPjWZctaVXWZDUrWCRKJb8AyJD2IVhECM2Mm9fhrMx9/9f0v7bSlMVZpyl+9X1wciOizbW3LApH0xVhgSaQTLo+oFBgsyWDJOnaSuUczmVBaYdxWAgiMoI5OV4pztc2O8Hnialob5j/8+/5djr3wujtfyYpWtZXoKbFsrUi8vLY4hQmKgtLjpIgISSEREgUZCgE0zt2BxVpmw0BAFSmqrvh0UcPPLrpzh2P++J+HXgKiYoMohY3thzNSuA/9X6cyyJuE5H0ahaNS6Nox1iqlrrz9yXsef5cC7QwAQoTSlt1u0xHf+sq20JZ1GYPRpZFQjPz9lmcgMnoj+7w/ohEh1+m3t7UXuLp7uggCWMlkkkOqKyCCcJQUdUkGLJ/pynG1fgCKWFHW+n1on0SHYZddNnvRsuN/fd3zL82iynJVk7HGWj+quRx6L4gQkUXEsBhr2QJLDECGAACEoAiIQBEQKCIADGqnCYCAY6XApKZ01Udz68+5+B4sS4GnQsbR/Ph2MYP7XiAmZmDCOvBjrIGIwGChOAoc2wBgmT2tp3+y8P8uvw8y5RIv8sBIKBeedsjGw6rufuqdTsOhVxoBgC1DJn3/Sx88/9ZHu03bzFpLRBGzS9+HUiOisZLzuTfmveiMBCjWZpLJslQCCklLSlKSgUiUEraO7+vUIFEBg6VDxwEAguU+2JpBDoq4CNLTb8/89k//uWR5qxpS5XJOwgdHglL3WlnL3J6FnAGPqirKaoZV1NSkayrK0572FFlhYyXb6bdmcy1tna1tna2tHc3tWZu1IAKeBi+BHpFCZhYWYbFsKe0BeMzcc0eDV/0+pmwAwf/+1CnHbjuxFPtzeM5wupRBfnrx7fVNWaosi9JSSCuub/rlyQfsvcMUADhin23/c9/rqqosrESMIKIIbdb8/fZnd5u2Wcx9Dv1a7HnITm/fKvBbidakI/KNkoouyQYsWNwm6cOpMHqKnXZ++KX3jjj76pacqKqM9f2AbTeM8pNW3Gltc0uyMr3TtAl77TBp+y3GjR1RN3JYVXV5mSIVZRJIWAoxm8s1tWZXNrUsr2+Zt6xhxqxF785c8PacJauWN1vLkE5S0kMA6+ox5ztUWKBOpLBT/VS5G8Cz7VzGg9vQYM+2hj1PX377kw8/P0PXVBq344qQIm7u2GPHSeee9HXfWE+rkw7f/dbH3+rksIgrIDiQZUXZ3c++98aHn24/ZXwA5+gn4E5DwEPU624ZzTrm8eslKckGLEFuAXddz6HtFEKjih/tAzWKYg17Wr00fdbhP/1nmyFV5lnfD7LLnHImAhFuaBs5svq7x+z+ra/suMUmGyOQyzJ0OdvWssS82c5zmEok0nXJ4bWVW07AEO8By+obX3nvk0df+fCJVz78ZN4KAE2ZFBCEPJzRw5/nqoCBqllEiXV/A1DVaywSO0mBsHienr1w2e+uehjKyoLSBOLYRLmyMnX5z49Kac8Ya6zdZetN9/viZvc+OZ0qy9lyWGhblKZci3/lbc9cc974gWlNDQBEqEmt7pOROYA53/q+hc8Aq16Sz618Jnu+sWxZQEVWZz5qlPDCStVFmhpoZ8ebs2hlw/G/uLYtK1Smg+SFoGaFkFbckUugnHrc3mcf+5URtdWOJ8lYCwBEUelkzBu3Me3KLAIizK51RLhRXc1Be2538F7br2pueezVD669+8VnX5/lG8BMOthsuljQ0N/QYPyLuP4SYXebkT5+Dgo2rx5QHO6VCACce/mdy1Y0q+pyG2aTKyLT3PzHM4/acuJo31ityenuUw7f695nZuSJAxEAka2F8tQdT7592pH7bDNpjLUcQaL76NQnENGakkmdd8kUz1IJB4KoJZtt7eiAiMikJCXZMMU3xhS4bqM4uZSlktDTthGW5HRr/8wLb5n5ab0uT7Jv85cRUFpxS3b88MrHr/zRX35yxNDqKmMssyCA1qSVq3gfqmgJa9w7dkpARCRCRaSIvPDTzGyZjbG1FZkjv7TTI5f/+OErTt1vt8nS0QEcBakKpd9sdt16m//3cyY9ojisZU/rWx59+fZH3qaqTKCdBZRWprHt6K/t8MPD9jLWunp7isgy77Pj5vvuNElaO1UMziECytPNDR1X3/UcRBtJfzIfiUWSCa+8LBXUUikq8QrHCtvbcyvrW4L7h3mQJSnJAKUHq2At31MAoDNnjNNrXc/xUpkO6Ca6PBRBsQgQy6wU3froK7c++paqKTe+DVRhGESyLR1TJ494+qozdp+2me9bBHE0SIHzACLuo4gL2r2A4HWMWzROx6EUKUUi4lsLAntvv/nDl/74ht8enVIIHDWgcHQH5HjegJ7qAfpw8mQjsYxtARBIed6iFfU/v+w+SCQjgA0psu25iePqLjjjmwAYVpNBABAWIjr5sN0A49FlB++xUFF24yNvzFqwRCsVsGhhX0O2JCIKKVOZBhsV0Cza36AbSMrv8OctXgkOeC+CJR1dkjWQz2bpCABANueDsdRlwSMAYGUmWfR7LoLHDIqoPdv5x+seBy8RUCdjYFeTQu7o3GRM3b2XnDJ25BDfN1oTorOOIcA/F1SwD3+H2OsuEt4/rCyEWiki9I0FgH122nJoZRoCZvbCLBXIbz99I7eONoOug7JeitvABtK8Hnk72Zal9QX/fnDu3JUqKNDqRk+0+H//+bdGDqkxxmJYMdXtmsJ84B7TdtlmgrR3ElGAt3OVtbVqWtV29Z3PQ8hWKn12PQSs2yOHVvdA5Re+Ec4sIYDPH8xbBiX3RkkGQ9YtiD5kswMAgKb2LHBXZLAAAGF1Jt3D9wMWJES848k33v1gPpUlmTnGRIRspMyj687/zphhdb6xUWljx9KPOCDnblDkSCTSwaFqAADft7YoUXCBsu5fwFBE+lPA8DOVAa2hrqFhEUBkEShLPv/B/KsfeAMqy0PODVBKS2Pr2d/Z58tf2No3Ju7HcGJYtFInH74bWBPLKRFAZGYoT9/wwKsLlq3ytGIWjOVm9y7kPrPx0MpYEeNiqTVxujzCt2cuhPV4Vy3J+i7dls461wQCAKsa24FjxJXuDyJANCxIxeryHXFnU6VIQG64/2UgFanL0LlB0NLxoyP32HXqpJxvnJvSKeSQOR0H9ui4oszorlOoOpXCPNSsG39d3kPSp3GO2kaxfWS91dTu1NK3RJVIvYXJQGHXMHYpAUWNrX7WhqB4AaXItrR+cftNfnnSga7CCQBEGtYNjVIkYg/bZ/utp4zkjixR/iwjIiqply5rCTzRceaV1UlAETVhozrQFAZDi5WLDW/HwpD03vlg3srGFq2VlLwbJVkTWcerJ/9kIAAsq2+GMO07/xEW9PTwusouf3JVmQWFRQjx/Y8XPDd9LpQl2XKEMXal7jceXfPjo/YBAK0oALHioBOmdzf7i/8lHyPsK0A20uYbzpPdx6GNZj+0mMMvS9ePUcHFrW8rM4lLf3pkOplkFgxTqCH/AgjAWkgnkycdsit05hApfnxhy1CW+Pf9ry5e2dAvtRnUJJw8drgqS7CNYUSKTA865CgmEh8vqH931nwA4IG5f0ryPy7Sw+u1ftvA7HE6a9mKeujCZ44ILMkyb+SwKuhiQYePtyuy/OirH+RaOikg3gxUICmCto5vfXnaiLoa37hqh+uI71F6+UPBM1p6XkPpYkFHWdgRrSACACiloL3jwtMOmbbZON9YF+aN5Zq4ooMoCI729Ztf3mnyJiNsNoeEEIYERUQlEwvnr7rxgZcBwLJg36xoQkIAmDR62MjaCjAmv88XMaIDeBERgG/ufvJtgKBiRCw9siQl6Y8M8Lg/8Lu5ZUqIArJoWSMoHTs4AiKAMcOqykbUVYXf6PplF75/8b05gIHZHB6R0Vqr0943dp8K+RyRdcVVI91euN8w4sDEQaiA+zkS6qriQmJOyYNhlFa2oeWI/bY9KYar68ZoEriAEdE3dkhVxfEH7QIdnYF3IvT8MzOUp66664WGlhZPq6gKTO8gbkIAZq7MlG81fhjkDFLIHtCTIDILpFP3Pj9j8YoGT2t2JXzCooolKcl6KxFTDRHmfLNgRQsoFediRkTw7fih1eWpFHeJH7oaViBKUX1L68dzlkHCCytbCwgQInSa8WOGbbvZGBEJbJ91tv1Ep14s/nYppt9NijsJogphRGg7/E3GDP3bOUeBw2QEIQDoKYriatx852u7jBxVbbM5jKX7iwilEnPmrrzhwVfBwTn6YEQThrwwO0/dBIyJjIFiAdxwFYuolLdw/opbHn8dXMp5Hry/lkd09dXnS7KBSb/5Y9ZAAg+kCAAub2hZsKoZPJIIEeUePssTxwxXRNIt9IQYhKPqm9oWrmgGrWKrUAABcv7m44dXlKVdEcJ1abD2SGUp8c+sz6R0a9b1QZG8kzr0cVj/r+ccPqKu2gTODSjqsArcHIiIaIwdUVd93Nd2hvYYB6mLLrOFpPfPO55taGn1Ak/0atYHRb6UL241EVPaOty+u2L3L4dp/swM6eSVtz0T3Il7yMkZPBGIsRWE9yop6c+BrEOFIVFNKgD4ZP7y1pYO0Cpf9DP4lzefNBIAivLZuSdqycqm5o4sKgpXZajcBSaOrHOfgzCrZV0PonT7U8jYNwC6pA1BBtij7rz8EHMMa62lseXMo/f6+u7b+sYqRSFQoviERtFCd3A69htfHDq80uaMK6rtHB3CosqSM2Ytuf3xNwCCYEbvSpOcJS8gO2wxbuLY4dBpEMJ0/qLfEwBEYVHp5Ow5Ky696QkAdNnoslbNaJGI0olFQqdP6dy24UsMq7SWJXiGnOZ9/9PF0mkICwCtVgATeusJriByN4LKUJY1tIBhpC7FkQBENhpSHb/j+uT3Kx76L/qxwhfruQxw08Huv0vwQ4pMS/su2044/+RvsATFeXuP9ka61mUPbTp6xFFf2Q7a2sMwcvgxFvASV9z5bEdnTikSAYHetBg5x4q1XFFW9uXtNoXO0Le9utOQtQwV5X++4cnXZnziedoVMAaXVDjoazL2DDMLOb7zsMD9586O3lAejMGRdQYDiuxn9+tbM+eDBIWlAtsHEXyz0ZDKzcaOAFdtqDvBugAAtLR0AEt3mDQgZjJhjvj6MZEC/X0ge4g2rqfi1NRASl51q6gSmM9IyDlbXeZdfu5RmXSamYkQVlcsLL5UXBj5xEP3yFSWWd/mv4bIzFSWfPf9hXc++aZTvL1n1pPELP3Dv7w9JDRL7+T9UaeEFLa2m9MvuKW5rV0rYrcdIObD4oMw/CKSd9u7wfpg7qL2bM6B9Yuh8zdcica80PLq87EkVDZrY5Nca31eJy0VEQxRT56itmzH9A8XBlnaEAaHCKHTTJwwfNTwOmu5F++EMSb/3OQtTgEoyJlY58U6B2Uo89pkHbZ8TVorMKCKKrFyrgVULIgIudzvTjtw283G+ca4cjbQB29VpKOJ0Bi7+YRRh31pKrR2KJf5HZ8c7f3jtmd9a5TCiEGp6GGL3HWJUER22mrCF7YYLe05wnxmVJFmBEEVYMu6Iv3K2/NOveAmABR25BxhRs4am0ahDz0we3xrlVIPvTx96rd+9887nwMAa2xgXK+LCOXal+BEAN370se+YYB7DFOgNgSRfqYgr+ntWADx00Urp89dDinNcW8eIhiz11YTIMLcSfFCFs7PWOThEOjM+YVvrMtZwD6/2dsIdfnvei/9KapS8LVu+SnOfO7onLbF6JMO3ZOZlQpZZ/t4C4ydxgB+cOgeXrkX8icFt2PLmEm8+N6c+55+i5DyOfrFFCZFSE1jOZVIHH3ATuD7kKeHLhoqzIfejbFUk7nhnlfPueRWrbUIMNsY3APCSGU/bLqIyiu4G4IVsZYTWt/2+KvfOusqI97PL7v3wRfe9Tzth1XCPg9+jmgjRzCxEBWubn1ICE4IN0WBQE1vEIOyDhsZujiefX1mZ3M7aQURRhRBmCGp9tppStfvdJOE5xXz6CIgNDW3R73CdRaTi890YcCwC7qjP6psg1g8/exTvHtY5HjjYDyV6USAqOv3HQLmACK0lnfccpODd98CWtoCOEfoDEBBELri9uctW+cP6ckTRZHOd/Qf39pvxwnjhnK2MwjAYdG4X8jLIQAAzFZVZy665vEz/3YrEiCSCeo7RKk6KNIXzFFIdi0F3zbWIqCn1cU3Pnrkz/7dJkqVeT7RCedd/94n8z2tLAvA54JUL4qDkmppy2ZzOYoq5vXcMwFBCYOokb/AcUZsAAcLDKzRKNSzVtsbXvz+l2aA1tGKA4eU6sxtPm7YVpuOgsCN2KOLryKThigNoeBTsnhVU+xua9nFEd46j6IqfL9rnuTnM0g4QDuEig6HAGDAGRp/+Po4jxg+xJEp8IPD91KJQpwbIlvGTPqp12c98coMosiILrLkCN1OIuLSYGorK045fHdo76B4tWMp3pIo38YyU3Xlxdc+fvQvr25ub9dKGWuZoz1BwuI+KJGlm/+vS0YMiW9DO1+CYomslVrR2HzUL/955kV3YCoNmmzOqKRe2th+zM+vXdXUqhU55wrA+hUw7I53Wp2dH3ZBU2Nj6/KG5qAqUs+fd9o5KC6NIiJRbR6BwPZeH+1ot/QRgbmjozNocJ7KePCjzBBUxRat1fTZ8195dw6kk0ENbBd8R4Ksv/dOk2srMr5vMIC2FneR11WVgcLgqYtIc0SA6NOljRAlMwSb59qSeN6Nop6tdYm/Wq2iWf9Wy2oEB8hmJ923NcyvzHAk+gsdj/xiSpG1ds/tp+y786bS1kEBKDPkgkEAkctueQpCawqKOdQoaGBYhhYAjjto1ymTNrYdnSEwu+eWQD4Zhtmq6oqb73tjj+MveuHdmZ7WGJQU4gKlHB7FY/+FgH8x8pUDWGZrmYi0Vve/+M7ux194872vq+oKiywsQGR9Vpny92YuPuH8a3PGBMMdPhRreUH0fQlAFJgCgISnY3/qMSQsIKDVkpUtH89fBg5TGG1eXTJ1JLSSAQXAWhYBpSiby3HMNlx/xqNLP918XXDdI53GD/DJEsY4BvlWzp4IkKf3PvN2U32bKkgzASuCSXXIHtuCoxCL6Ni7JuYhAAyrqUinPeHYw+sOcJ6evXBZY2ubIgo5ENbK2LmRQgRXMXbuouXLGttBU7ehC4oDuC/1U42tn+umeyMHiJeRIi8HxTyI6psEVzr18L3yagnzATwoTz/6ysxn3/xQqcCI7u4ECOApGDq2jbU1FZlffv+rkPMBY36T4g9NPvEQAKxlVVv53pwV+5x4yWkX3bRoRb3WymVkGWM4H94Ow6EhEbmEtQ1YxLfMIopIa5o+Z+GRv/jngadd+fHiJlVTYY0N0YYABNYYVZO55/H3zr3sDhViSNwJY72yoyHcdSrL0/G3ehQBpQiy8siLMyAIaoXjJoFjLDDQgsocYKwVFlcX6W83P3rVnU8pUpYHEtpel8KWqTz15PMf/unfDymlOLT9YXBnUEKIKounVWNr27/vfw3SqfyCFEEiaM/usNnYL243WUSIorSDruKWbnVF2Yjq8oAgP38fgYS3eGn9R3MXB5vB2nESRJu7taK1amhp/fGFt+SyPkVogfh9u6qv9evRGKQR6dvHsOivXdLfBkNCkAUhMsuXd9lqt20ncFuWiEIIgDv3kMnZK259KrpxeGLL9yfED4ahQkVkLR+5304H7bsNN7UpHQti9vjMYLQKrG+oLOF7icv+8+z2R//p7Etv/eDThSDoac9RE4iItdY31lg2xrgX1lgRRgBC9LRi5ldnzD7+t//e9dgLb3ngDSwro6RnTVTwTULqarC+pdrKv1z3+DV3P6OVMsbC+uTlcA6KoFAGwEZ1FdDluNmDWGbIJP/9wGsfzV3sedr3bXDkQPefwJD2DTscpacVAz/+6vt7/uAvZ5z/37HDhnzWXV+dhDqGmamq4vyrHrrnmTe0Vn4wyxLmia75VIZQeQQXGvnPA6/MmbOcUp7EgMxECL456qs7JJQylnu9GDLL0OqKcRvVgW8QY+nT7vjSknvx7TkQrMO1kBEQHcvC8PsP/nDjG9MX6kwZM3c/luXHMKro0lfZUHzQ0I9xLrLrYtFPrbmgo59lTmh90mF7QDQ7UeTAWsiU3/X8By+/N0spxcXYOWIA7xg2AwD/csbho4ZX2KyjTwrzWXvSfbH32bKIqJqy5S3+n69+cufvXPC1My7/2y1PvP3xvLZcJxIqRVopTytPa08rrUgphUTtudzbM+dedMMjX/rhX/c8/q//vuPlViZVVc7M7hwXurwxVl4ehBkyFT+64LbHXp3hedpYG8RR1wM1LQDxMP7YjYeCJuhLWjwLaVVf33ry729oaGlNeBoAmcVatpadaUxEWpHWqi2bve2J1/c/7dL9Tvn7sy/OGrbJxl+cNgkACCnaeD/rkegmkYHqHOg6+b3z/vvy9NkJN4MA0VJeEx0tMTcjsyQ8vby+6ZIbHoeypETuPhFEtJ3+qFF13/zSdgDQm2cPAQEsW0Vqy8mjQvZHiRljAonEXU+9baxRgbm0pr2IdyZI2Ap3aK3UWX+99dYH31A1FcaY4HNFfDIDQXEIrK0TwKCKC6L1c3gjAHTU18IXa97viJ3DFZA8dJ/tt918NLdnAza6UH0pAtPuX3H7sxCqLJECT3TeK4qAEqYqGmsnbDzsL+ccdsTZ12IiI2DDxmNPzSl0qoq1jBpVbUWbMY88+8EjT8/QFamxI6qmjBo2buO6IUMr6yorUp421ja2daxY0TRnwYoZ85fOXdTot3aA52FZSqWTTh/lYbISxmRi93UH0g6fjvvV9c9ec8bEUSOstUS0XoA6QmvHeYImjBpSVpFqN9xj3lMMfs7MlEk/++acLx534S++v/8+O2w+vKYyjBegb/z5y+o/+HTpU69++MBL78/+dAUI6apybmmfOnbo0JpKx8SGvbi6P2PJ6zURUZ6ub88ddtZV9192yrRJ43wT8joiOI78AWAhIlvE6TRmVkS//ed9c+avVLWVQSkjAHDFQJvajvvO3hsNqTHGKkVFEgiDNrvhRwDYZcuxl6mQTyl0hLJlSCdeen/eE69+sN8uW5sCDoc1GK0wFuwefGZhFq3Vr6669y//flJXV4baudgoFB30vg/ihiF961PBmij6Jez7xVbfpuAADb7hVMI7+bBdTzr/RixPCdsIYm8tQ6bsjqfePvOYeVMnjbXWkiKUfPhXF1wxQDuJIvKN/ea+O797wqI/XPmQrqsyxobQuh4wJwX2dQDyMsYHJFVZJoDG8icLGj75ZAVYAXDWPjsyEAACjaA1JDTVVDr1ZE3UptDYLFTNkb5mFp1KLF7ecuwvrrvnklOGVlcw939TXRsSNZwQQEYOqd50ZM27n6yglMcFBca6DSMigLBhzJR/OK/+6HOuHTmqbuKYYcOqywmguaNz0cqWpcvqV65qE2MhlaTyckQBYch17rzdJi6W4Oiy1kvt3DWwYdlQ2lvc0P71Uy699aKTd506yfcNKSKgoIpJEOcrTvRY7PJxgKEYaz2t73nmzStufwmqKqwJs29D83n0qJqTv7mnm6ngueoeHsSA/MjRd+w2bXLdkOpVLTnQUW5R4Fhkw3+58Ykv7bwlRlECGChxkkDUdwABFGOZkJTGcy699aJ/PaGqKoy1MXNk0Dbl9XTl9DRIg9PJguDemotbhIoQQA7/0vYX3fjk7IUNmNQx95ooRdnG3JW3PX3V/33XecWCzTiItXW9oriVqQhF5Lc/PPiob+xo6pu0lwiwcNgz83O+KrEDzjnfq1gWthZAMKGpMq1qy3VNharOqOpKVZ1RNRWqppwq0pjSAMIBPi+CILojWjckjXMfhO8Za1VN+UuvfvDoC+8CgGWWdcWTvpoJQtdQ8A1XlZdtt+V46PTz5XB6GsZwFxRrKaWpqmLxyrbnXvn4jofevO3BNx95Zsb0jxataMlBeVrVZDChma21bK1Awtt5q4nBNTYY+wcRiY2lVGJxU+6AUy678eGXPU+DiLVWovprLvrbG8QjUJOhIyCYfd9aT+u3Z849+bf/Zc/L+5wEAMDVQPnxt/ceOaTGN0HWQC8tdTaRtbzx0NqvbD8JOjoVUeTpBgS2ljLlT7z44bX3PqcUmSh2AhFisk/zEro0QpgTiIgYY7VS7Z3Z7/zy6ouuflxVV7CwIpLOMH2xS7m7QtjihrIg1pZ0sYi6xgjijqBBHCpEImO4piJz0sExIv8osZAtZMpue+zt9z9ZqHXoiQ6Ful8NBASR3J4scPV53/3yHluZlQ2e5wFAqHl7f/xDNe0OfoFaRxFhy9ZYY7sKWw5BS0FaV+huLnabfFgmCJcoT9tljacfv9/RX9uV2eroXPmZCoYDGsJV4IAvbAlac8FHimZDYCyMJswWE4oq0lRVTtXlVFlO6SRqJcLWsPNZIaIYf+PhldtsMhJcnoXAeureiMZGQCvMaBFjQSs2llJeC6tjzr3+tItuam7Paq0cFh4gwE860HIPEua5B7cQAfCNSWg9Y86Cg8+6YllTJyVCaJ0IgJAi29qx8zZjfnjE3iKgFeXjuj02OkTUAXzrqztABNWTPJenCENZ2RmX3PX6h3MSno5FPrEQLOk+XIB0j3Umb2EIiDNbPK3f/njuXidedOO9b+raKucNt82t39x3q7F1leBbDGHd0eVih/b+PQ8bzAaPsGbZQN1OsSKDdn4IPJxBYOPor35h3JjaoBoWBGtGBJSnGhvar77reQhd/+EcChW7ZpAEQ0giUJZM3vqnE7+8+xR/ZZP2dLDvCMaTTnocOffhMOoclo7v5QfyVWeCqu/dRiqClIR4FJ3w7Mrmow/a6c9nHSHCATpwvQFEu0a4SsBf/uIWk8cPkY7OoF5Z8UTN/NSGRwcUAbYc/rgTAuQHMyD68TefOGrk0FprbVj1Yb0YgV4Emf906tfHVZdB1pBSbCwooIr0Zf95arfjL7z/+bcRUStyxiNzHkSGBUo08Hw4ZeR2bWMtiCQ875m3Zu5/ymXzlrRQWZJNLAKJIAxJBReeeUTKS1i2fWowAACQQmbZf5etvrD1eG7vDBN5o1iJoKbWNvuts656b/aChKeNsRzGfmKZEBJtBkWLUDhNbiy7eGA25194/cN7Hn/xGx8sVTXlxvc95ZlVTUfsP+2iHx8uzg3d1W0ehwFCPwn719vdvVD6HoPtfmzto8tsTRvo4kZ2RF31d7+xM7R3Eqk87ALRWguZshseeW3OomVaKWGJbODi4SoM08MVobVclSm766+nHvqVbc2KRq10PnIdT8MoLpi3I/O540G6SrGfyKcBIR1v0f7mR1prbVY2ffOAba/59XGExAKEvQJOPjsxxmZSqWMP2AGyudUlagbTUDAy8ROYQ47HTvyICIZ332os5IG/6/kDJkjot3fuvt2Ua35zLPi+CKIid8xStZUzPl1x0OlXHXTG5U+98aEgeFoTIbMYY4N9Kv9ksgBYESti2TqL29O6ozN3/r/u++oply6o76DyNBsLATMZAICnPWls+dn39ttt28kBaVkfgmgSHolYWCt99rFfBmswRoHilq4wUzoxZ2nzl37wN0cagwCu5SyRGyayhSJge3CwYBHLbA0joqeVZb7rydd3+96FP/3zXa2MVJ60vtFJz69v3mv3Kf/57QlaKd+aPIqraBcGiTZ5PZQ1bmKXWOpgdjlM/EaXBvi9A3cbOqLS5vwYwEtAQCVUw/Lmf971HETVsACgFx7VUEejUsiW017i5gtP+tFx+5jGFmAhRTGXdKhtIfrp8aKh0Qc9W9C9DmNkhgcQFmVWNH33sC/c8IcTFZGIEAZOlfXAAR3NUDjWLlHzwN3GjK21Wb9PLYyPDBS+LsxwZRZIql2mTQKI/jJ4J7W1J4rqm9r23nHzy889TFpbUJQbFmsspROQKX/gmQ/2Pflv+5588dX3PffpkpVEqBUpIkWoCAmDJUUIiogQFZFWalVzyz/veWan717w60se6CSNCcXGQpQMLaK18lc1fX2frX9xwteY2RHR9AUxEmz9EmQMHLzXdod9ZaptatEuYyCI5oHjW6Dy1PJm/8Af/+PkP90wb9kql7dFiMzsG2usdam2ltlYa617kwFAESkirdXyxubr7n9+z5P+fOhZV789c6mqrRQCNlYnPLOqeY8vTLztjycmPM9YS4ErvNt5faAzg2t6gXUnA2xi10SVtfO8BJAgcdWwRo0Y8p39d4DWDuUyvwNHrispW3bjA68uWlHvacUScJ3p3q4clgdShFaEgC4966jNN9n4rL/c0daWU5kytiZezy0GXlkbXQ1pKV2BTk1ixDY3nXPyfn849RAEEhCFQTRl/fK95tNVwNUrO/f4/U8+70ZVWxWgo3CgkfcoFEQonf7Y4dWbjwsd0BuOaE0AcMrh+3Z0mrMvvF1VVjGxsDALgKHKtLA88+rsZ17+qG5Y1XZbjNl1y3HTthg/YeMhleXpdNLztAaAnG/as7kVjS3TP17w1Buznn7j4wXzV0EiQbXlbDlWvUoARHvaNLVP23Ljf/36u55SzOzKVoj0Fc8Xn7GLfnL4G9PnzF3RQalEkCoS4+PFBDEkr7rp+TueeOewfbY94svbbT1xVG1lRqnuzm6XhILZXG7Jqqa3Ppz74IszHn9t5sL5K0AnsDKDLNb3gVBrz6xs3G+vLW7640lVZWUAoIiK6Oagv6HV0wP8qifJxxnXexmYDzpMJOoBTDW47QuIohEAjjto12vve7UhZ5GCtFlHo6OS3qLFTdfe/fwvTzyQmQkVYK8KOmioiAAqBAYwxp588B5f2GLsyX+4+ZU3P6GKMvJ0mIEdD0Xg6iky+zGQoWMawCGtyNO2pb08pf7862NOPnhPay2gEGHo1F0P9ZNLyQjMrhMO2v2xl2bc9di7ujZjXMGFPPVwn9seG2BCsjl/681Gj6irspaJNqTqoM5Jk/PNWUfvR4RnXngHlZWhJpf57f6lyjQCrmrzH3v2g8eeeh80pcuTdRXpqkwq4Xko0JHLNbR2rmhqsx05EIBUgqozIsyGI9e0u5P2PNPUvunY2tv/csqw6kpjrVKE0ltssOhkQuhYHDdi6NXnf/eAH12Rs5a0Yhul87k6oQIAVFOxqi131c3PXXXnCxNH1W2+ycZbbLLRyCGVdVWZsmSCRbK5XGNLx/LGltmLV83+dNnH81fUr2oFZkgnqaoShNlYASClkMHUN37nsF2u/NnR6WTCWEukIzBfl4XhiNC6pGRsIOuizzIAp3pxuHPkbVxtdK3f4pCShOgbu8WEUYfvu80/b3uJaiqsMZGfky1DWepf975y4qF7DK+rZmZEXJ2CDuHWbpMiRcbabSaNfeqqMy+5+bE/XfdEU30LVpaTImtsCNWEIIoYNQxCVNxqexHlwsQjfLHyyKQUWLarWqZtPfofvzhqh803yRlfk8LQL943E2hdhw/z6WYIwKAIL/v5tz+eu+z9OSt0ZdrkDDh2nngafiHLQ8GbsSIg+Ugs885bjwMAdn6eAXSw+6Jch48yIRpjzzjqK7VV5Sf/9ubODlDlKev7btScmkZFWFmGiMDQYXnhqvaFK1sDBz0hkEKtdVVSQJiZo4zEMPsOALSnTX3rFpOG3/XXUyaMHGqsUUp1B9mvZjbz+TOiFfnG7rvjljf87rvHnvvvLINKautHWTBBPJ6NQaWoOiMCsxc3zZ678r7H3gJEUAQU+u5YgjJaWkPCo4pyx4UUdARAedq2dibB/v6cw875zn7ukOGCGRj5r4OpjLJqus5n/9dFj0kd64v0N1Gz91GQ0Ek1qBJloLnT7UmH73n9Q693RiHrAM4hKunNn7/8xodfOvPor1pmna8K3vvVwwceARSRsZzw9M+++7XXbvjZsQfvnLLGNrQ6yuaADcTZvBFUTjCvWYrw00Y7fwjOwxBDHZmWIkoRIXFTexrMz07+yjP/PGuHzTfxjfWCoCWE9F59GdyeYSQDSort2xQFKYWEaAyPrKu+4+IfTti42jS26WQi78Qv7HXM0R++CRBrauCLByJMqF22GA8BgWz/n8XVjcZgjkPRWUJQinxjvnvAro9dedrEkdW2vlmRVpFrFSCCaVq2QIIJhakEpZNUlsSUhx4JgMNwSh4AA+HWTkRkVjZ8ebdJj/7jJ5NGDzfWKlKQ9xv3q7MYoW+0ImPsN7+0460Xn1ibJNvSoRI6wFFBfiULiLWW2WJSq8oyVVNB1RnMlGFZGWbKMFNOVRlVW6mqM5RJoSZmG3VEaYVIdlXTpmOqH7zy1J9+Z38XCCWVb3M8S2yQTugSgFwLDc+CD3wmgsW3n750KPi3Kxwd85fFAG4/2G0O6ZMIjbXTJo87aI+tobVdKZV3liEyM6TTV935UlNrm6eVSN+LLcaizC7QYazddNTw6379veeuPevog3ZKK+vXN3FnznGEBnRgUeQwrn2KD7bk01sk/K4IISitEdA2tWFn54Ff2uq5a8/64ymHlqeT1nKQEBxvXB8md7U9XdsuEqXIWDt5zIiH/n7adluPMcsbiVRQ/be7Usb8aBQobhBCUEqhFbu8fsuJI3bYchMAwL6hEWKjj6v5RD/HLxSJ4waiBKjCgc5fz82gY7zafdvJz11z1rcP2sm2tNqOHGmtFMU/5y7vAMUswizCoYbKb2Yh2JmIlOLWLGWz5/7ggHsvPnVkXbWxVgW25wCdcWHUFt2E+tZ8Y7dtn772zJ2njrYrm9Cy0grzcxfZHyAOoWGZLQdZ25aFOdh73PshtEMpRQi2sS1h/dO+u8+r1/1snx228I0lQnTWT9Qe6jqXWPAfiHb0vnYwhgeMvRd/hNeRWV0AXyoAEmD45kAvW9zXsXZy3CLsGyAA/ODw3SiluzBOiohKJ2bNWf7fh14BQGu5H9VwwyhK8FoRMYuxdrvNxt7wm++9+Z+f//wHB2w2to7b2019C3f4IKiUUoqQYg9MAaIu/jpvJyECKXJZA5w1tr45BfagL0997Mof3fPnU7adNNZlZwWRdxfK7If7cPUiQaGZwZ4gzKdwKnI6eqMnr/zJ6cfthZ1Zbm5HQOVOIV0s6HjeASIp0kohEHf6tr65Kk1nnrjfY1ecXp5KMjP2jb8ixJyvLiM+iChQ3hrts/YPxzty3GAXqJcUonbcbuF2r+G1VTf+5nt3XXLy1pOGc0OLbetEIqVClRddQPLfDPVPMGhIoBQhErd3clPzLlNHP3HV6b8/5VCtNbME2jnM3h7ghEI+rqhJGWu33mT0U/8484KfHjo049lVzdJpSJFSUVWjUFP3dIYL14lSREqJYdvYSp25g7+87fPXnnHJmUdWlpcZa92jATFDWfLWYRcVHd8D+5moIhh8J37kL9BoMqDD2ppJkbutyaMaLtCgQ9h9lQ5uwxFcNSy7x7QpB+w0WVrbVVB3zX0IRQASiX/e+XxzW4en1ep90AX3CHOo3L9KoQhaZgSYPHajP/zw4J8e+5WX35197zPvPf3O7LkLVnS2dAIieB4kPCBCcjZHzKcRtRwRJMDAim8k57NlL52YuMnQb+y61ZFf2WHLTUa5CJuAONUcZKr0Oaeb2YLxtfHBrIYomRTmcrko9D+oCzDYThzhiWWbSaf+dta3v/nlHf503WNPvDKzo74FEglIJEAhUugLcLsbC1gWY8X32Tc65U3cZPi39p56zNd3mbDRsIDAP3+o73FIQqQZg5/zjMeroY0WBWL9HFgf+iUS+xcArAE/p43GGJknEvrGB7H5jztoEZEVMZYP3mPavjtMuf2JN6684/m3P1hgcwaSSUhoFXgmwjuEqV+IAEACwMySNTabowRtt+WY04/Y47B9d0glEsZaCtM68wbNGk1nnpPLVdBIKO+cY/Y/4ss7XHP38/995I0581aABUh6kPAwT3AHYTp6YIgDgNP2zCI5aztzIDx0RPUB+0898ZDdd9xigtvREVEFLO1YGEoR9HNojEaRaN0isjHod4aUTn1axvnPWAu+7xkfY8+LZjJ+DmyfknoGTZxHwhrwuz6/SpGNPar9EhYm04niFaRWC+WMD8ashX0nSlMNDkinHLn3gy+8T7lcoU9FPE+9O33ubY++esIhew6c8k3yR0oRABZh5uDRQWxp75g+e+GbM+e/Pn3uB3OWfLK8oaklK1kDlkOrh/I7cLRYNXrpxPC6ym3GDdt56sTdpk7cYfMxZam0CDBbCfPxgiDN6vAa0WAICCHNX7pq1sJlCa15dQmQiGCt3WbTMbWV5S7mNtjIPZGImptFJNhy3po5//7n3nnqtY/en7e8sTnLOROazwgEqJSX9obXlE8ZVbfT1hP23G7yjptPyJSl3JGZEKP8/J6GxQ2I69Gq5pb3Zi1yXIi9t5UQc8ZMHDV07IihzIyEfbHQ84MvQkTvzZq/sqlNx4qYuDkU5m03HVNVUR5buWF6rIBldme1zpz//Duz7n3mnafe+HjWwlV+mw+OEowIKBbhsALOPZ1OTNiobt/tJx28z7Z7bDsplUxYy87dES2ewFQYjGl1ONeQ1QgssyJCxFVNbU+/+eFDz7/3/Luffrqs0bZlwQqgAoUunypoNgsIgxVQQKnEyGFVO08Z/ZUvbvmlnTYfO6JOxCEOAxx9GMsIeyCCiJ2duTdnzssZJso/0Y6MOJnQ200em/C8gECq12cmWCEsRDhr/tL5yxsSWnEsdk2Ivm+njN9ooyHVfV8MAx/YWHtmzlu8cEWTV1gHx4GLNx8/ckRdVfioSu/rX4I0aWxsbX3n44Xh6SZ/QWttdSY9ddLY+F0GtUdBKpmx/hsfzuv0maiAHtk9ccNrKrfcZNSacHLmsT1xy9ql46oYrYFv/OUNLfOWrFqwpH7JqqYVLW2NrZ2tHTnfNyKS0DqdSgypKhtaVb7xsKqJo4aNGzW0Kp12w8IibBkRomi1RLDCPoxawYbVn+cxSMHO924wFXQQoogNndNEzrhj5mUNLZ8uXLZkZVNjW9Za1kpXlCVqK8s2Glo1alhtRVnafZFFrLVE6DI2JGpuj6MBoZsqGpA+PFrxQezPaBQMPsTw4PFBjpUeDyZI4msKnYc5GpyOXO7DOYvfnbngo4XLP13S0NDY0tHWaSwrRYmkV1VVNmZ47Wajh06dPGrrTUdn0mkQYQFmJkIsqNoXWtGDNqfBknHtZhFht+8iorRlsx9+umT6rEWfLFk1f1nT8pVNbW1Z3zeAkNA6lU7U1GTGDq/eZOSQLSeM3GLTjavLy9woOO7vKMsxb9WFjr24Ou6um7qu/9X5c2R111ztBwZXuiwh6bb4BtCeLvq96OfXagf7MCn552BQSJMDGF7MhgAAYGaXeB2xM6+2n2GlNRc4CaynwrNGsE77vGMHXUVn4/c5E0EAVMQ8ubbWXzhu4ai5EdOESNTTunEaPD84+VFZrXkU9AtCO5q5r3iiAMbmkOb98QpEq9BaG+6sBd8OxzmwCsNLS2i0BStZxBXfE+eGji9u3xjLQgRaKUUq/idrWQDizFkBhdTaim5J3tYMooNiWRBBxQjK3QtjrbEWEDWR1ipfpw0BBS0zCxNRrKx7YJx0meHoabfM3fsVOT37rmmiRrrV2P3zzvx01bX6usev0Ziurj0iRBQcHfrYwSAFLyK96LomEcEZBGvn8c+raA55h4t1Cil+IBqkccw/COEwuD5HJF6BIYd5kwAlZI1Bl9EV6QCnc9zlIYL7DWS8Bva1orzAgyvOinSDEu5xIdFkbG6CMUGIBidvrUGfVHOsU9CvzxcMIvR7HONcE/39TF5PF2xjwToKVwtGSp3Fua8kOFTkRy/C8q8L4EFcEUd7g9sRI6xV0PDggYiYkpDcYSG6Vujv6SmJKW7r9dKg4FKDN2XrEh7dh/as9vRYZMT6csW1aZyt/gwnIv8PMGBZ/l06jO0AAAAASUVORK5CYII=",
  "goldman_sachs.png": "data:image/png;base64,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",
  "jpmorgan.png": "data:image/png;base64,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",
  "morgan_stanley.png": "data:image/png;base64,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",
  "natixis.png": "data:image/png;base64,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UVVVXllZcd5557799rs33HBDQzEkHgghm83GMMyNN97Ut2/fYDBoyiP0SJBlefHixT//vDIvL/eDDz7o2rXbuHHjNE1jWbZhgxAim80OAMjKyrzhhqGiaInnAoQQluWqqio//vij3bt3IyT873//69KlS5cuXSjFzCHTPodCoSuuGLhixXfp6WnRaPTss/8zZMgNvXufkZKSyjBMWVnZL7+sevvtt9auXefxuKZNm7J69S+LF3/u8/nozxudGgghz/Msy2Rnt7755mF2u51yT/otxoTnuZ07d7z//ryKirLkZP/48fd263Zqfn5+U/Wkzd5+8803X375pcViy8zMateu7Y8//lhUVPTUU08tWPBBvCBsgo738cefqqqqYlmOCqMsy1ZXV8+a9WZpaamu63fddVerVgW6rpnnosVicbvddOIanUqMscPheOCBB5KTA5qmI9RInzEmhDRJopbDlCe2bNny4YcfOJ2urKysgoI2P/74Y3Fx0SuvvDp3bq+WcGez8zabTZKk1q3zX399GsuykydPrrvY/amys1hECBlBEJrvP4TIarVhbKiq+vzzL7Rq1cpUvwAACAG1tTWff/7FsmXfyDK3b9+e/v37L1786amnntZk4XCSwD8C9BCORCLnnNPf4bBnZqZ17dp1/fr1hBDDMKi9y3xM1/VXX33F6/W6XM7u3U9buXIlfSy+NSpg/vLLKofDkZOT5fE4Lrvs0sLCwvgnzWYPHNh/ySUXBwJJubnZaWmBrVu31WuQgn4SCoU6dOiQlORLTU2ZN+99QoiqqvXGQp8cNWpERkaa3W67/PIrzIHEP0k7uXjxp+npAavV0q9fv1gs1vDV9L8VFRVdu3bOz89FCLzwwguEEEVR4l+qaRohZOzYsaLItWqV5/G4p017Tdd1c5jxQ54/f15OTla7dgWCwI0aNYp2pl73TIIvWbJEFAWfzzN48NXRaNT8vN5j27ZtS09Pa9++LQBg4cKFjQ7Z7AMd+9lnnx0IJLlc9qefftowjPT09ORkf3p6+ooVKwghmqY1/Hmjk1JeXnHeeef4/V6n07lmzZpGp68eKLlWrFhRUJDv8bjat29XVlbakh8eDeLHPnDgwMzMNLvd+uSTTxJCWrdunZWV7na7li5danavqUbot6eccorf723VKu+007r5fN68vBye5x5//HFzNulYtm3blp4e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pZddcsklNCgGQlhUVLR8+beffPLJunXrNE2nfsccxzEM7NCh/cUXX9qvX7+cnBxBEFRVfeKJxysqKmjGJfoWUhdPDCHkOM4UGFmWCYXC+fmtaDrW1157bf36dU6nKxIJX3311eeee54ZM1aPPjTJw8cff7x06RKbzVZbW9unT5+hQ2+ij+3evfu5556l+ZLqQsgaGSyEQFU1URQnTHggNTUFALBly5Zp06Z5PJ5QKJiSkjp+/Hi6t+PPJ9r5L774YunSpQzDtG7d6rbbxlRVVb344ovl5aW0Sil9FyGExptQWsULvLFYLD0947HHHmsoQe/cufOGG4ZkZWXpus7zfENmQbttt9t79+594YUXpaenm+yv4TKYMWP6qlWrRFHUNC09Pf2BB/5bL98pORhepE6a9EIwGGRZLhgMjhkzpm3bto0Sf+/evS+88ILX643Fov/97/+8Xi99bNq019avX19RUXH55ZffeONQmj4b1sUZ0YS6X3/99SeffKxpWjgcTk1NGzNmDC0qNm3atNWrf2VZ5pJLBlx99SDqxb9x48aZM2e63a5QKJSZmTlu3N2wLkC33lx88skny5Z9ixDs0qXLTTfd1HCdhMPh5cuXL1gw/4cffgAA0EK6LMuxLHPBBRcMHXpT165dOY6jEUnN8WiSwD8C1PW9pqb29NN7iSKHEFq69EtSlzTgvvvugxAEAknJyX4IwfPPv0DishPEp5I47bRufr8XAPDSSy8RQubMmQsASEsLcBz75ZdfksbybJgpO268cYgock6nvXPnzjQpR79+Z4kin5aWIopCM6cLXZpPPPEEIYQG3T377ES/32e1ioLAAQAsFt7hsNntVo5jAAAWi2CxcE6n/YorLl++fHksFsvIyACHil0sy/j93uRkf3KyPz4emMaC9+vXt6ammhBy001DAQAZGakOhzUQSF69ejWp4+zx6TXoJz/88L3X63E6benpqQCA22+/3STIb79t8vv9brczOdnPcWzzIw0EAjQXJSFkxYoVAIDkZJ/f74EQ3H777TSdqUlnc3aefPJJAADLouuuu44Qsnfvno4dac1GFN9+SkpycrLf5/PEf07/3bVrV9IgKwUhZP369QCAzMz09PRUnucQAgwD6/1BCNrtVrvdmpycTJdWvUSatMN79uxJTU11uRwul8Pn8yAE6bKJJ6aZOLRXr55Wq+j3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",
  "nomura.png": "data:image/png;base64,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",
  "ubs.png": "data:image/png;base64,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",
};
const NavPill = ({ active, onNav }) => (
  <div style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.05)", borderRadius: 12, padding: 4 }}>
    {[["home", "Home"], ["fx", "FX"], ["rates", "Rates"], ["equity", "Equity"], ["hybrid", "Hybrid"]].map(([id, lab]) => {
      const on = active === id;
      return (
        <div key={id} onClick={() => onNav(id)} className="sp-btn"
          style={{ position: "relative", padding: "8px 16px", borderRadius: 8, cursor: "pointer",
            fontSize: 14, fontWeight: 500, fontFamily: sans, userSelect: "none", whiteSpace: "nowrap",
            color: on ? "#fff" : "rgba(255,255,255,0.60)",
            background: on ? "rgba(25,112,240,0.20)" : "transparent",
            border: on ? "1px solid rgba(25,112,240,0.30)" : "1px solid transparent",
            transition: "color 0.2s" }}>
          {lab}
        </div>
      );
    })}
  </div>
);
const SiteHeader = ({ active, onNav }) => (
  <div style={{ position: "relative", display: "flex", alignItems: "center",
    minHeight: 86, marginBottom: 26, paddingBottom: 16, borderBottom: `1px solid ${C.line}` }}>
    <div onClick={() => onNav("home")}
      style={{ display: "flex", alignItems: "center", gap: 13, cursor: "pointer", zIndex: 2 }}>
      <Glyph size={34} />
      <Wordmark />
    </div>
    <div style={{ position: "absolute", left: "50%", transform: "translateX(-50%)" }}>
      <NavPill active={active} onNav={onNav} />
    </div>
  </div>
);

// ---------- desk-style form primitives ----------
const Row = ({ name, children, caption }) => (
  <div style={{ display: "grid", gridTemplateColumns: "148px 1fr", gap: 14,
    alignItems: "start", marginBottom: 12 }}>
    <div style={{ fontSize: 12.5, color: C.mute, fontFamily: sans, paddingTop: 12 }}>{name}</div>
    <div>
      {children}
      {caption && <div style={{ fontSize: 11, color: C.faint, marginTop: 4, fontFamily: mono }}>{caption}</div>}
    </div>
  </div>
);
const PanelTitle = ({ children, right }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
    marginBottom: 18, paddingBottom: 10, borderBottom: `1px solid ${C.line}` }}>
    <div style={{ fontSize: 11.5, letterSpacing: "0.18em", textTransform: "uppercase",
      color: C.mute, fontWeight: 700, fontFamily: sans }}>{children}</div>
    {right}
  </div>
);
const PRODUCT_OPTS = [
  ["TARF · Vanilla", "TARF", "vanilla"],
  ["TARF · Liability Knock Out", "TARF", "lko"],
  ["TARF · EKI", "TARF", "eki"],
  ["TARF · Pivot", "TARF", "pivot"],
  ["TARF · EKI Pivot", "TARF", "ekipivot"],
  ["TARF · Discrete", "TARF", "count"],
  ["Accumulator", "ACCU", "vanilla"],
  ["Dual Currency Deposit", "DCD", "vanilla"],
  ["Vanilla Option", "VAN", "vanilla"],
];

/* ———————————————— main app ———————————————— */
export default function StructuredPricer() {
  const [page, setPage] = useState("home"); // 'home' | 'setup' | 'results' | 'rates'
  const [product, setProduct] = useState("TARF");
  const [tarfType, setTarfType] = useState("vanilla"); // 'vanilla' | 'lko'

  const [pair, setPairState] = useState("EUR/USD");
  const PC = PAIRS[pair];
  RATE_DEC = PC.dec;
  const BASE = PC.base, QUOTE = PC.quote;
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
  const [civLoss, setCivLoss] = useState(0.60);
  const [koConvS, setKoConvS] = useState("capped");
  const [accelFA, setAccelFA] = useState(2.00);
  const [sharkDir, setSharkDir] = useState("Bullish");
  const [sharkStrike, setSharkStrike] = useState(1.0850);
  const [sharkBar, setSharkBar] = useState(1.2050);
  const [sharkObs, setSharkObs] = useState("American");
  const [sharkRebate, setSharkRebate] = useState(0.50);
  const [sharkMargin, setSharkMargin] = useState(0.20);
  const [sharkPayCcy, setSharkPayCcy] = useState("USD");
  const [sharkQConv, setSharkQConv] = useState("Strike K");
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
  const [curves, setCurves] = useState({ loading: true, eur: null, usd: null, eurSrc: null, usdSrc: null });
  const touchedEUR = useRef(false), touchedUSD = useRef(false);
  const [volMode, setVolMode] = useState("Flat");
  const [rr25, setRR25] = useState(-1.00);
  const [bf25, setBF25] = useState(0.35);
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

  const omega = buyCcy === BASE ? 1 : -1;
  const effLevBar = levBarSameAsStrike ? strike : levBar;
  const { maturity } = buildSchedule(startDate, Math.max(1, nFix || 1), freq);
  const isLKO = product === "TARF" && tarfType === "lko";
  const isEKI = product === "TARF" && tarfType === "eki";
  const isPivot = product === "TARF" && tarfType === "pivot";
  const isPivotEKI = product === "TARF" && tarfType === "ekipivot";
  const isCount = product === "TARF" && tarfType === "count";
  const isCapLoss = product === "TARF" && tarfType === "caploss";
  const isAccel = product === "TARF" && tarfType === "accel";
  const isPivotFam = isPivot || isPivotEKI;
  const tarfName = tarfType === "lko" ? "Liability Knock Out TARF"
    : tarfType === "eki" ? "EKI TARF"
    : tarfType === "pivot" ? "Pivot TARF"
    : tarfType === "ekipivot" ? "EKI Pivot TARF"
    : tarfType === "count" ? "Discrete TARF"
    : tarfType === "caploss" ? "Cap Loss TARF"
    : tarfType === "accel" ? "Accelerator TARF" : "Vanilla TARF";

  const nav = id => {
    if (id === "home") setPage("home");
    else if (id === "fx") setPage("products");
    else if (id === "equity") setPage("equity");
    else if (id === "hybrid") setPage("hybrid");
    else setPage("rates");
  };

  const anchorLevels = useCallback((s, cfg) => {
    const F = cfg.pip / 0.0001; // offset scale vs a 4dp pair
    const rnd = x => +x.toFixed(cfg.dec);
    setSpot(rnd(s));
    setStrike(rnd(s - 0.0250 * F));
    setLevBar(rnd(s - 0.0250 * F));
    setLkoBar(rnd(s - 0.0850 * F));
    setEkiBar(rnd(s - 0.0650 * F));
    setKLow(rnd(s - 0.0450 * F));
    setPivotLvl(rnd(s));
    setKHigh(rnd(s + 0.0450 * F));
    setELowBar(rnd(s - 0.0750 * F));
    setEHighBar(rnd(s + 0.0750 * F));
    setAccKO(rnd(s + 0.0350 * F));
    setDcdStrike(rnd(s));
    setVanStrike(rnd(s));
    setSharkStrike(rnd(s));
    setSharkBar(rnd(s + 0.1200 * F));
    setSharkPayCcy(cfg.quote); // ~2.6 sigma at default vol: textbook regime, vega and gamma positive at spot
  }, []);

  const changePair = np => {
    const cfg = PAIRS[np];
    setPairState(np);
    setBuyCcy(cfg.base);
    setNotionalCcy(cfg.base);
    setDepCcy(cfg.base);
    setSigma(cfg.vol);
    touchedEUR.current = false; touchedUSD.current = false;
    appliedLive.current = false;
    anchorLevels(cfg.spot0, cfg); // instant sensible levels; live fetch refines
  };

  const fetchSpot = useCallback(async (applyLevels) => {
    const sources = [
      [`https://api.frankfurter.app/latest?from=${BASE}&to=${QUOTE}`, d => d && d.rates && d.rates[QUOTE], "ECB · frankfurter"],
      [`https://open.er-api.com/v6/latest/${BASE}`, d => d && d.rates && d.rates[QUOTE], "open.er-api"],
    ];
    for (const [url, pick, src] of sources) {
      try {
        const r = await fetch(url);
        if (!r.ok) continue;
        const d = await r.json();
        const v = pick(d);
        if (v && isFinite(v)) {
          const s = +(+v).toFixed(PC.dec);
          setSpotLive({ rate: s, src });
          setSpot(s);
          if (applyLevels && !appliedLive.current) {
            appliedLive.current = true;
            anchorLevels(s, PC);
          }
          return;
        }
      } catch (e) { /* try next source */ }
    }
    setSpotLive(null);
  }, [pair, anchorLevels]);
  useEffect(() => { setSpotLive(undefined); fetchSpot(true); }, [fetchSpot]);

  const fetchCurves = useCallback(async () => {
    // EUR: ECB euro area AAA government spot curve (CORS enabled, no key)
    const eur = [];
    const tenors = [["3M", 0.25], ["6M", 0.5], ["1Y", 1], ["2Y", 2], ["5Y", 5]];
    await Promise.all(tenors.map(async ([lab, t]) => {
      try {
        const r = await fetch(
          `https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_${lab}?format=jsondata&lastNObservations=1`);
        if (!r.ok) return;
        const d = await r.json();
        const ser = d && d.dataSets && d.dataSets[0] && d.dataSets[0].series;
        const k0 = ser && Object.keys(ser)[0];
        const obs = k0 && ser[k0].observations;
        const o0 = obs && obs[Object.keys(obs)[0]];
        const v = o0 && o0[0];
        if (v != null && isFinite(v)) eur.push({ t, r: +v });
      } catch (e) { /* skip tenor */ }
    }));
    eur.sort((a, b) => a.t - b.t);
    // USD: best-effort public attempt (US Treasury FiscalData); falls back to manual input
    let usd = [];
    try {
      const r = await fetch(
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/daily_treasury_yield_curve?sort=-record_date&page%5Bsize%5D=1");
      if (r.ok) {
        const d = await r.json();
        const row = d && d.data && d.data[0];
        if (row) {
          const cand = {
            0.25: ["bc_3month", "3_mo", "bc3month", "three_month"],
            0.5: ["bc_6month", "6_mo", "six_month"],
            1: ["bc_1year", "1_yr", "one_year"],
            2: ["bc_2year", "2_yr", "two_year"],
            5: ["bc_5year", "5_yr", "five_year"],
          };
          Object.entries(cand).forEach(([t, names]) => {
            for (const nm of names) {
              if (row[nm] != null && isFinite(+row[nm])) { usd.push({ t: +t, r: +row[nm] }); break; }
            }
          });
          usd.sort((a, b) => a.t - b.t);
        }
      }
    } catch (e) { /* fall back to manual */ }
    setCurves({
      loading: false,
      eur: eur.length >= 2 ? eur : null, usd: usd.length >= 2 ? usd : null,
      eurSrc: eur.length >= 2 ? "ECB AAA curve" : null,
      usdSrc: usd.length >= 2 ? "US Treasury" : null,
    });
  }, []);
  useEffect(() => { fetchCurves(); }, [fetchCurves]);

  // maturity of the current trade in years, for curve interpolation
  const matYears = (() => {
    const today = new Date(); today.setHours(0, 0, 0, 0);
    let d;
    if (product === "DCD" || product === "SHARK") d = dcdMatDate(startDate, dcdTerm);
    else if (product === "VAN") d = dcdMatDate(startDate, vanTerm);
    else d = maturity;
    return d ? Math.max((d - today) / 86400000 / 365, 1 / 365) : 1;
  })();

  // auto-fill the rate fields from the curves at the trade maturity, until the user edits them
  useEffect(() => {
    const byCcy = ccy => ccy === "EUR" ? curves.eur : ccy === "USD" ? curves.usd : null;
    const baseCurve = byCcy(BASE), quoteCurve = byCcy(QUOTE);
    if (baseCurve && !touchedEUR.current) {
      const r = interpCurve(baseCurve, matYears);
      if (r != null && isFinite(r)) setREUR(+(+r).toFixed(2));
    }
    if (quoteCurve && !touchedUSD.current) {
      const r = interpCurve(quoteCurve, matYears);
      if (r != null && isFinite(r)) setRUSD(+(+r).toFixed(2));
    }
  }, [curves, matYears, BASE, QUOTE]);

  const runPricing = useCallback(() => {
    setBusy(true); setErr("");
    setTimeout(() => {
      try {
        if (product === "VAN") {
          const S0v = +spot, Kv = +vanStrike, rd = +rUSD / 100, rf = +rEUR / 100;
          const Nv = +notional;
          if (!(S0v > 0 && Kv > 0 && Nv > 0)) throw new Error("Check spot, strike and notional.");
          const eDate = dcdMatDate(startDate, vanTerm);
          const today = new Date(); today.setHours(0, 0, 0, 0);
          const T = Math.max((eDate - today) / 86400000 / 365, 1 / 365);
          const sig = volMode === "Flat" ? +sigma / 100
            : smileVol(+sigma / 100, +rr25 / 100, +bf25 / 100, S0v * Math.exp((rd - rf) * T), Kv, T);
          const eurN = notionalCcy === BASE ? Nv : Nv / Kv;
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
          const hS = S0v * 0.001, hV = 0.005, hR = 0.001, hG = PC.pip * 1;
          const b0 = posUSD(S0v);
          const delta = (posUSD(S0v + hS) - posUSD(S0v - hS)) / (2 * hS);
          const gamma = (posUSD(S0v + hG) - 2 * b0 + posUSD(S0v - hG)) / (hG * hG);
          const vega = (posUSD(S0v, sig + hV) - posUSD(S0v, sig - hV)) / (2 * hV) * 0.01;
          const theta = posUSD(S0v, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - b0;
          const rhoUSD = (posUSD(S0v, sig, rd + hR) - posUSD(S0v, sig, rd - hR)) / (2 * hR) * 0.0001;
          const rhoEUR = (posUSD(S0v, sig, rd, rf + hR) - posUSD(S0v, sig, rd, rf - hR)) / (2 * hR) * 0.0001;
          const dom = axisDomain([Kv, breakeven], S0v);
          const NS = 25, sLo = dom.lo, sHi = dom.hi;
          const prof = { delta: [], deltaU: [], gamma: [], vega: [], theta: [], rho: [], pv: [] };
          for (let i = 0; i < NS; i++) {
            const s = +(sLo + ((sHi - sLo) * i) / (NS - 1)).toFixed(4);
            prof.pv.push({ s, v: posUSD(s) * cv0 });
            const dCash = (posUSD(s + hS) - posUSD(s - hS)) / (2 * hS);
            prof.delta.push({ s, v: dCash });
            prof.deltaU.push({ s, v: dCash / eurN });
            prof.gamma.push({ s, v: (posUSD(s + hG) - 2 * posUSD(s) + posUSD(s - hG)) / (hG * hG) * PC.pip });
            prof.vega.push({ s, v: (posUSD(s, sig + hV) - posUSD(s, sig - hV)) / (2 * hV) * 0.01 * cv0 });
            prof.theta.push({ s, v: (posUSD(s, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - posUSD(s)) * cv0 });
            prof.rho.push({ s, vUSD: (posUSD(s, sig, rd + hR) - posUSD(s, sig, rd - hR)) / (2 * hR) * 0.0001 * cv0,
                               vEUR: (posUSD(s, sig, rd, rf + hR) - posUSD(s, sig, rd, rf - hR)) / (2 * hR) * 0.0001 * cv0 });
          }
          setRes({
            kind: "VAN", name: vanSide + " " + pair + " " + vanType,
            pair, base: BASE, quote: QUOTE, axLo: dom.lo, axHi: dom.hi,
            sigUsed: sig * 100, volSmile: volMode !== "Flat",
            S0: S0v, K: Kv, side: vanSide, type: vanType, eurN, T,
            expiry: fmtDate(eDate), term: vanTerm,
            premEUR, premUSD, premPct: (premEUR / eurN) * 100, pips: pxPerEUR * 10000,
            probITM: probITM * 100, breakeven, fwd: fwdV,
            delta, deltaPct: (delta / eurN) * 100, gammaPip: gamma * PC.pip,
            vega, theta, rhoUSD, rhoEUR,
            prof, pivotOn: false, lkoOn: false, ekiOn: false, pivotEkiOn: false, accOn: false,
          });
          setSelGreek(null); setPage("results"); setBusy(false);
          return;
        }
        if (product === "SHARK") {
          const S0s = +spot, Ks = +sharkStrike, Hs = +sharkBar, rd = +rUSD / 100, rf = +rEUR / 100;
          const Nn = +notional;
          const omS = sharkDir === "Bullish" ? 1 : -1;
          if (!(S0s > 0 && Ks > 0 && Hs > 0 && Nn > 0)) throw new Error("Check spot, strike, barrier and notional.");
          if (omS === 1 && !(Hs > Ks)) throw new Error("Call up & out: the KO barrier must be above the strike.");
          if (omS === -1 && !(Hs < Ks)) throw new Error("Put down & out: the KO barrier must be below the strike.");
          const sDate = new Date(startDate + "T00:00:00");
          const mDate = dcdMatDate(startDate, dcdTerm);
          const today = new Date(); today.setHours(0, 0, 0, 0);
          const T = Math.max((mDate - today) / 86400000 / 365, 1 / 365);
          const tauAcc = accrualFrac(sDate, mDate, dcdDayCount);
          const sig = volMode === "Flat" ? +sigma / 100
            : smileVol(+sigma / 100, +rr25 / 100, +bf25 / 100, S0s * Math.exp((rd - rf) * T), Ks, T);
          const depIsBase = depCcy === BASE;
          const rDep = (depIsBase ? +rEUR : +rUSD) / 100;
          const marg = (+sharkMargin || 0) / 100;
          const reb = (+sharkRebate || 0) / 100;
          const bud = Math.max((rDep - marg) * tauAcc, 0); // maturity value per 1 notional, in dep ccy

          // ——— Monte Carlo engine: antithetic, seeded, Brownian bridge KO (exact continuous barrier) ———
          const amerS = sharkObs === "American";
          const nSt = amerS ? Math.min(60, Math.max(12, Math.round(T * 52))) : 1;
          const NP2 = Math.max(1000, Math.min(500000, Math.round(+nPaths || 40000)));
          const zS = makeNormals(NP2, nSt, 20240701);
          const uS = makeUniforms(NP2, nSt, 91120033);
          // returns risk-neutral (quote measure) moments of the note payoff pieces
          const runShark = (o = {}, np = NP2) => {
            const S0x = o.S0 !== undefined ? o.S0 : S0s;
            const sg = o.sig !== undefined ? o.sig : sig;
            const rdd = o.rd !== undefined ? o.rd : rd;
            const rff = o.rf !== undefined ? o.rf : rf;
            const TT = o.T !== undefined ? o.T : T;
            const dt = TT / nSt, drift = (rdd - rff - sg * sg / 2) * dt, vs = sg * Math.sqrt(dt);
            let ePerf = 0, ePerfS = 0, eKO = 0, eKOS = 0;
            for (let pth = 0; pth < np; pth++) {
              const base = pth * nSt;
              // American KO: starting at or beyond the barrier means the option is already dead
              let x = S0x, dead = amerS && (omS === 1 ? S0x >= Hs : S0x <= Hs);
              for (let i = 0; i < nSt; i++) {
                const xp = x;
                x = x * Math.exp(drift + vs * zS[base + i]);
                if (!dead) {
                  const crossed = omS === 1 ? x >= Hs : x <= Hs;
                  if (crossed) dead = true;
                  else if (amerS && vs > 0) {
                    const a1 = omS === 1 ? Math.log(Hs / xp) : Math.log(xp / Hs);
                    const a2 = omS === 1 ? Math.log(Hs / x) : Math.log(x / Hs);
                    if (a1 > 0 && a2 > 0 && uS[base + i] < Math.exp(-2 * a1 * a2 / (vs * vs))) dead = true;
                  }
                }
              }
              if (dead) { eKO += 1; eKOS += x; }
              else {
                const perf = Math.max(omS * (x - Ks), 0) / Ks;
                ePerf += perf; ePerfS += perf * x;
              }
            }
            return { ePerf: ePerf / np, ePerfS: ePerfS / np, pKO: eKO / np, eKOS: eKOS / np };
          };

          const base0 = runShark();
          const payBase = sharkPayCcy === BASE;
          // Quanto conventions (Clark 10.3, Trading School FX): the natural payoff O = max(ω(S−K),0) is a
          // QUOTE amount. Settled in BASE it is divided by a conversion rate: S_T (fair conversion, no
          // adjustment), K (standard self quanto), or today's spot. Engine tracks E[O/K·alive] (ePerf)
          // and E[O/K·S_T·alive] (ePerfS): every convention is exact from these two moments.
          const perfMoment = m =>
            !payBase ? m.ePerf                                 // rate O/K paid in QUOTE
            : sharkQConv === "Spot at expiry" ? Ks * m.ePerf   // rate O/S_T in BASE: E[(O/S_T)·S_T] = E[O]
            : sharkQConv === "Spot at T0" ? m.ePerfS * Ks / S0s // rate O/S0 in BASE
            : m.ePerfS;                                        // rate O/K in BASE: self quanto
          const rebMoment = m => payBase ? m.eKOS : m.pKO;     // rebate settled in the payout ccy
          // budget: certain deposit-ccy cash at maturity, valued in QUOTE
          const budPVq = bud * (depIsBase ? Math.exp(-rf * T) * S0s : Math.exp(-rd * T));
          const dfq = Math.exp(-rd * T);
          const perfPVq = dfq * perfMoment(base0);
          const rebPVq = dfq * reb * rebMoment(base0);
          const partRaw = perfPVq > 1e-14 ? (budPVq - rebPVq) / perfPVq : 0;
          const part = Math.max(partRaw, 0);
          const cpnDiv = !payBase ? Ks
            : sharkQConv === "Spot at expiry" ? Hs
            : sharkQConv === "Spot at T0" ? S0s : Ks;
          const maxCpn = part * omS * (Hs - Ks) / cpnDiv;
          const pKO = base0.pKO;

          // package value in QUOTE (participation + rebate legs), participation fixed at the solve
          const pkgQuote = (m, rdd = rd, TT = T) =>
            Nn * Math.exp(-rdd * TT) * (part * perfMoment(m) + reb * rebMoment(m));
          const posQuote = (o = {}, np = NP2) =>
            pkgQuote(runShark(o, np), o.rd !== undefined ? o.rd : rd, o.T !== undefined ? o.T : T);

          const cv0 = 1 / S0s;
          const hS = S0s * 0.002, hV = 0.005, hR = 0.001, hG = S0s * 0.0025;
          const b0 = pkgQuote(base0);
          const delta = (posQuote({ S0: S0s + hS }) - posQuote({ S0: S0s - hS })) / (2 * hS);
          const gamma = (posQuote({ S0: S0s + hG }) - 2 * b0 + posQuote({ S0: S0s - hG })) / (hG * hG);
          const vega = (posQuote({ sig: sig + hV }) - posQuote({ sig: sig - hV })) / (2 * hV) * 0.01;
          const theta = posQuote({ T: Math.max(T - 1 / 365, 1e-6) }) - b0;
          const rhoUSD = (posQuote({ rd: rd + hR }) - posQuote({ rd: rd - hR })) / (2 * hR) * 0.0001;
          const rhoEUR = (posQuote({ rf: rf + hR }) - posQuote({ rf: rf - hR })) / (2 * hR) * 0.0001;
          const eurEqN = depIsBase ? Nn : Nn / S0s;
          const dom = axisDomain([Ks, Hs], S0s);
          const NS = 25, sLo = dom.lo, sHi = dom.hi;
          const npLad = Math.min(NP2, Math.max(8000, Math.round(NP2 / 3)));
          const prof = { delta: [], gamma: [], vega: [], theta: [], rho: [], pv: [] };
          for (let i = 0; i < NS; i++) {
            const s = +(sLo + ((sHi - sLo) * i) / (NS - 1)).toFixed(4);
            const v0 = posQuote({ S0: s }, npLad);
            const vp = posQuote({ S0: s + hG }, npLad), vm = posQuote({ S0: s - hG }, npLad);
            prof.pv.push({ s, v: v0 * cv0 });
            prof.delta.push({ s, v: (vp - vm) / (2 * hG) });
            prof.gamma.push({ s, v: (vp - 2 * v0 + vm) / (hG * hG) * PC.pip });
            prof.vega.push({ s, v: (posQuote({ S0: s, sig: sig + hV }, npLad) - posQuote({ S0: s, sig: sig - hV }, npLad)) / (2 * hV) * 0.01 * cv0 });
            prof.theta.push({ s, v: (posQuote({ S0: s, T: Math.max(T - 1 / 365, 1e-6) }, npLad) - v0) * cv0 });
            prof.rho.push({ s, vUSD: (posQuote({ S0: s, rd: rd + hR }, npLad) - posQuote({ S0: s, rd: rd - hR }, npLad)) / (2 * hR) * 0.0001 * cv0,
                               vEUR: (posQuote({ S0: s, rf: rf + hR }, npLad) - posQuote({ S0: s, rf: rf - hR }, npLad)) / (2 * hR) * 0.0001 * cv0 });
          }
          prof.gamma = smoothProf(smoothProf(prof.gamma));
          // vega at fixed spot across volatility levels, from exactly sigma = 0
          // (at zero vol the path is deterministic: no uncertainty, vega must come out 0 — a built-in engine check)
          {
            const sgHi = Math.max(0.22, sig * 2.4), NV = 25;
            const vv = [];
            for (let i = 0; i < NV; i++) {
              const sg = (sgHi * i) / (NV - 1); // grid starts at 0
              const lo = Math.max(sg - hV, 0), hi = sg + hV;
              vv.push({ sg: +(sg * 100).toFixed(2),
                v: (posQuote({ sig: hi }, npLad) - posQuote({ sig: lo }, npLad)) / (hi - lo) * 0.01 * cv0 });
            }
            prof.vegaVol = smoothProf(vv);
          }
          setRes({
            kind: "SHARK", name: "Sharkfin Note · " + (omS === 1 ? "Bullish " + BASE : "Bearish " + BASE),
            pair, base: BASE, quote: QUOTE, axLo: dom.lo, axHi: dom.hi,
            sigUsed: sig * 100, volSmile: volMode !== "Flat",
            S0: S0s, K: Ks, H: Hs, om: omS, obs: sharkObs, depCcy, N: Nn, T,
            expiry: fmtDate(mDate), term: dcdTerm,
            payCcy: payBase ? BASE : QUOTE, qConv: sharkQConv,
            selfQuanto: payBase && sharkQConv !== "Spot at expiry",
            partPct: part * 100, maxCpnPct: maxCpn * 100, pKOPct: pKO * 100,
            rebPct: reb * 100, budPct: bud * 100, rDepPct: rDep * 100, margPct: marg * 100,
            nPathsUsed: NP2, nStepsUsed: nSt,
            quanto: false, shortBudget: partRaw < 0,
            delta, deltaPct: (delta / Math.max(eurEqN, 1e-9)) * 100, gammaPip: gamma * PC.pip,
            vega, theta, rhoUSD, rhoEUR, prof,
            pivotOn: false, lkoOn: false, ekiOn: false, pivotEkiOn: false, accOn: false, countOn: false, capLossOn: false,
          });
          setSelGreek(null); setPage("results"); setBusy(false);
          return;
        }
        if (product === "DCD") {
          const S0d = +spot, Kd = +dcdStrike, rd = +rUSD / 100, rf = +rEUR / 100;
          const Nd = +notional;
          if (!(S0d > 0 && Kd > 0 && Nd > 0)) throw new Error("Check spot, conversion strike and notional.");
          const mDate = dcdMatDate(startDate, dcdTerm);
          const today = new Date(); today.setHours(0, 0, 0, 0);
          const T = Math.max((mDate - today) / 86400000 / 365, 1 / 365); // option expiry, ACT/365
          const sDate = new Date(startDate + "T00:00:00");
          const tauAcc = accrualFrac(sDate, mDate, dcdDayCount);           // deposit accrual
          const sig = volMode === "Flat" ? +sigma / 100
            : smileVol(+sigma / 100, +rr25 / 100, +bf25 / 100, S0d * Math.exp((rd - rf) * T), Kd, T);
          const marg = (+dcdMargin || 0) / 100;
          const cv0 = 1 / S0d;
          // client's embedded short option, valued in USD
          const posUSD = (s, sg = sig, rdd = rd, rff = rf, TT = T) => {
            const g = gk(s, Kd, TT, rdd, rff, sg);
            return depCcy === BASE ? -Nd * g.call : -(Nd / Kd) * g.put;
          };
          const g0 = gk(S0d, Kd, T, rd, rf, sig);
          const premUSD = depCcy === BASE ? Nd * g0.call : (Nd / Kd) * g0.put;
          const premDep = depCcy === BASE ? premUSD * cv0 : premUSD;
          const rBase = depCcy === BASE ? rf : rd;
          const premNet = Math.max(premDep - Nd * marg * tauAcc, 0);
          const rEnh = rBase + premNet * Math.exp(rBase * T) / (Nd * tauAcc);
          const prob = depCcy === BASE ? normCdf(g0.d2) : normCdf(-g0.d2);
          const cpn = rEnh * tauAcc;
          const breakeven = depCcy === BASE ? Kd * (1 + cpn) : Kd / (1 + cpn);
          const hS = S0d * 0.001, hV = 0.005, hR = 0.001, hG = PC.pip * 1;
          const b0 = posUSD(S0d);
          const delta = (posUSD(S0d + hS) - posUSD(S0d - hS)) / (2 * hS);
          const gamma = (posUSD(S0d + hG) - 2 * b0 + posUSD(S0d - hG)) / (hG * hG);
          const vega = (posUSD(S0d, sig + hV) - posUSD(S0d, sig - hV)) / (2 * hV) * 0.01;
          const theta = posUSD(S0d, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - b0;
          const rhoUSD = (posUSD(S0d, sig, rd + hR) - posUSD(S0d, sig, rd - hR)) / (2 * hR) * 0.0001;
          const rhoEUR = (posUSD(S0d, sig, rd, rf + hR) - posUSD(S0d, sig, rd, rf - hR)) / (2 * hR) * 0.0001;
          const eurEquivN = depCcy === BASE ? Nd : Nd / Kd;
          const dom = axisDomain([Kd, breakeven], S0d);
          const NS = 25, sLo = dom.lo, sHi = dom.hi;
          const prof = { delta: [], gamma: [], vega: [], theta: [], rho: [], pv: [] };
          for (let i = 0; i < NS; i++) {
            const s = +(sLo + ((sHi - sLo) * i) / (NS - 1)).toFixed(4);
            prof.pv.push({ s, v: posUSD(s) * cv0 });
            prof.delta.push({ s, v: (posUSD(s + hS) - posUSD(s - hS)) / (2 * hS) });
            prof.gamma.push({ s, v: (posUSD(s + hG) - 2 * posUSD(s) + posUSD(s - hG)) / (hG * hG) * PC.pip });
            prof.vega.push({ s, v: (posUSD(s, sig + hV) - posUSD(s, sig - hV)) / (2 * hV) * 0.01 * cv0 });
            prof.theta.push({ s, v: (posUSD(s, sig, rd, rf, Math.max(T - 1 / 365, 1e-6)) - posUSD(s)) * cv0 });
            prof.rho.push({ s, vUSD: (posUSD(s, sig, rd + hR) - posUSD(s, sig, rd - hR)) / (2 * hR) * 0.0001 * cv0,
                               vEUR: (posUSD(s, sig, rd, rf + hR) - posUSD(s, sig, rd, rf - hR)) / (2 * hR) * 0.0001 * cv0 });
          }
          setRes({
            kind: "DCD", name: "Dual Currency Deposit",
            pair, base: BASE, quote: QUOTE, axLo: dom.lo, axHi: dom.hi,
            sigUsed: sig * 100, volSmile: volMode !== "Flat",
            S0: S0d, K: Kd, depCcy, N: Nd, T, cpn, maturity: fmtDate(mDate), term: dcdTerm,
            dayCount: dcdDayCount,
            rEnh: rEnh * 100, rBase: rBase * 100, pickup: (rEnh - rBase) * 100,
            premDep, premPct: (premDep / Nd) * 100, prob: prob * 100, breakeven,
            fwd: S0d * Math.exp((rd - rf) * T),
            optName: depCcy === BASE ? `short ${BASE} call / ${QUOTE} put` : `short ${BASE} put / ${QUOTE} call`,
            delta, deltaPct: (delta / eurEquivN) * 100, gammaPip: gamma * PC.pip,
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
          if (omega === 1 && B > K + 1e-12) throw new Error("Buying " + BASE + ": leverage barrier must be at or below the strike.");
          if (omega === -1 && B < K - 1e-12) throw new Error("Buying " + QUOTE + ": leverage barrier must be at or above the strike.");
        }
        const H = +lkoBar;
        const EB = +ekiBar;
        if (isEKI) {
          if (omega === 1 && !(EB < K)) throw new Error("Buying " + BASE + ": KI barrier must be below the strike.");
          if (omega === -1 && !(EB > K)) throw new Error("Buying " + QUOTE + ": KI barrier must be above the strike.");
        }
        if (isLKO) {
          if (omega === 1 && !(H < Math.min(K, B))) throw new Error("Buying " + BASE + ": LKO barrier must be below the strike and leverage barrier.");
          if (omega === -1 && !(H > Math.max(K, B))) throw new Error("Buying " + QUOTE + ": LKO barrier must be above the strike and leverage barrier.");
        }
        if (isACC) {
          if (omega === 1 && !(KOL > K)) throw new Error("Buying " + BASE + ": KO barrier must be above the strike.");
          if (omega === -1 && !(KOL < K)) throw new Error("Buying " + QUOTE + ": KO barrier must be below the strike.");
        }
        const figUnit = PC.pip * 10000; // rate units per 1.00 CIV (100 figures)
        const target = (isACC || isCount) ? 1e18 : +civ * figUnit;
        if (!isACC && !isCount && !(target > 0)) throw new Error("CIV must be above 0.");
        const targetS = +civLoss * figUnit;
        if (isCapLoss && !(targetS > 0)) throw new Error("Loss cap CIV must be above 0.");
        const nGainTarget = Math.max(1, Math.round(+countTarget || 1));
        const n = Math.max(1, Math.min(60, Math.round(+nFix)));
        const { taus, maturity, dates } = buildSchedule(startDate, n, freq);
        const rd = +rUSD / 100, rf = +rEUR / 100;
        const tauNs = taus[taus.length - 1];
        const Kvol = isPivotFam ? PL : K;
        const sig = volMode === "Flat" ? +sigma / 100
          : smileVol(+sigma / 100, +rr25 / 100, +bf25 / 100, S0 * Math.exp((rd - rf) * tauNs), Kvol, tauNs);
        const convRate = isPivotFam ? PL : K;
        const totN = notionalMode === "Per fixing" ? (+notional) * n : +notional;
        const amtPerFixCcy = totN / n;
        const amtEURperFix = notionalCcy === BASE ? amtPerFixCcy : amtPerFixCcy / convRate;
        const totalEUR = notionalCcy === BASE ? totN : totN / convRate;

        const NP = Math.max(2000, Math.min(60000, Math.round(+nPaths)));
        const z = makeNormals(NP, n, 12345);
        const u = makeUniforms(NP, n, 98765);
        const payAtMat = payTiming === "At maturity (ZC)";
        const base = { S0, K, B, L, target, sigma: sig, rd, rf, omega, amtEURperFix, koConv,
          lkoOn: isLKO, H, lkoStyle, lkoVariant, ekiOn: isEKI, E: EB,
          pivotOn: isPivotFam, kLow: KL, kHigh: KH, pivotL: PL,
          pivotEkiOn: isPivotEKI, eLow: ELB, eHigh: EHB, payAtMat,
          accOn: isACC, koLevel: KOL, accStyle: accKoStyle,
          countOn: isCount, targetCount: nGainTarget,
          capLossOn: isCapLoss, targetS, koConvS,
          accelOn: isAccel, accelFA: +accelFA || 0 };
        const P = (over, np = NP, tt = taus) => priceEngine({ ...base, ...over }, z, u, np, tt).pv;
        const r0 = priceEngine(base, z, u, NP, taus, true);

        const hS = S0 * 0.001, hGm = S0 * 0.0025;
        const pvUp = P({ S0: S0 + hS }), pvDn = P({ S0: S0 - hS });
        const delta = (pvUp - pvDn) / (2 * hS);
        const gamma = (P({ S0: S0 + hGm }) - 2 * r0.pv + P({ S0: S0 - hGm })) / (hGm * hGm);
        const hV = 0.005;
        const vega = (P({ sigma: sig + hV }) - P({ sigma: sig - hV })) / (2 * hV) * 0.01;
        const tauTheta = taus.map(t2 => Math.max(t2 - 1 / 365, 1e-6));
        const theta = P({}, NP, tauTheta) - r0.pv;
        const hR = 0.001;
        const rhoUSD = (P({ rd: rd + hR }) - P({ rd: rd - hR })) / (2 * hR) * 0.0001;
        const rhoEUR = (P({ rf: rf + hR }) - P({ rf: rf - hR })) / (2 * hR) * 0.0001;

        const npLad = Math.min(NP, 8000);
        const faV = +accelFA || 0;
        const diagT = isACC ? Math.abs(KOL - K)
          : isCount ? Math.max(Math.abs(S0 - K) * 1.8, 0.028 * K)
          : isAccel ? (faV > 1e-9 ? (Math.sqrt(1 + 4 * faV * target) - 1) / (2 * faV) : target)
          : target;
        const domLvls = isPivotFam
          ? [KL, KH, PL, ...(isPivotEKI ? [ELB, EHB] : [])]
          : [K, ...(!isCount ? [K + omega * diagT] : []),
             ...(isLKO ? [H] : []), ...(isEKI ? [EB] : []),
             ...(isCapLoss ? [K - omega * (targetS / Math.max(L, 1))] : [])];
        const dom = axisDomain(domLvls, S0);
        const NS = 25, sLo = dom.lo, sHi = dom.hi;
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
            prof.gamma.push({ s, v: (l0[i + 1] - 2 * l0[i] + l0[i - 1]) / (ds * ds) * PC.pip });
            // (smoothed after the loop)
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
          sigUsed: sig * 100, volSmile: volMode !== "Flat",
          pair, base: BASE, quote: QUOTE, axLo: dom.lo, axHi: dom.hi,
          targetRate: (isACC || isCount) ? null : target, targetSRate: targetS,
          name: isACC ? "Accumulator · " + accKoStyle + " KO"
            : isAccel ? "Accelerator TARF"
            : isCapLoss ? "Cap Loss TARF"
            : isCount ? "Discrete TARF"
            : isLKO ? "Liability Knock Out TARF" : isEKI ? "EKI TARF"
            : isPivotEKI ? "EKI Pivot TARF" : isPivot ? "Pivot TARF" : "Vanilla TARF",
          pvUSD: r0.pv, se: r0.se, pvEUR: r0.pv / S0,
          pvPct: (r0.pv / (totalEUR * S0)) * 100,
          delta, deltaPct: (delta / totalEUR) * 100,
          gammaPip: gamma * PC.pip, vega, theta, rhoUSD, rhoEUR,
          koProb: r0.koProb * 100, lkoProb: r0.lkoProb * 100,
          ekiProb: r0.ekiProb * 100, expLife: r0.expLifeFix,
          prof: { ...prof, gamma: smoothProf(smoothProf(prof.gamma)) }, fwd, n,
          amtEURperFix, targetFig: isACC ? Math.abs(KOL - K) * 100 : target * 100, S0, K, B, L, omega,
          accOn: isACC, koLevel: KOL, accStyle: accKoStyle,
          countOn: isCount, targetCount: nGainTarget,
          capLossOn: isCapLoss, targetSFig: targetS * 100, koConvS,
          accelOn: isAccel, accelFA: +accelFA || 0,
          lossKoProb: r0.lossKoProb * 100,
          lkoOn: isLKO, H, lkoStyle, lkoVariant, ekiOn: isEKI, E: EB,
          pivotOn: isPivotFam, kLow: KL, kHigh: KH, pivotL: PL,
          pivotEkiOn: isPivotEKI, eLow: ELB, eHigh: EHB, payAtMat,
          terms: {
            pair, buyCcy, notional: totN, notionalCcy,
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
      vanType, vanSide, vanStrike, vanTerm, isCount, countTarget, volMode, rr25, bf25,
      isCapLoss, civLoss, koConvS, isAccel, accelFA, pair, BASE, QUOTE, PC,
      sharkDir, sharkStrike, sharkBar, sharkObs, sharkRebate, sharkMargin, sharkPayCcy, sharkQConv]);

  const products = [
    { id: "TARF", name: "TARF", desc: "Target redemption forward family: strip of leveraged forwards knocked out on accumulated gains", ready: true },
    { id: "ACCU", name: "Accumulator", desc: "Accumulative forward with a knock out barrier: European or American observation, rolling or ZC settlement", ready: true },
    { id: "DCD", name: "Dual Currency Deposit", desc: "Yield enhanced deposit: the client implicitly sells an FX option and earns its premium as extra coupon", ready: true },
    { id: "VAN", name: "Vanilla Option", desc: "European FX call or put, buy or sell, closed form Garman Kohlhagen", ready: true },
    { id: "SHARK", name: "Sharkfin Note", desc: "Capital protected deposit with knock out participation: full capital back, upside via a call up & out or put down & out, rebate if the barrier is hit", ready: true },
  ];
  const tarfTypes = [
    { id: "vanilla", name: "Vanilla TARF", desc: "Leverage applies on every losing fixing until maturity or target knock out" },
    { id: "lko", name: "Liability Knock Out TARF", desc: "A barrier on the loss side knocks the leverage down to 0x if triggered" },
    { id: "eki", name: "EKI TARF", desc: "European KI barrier: obligation only when the fixing lands beyond the barrier; in between, the client trades at market" },
    { id: "pivot", name: "Pivot TARF", desc: "Two sided: sells the base currency at the high strike above the pivot, buys it at the low strike below, leveraged beyond the strikes" },
    { id: "ekipivot", name: "EKI Pivot TARF", desc: "Pivot TARF with a pair of European KI barriers: leveraged obligations only knock in beyond them, participation in between" },
    { id: "count", name: "Discrete TARF", desc: "Knocks out after a fixed number of gaining fixings, regardless of their size, instead of an accumulated CIV amount" },
    { id: "caploss", name: "Cap Loss TARF", desc: "Two targets: knocks out when accumulated gains reach the long target, or when accumulated leveraged losses reach the loss cap" },
    { id: "accel", name: "Accelerator TARF", desc: "Gaining fixings trade at an improved strike: gain = d + F × d², so the target is reached faster. Losses stay a standard ×L at the original strike" },
  ];

  const dirText = omega === 1 ? `Client buys ${BASE} / sells ${QUOTE} at strike` : `Client buys ${QUOTE} / sells ${BASE} at strike`;
  const lossSide = omega === 1 ? "below" : "above";

  /* ————————————————— HOME ————————————————— */
  if (page === "home") {
    const gutter = "clamp(24px, 8vw, 150px)";
    return (
      <div style={{ minHeight: "100vh", background: "#040815", color: "#fff", fontFamily: sans }}>
        <style>{GLOBAL_CSS}</style>
        {/* fixed full-width header, momentum spec */}
        <div style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 50, background: "#040815" }}>
          <div style={{ padding: `0 ${gutter}`, height: 88, display: "flex", alignItems: "center",
            position: "relative" }}>
            <div onClick={() => nav("home")}
              style={{ display: "flex", alignItems: "center", gap: 13, cursor: "pointer", zIndex: 2 }}>
              <Glyph size={40} />
              <Wordmark size={24} />
            </div>
            <div style={{ position: "absolute", left: "50%", transform: "translateX(-50%)" }}>
              <NavPill active="home" onNav={nav} />
            </div>
            <button onClick={() => nav("fx")} className="sp-btn"
              style={{ marginLeft: "auto", padding: "8px 20px", borderRadius: 12, cursor: "pointer",
                background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.10)",
                color: "rgba(255,255,255,0.90)", fontFamily: sans, fontSize: 14, fontWeight: 500 }}>
              Open the pricer
            </button>
          </div>
        </div>

        {/* hero: full screen, picture centered as cover background, text block left */}
        <section style={{ position: "relative", minHeight: "100vh", display: "flex",
          alignItems: "center", justifyContent: "flex-start", overflow: "hidden" }}>
          <div style={{ position: "absolute", inset: 0, backgroundImage: "url(/hero.png)",
            backgroundSize: "cover", backgroundPosition: "center", backgroundRepeat: "no-repeat" }} />
          <div style={{ position: "absolute", inset: 0,
            background: "linear-gradient(90deg, rgba(4,8,21,0.85) 0%, rgba(4,8,21,0.48) 45%, rgba(4,8,21,0.05) 75%)" }} />
          <div style={{ position: "relative", zIndex: 10, width: "100%",
            padding: `128px 24px`, paddingLeft: gutter }}>
            <div className="sp-fade" style={{ maxWidth: 672 }}>
              <h1 style={{ fontSize: "clamp(36px, 5vw, 60px)", fontWeight: 700, lineHeight: 1.12,
                margin: "0 0 48px" }}>
                Price<br />structured products<br />
                <span style={{ color: "#1970F0" }}>to trading floor standards</span>
              </h1>
              <button onClick={() => nav("fx")} className="sp-btn"
                style={{ padding: "12px 24px", border: "2px solid transparent", borderRadius: 8,
                  cursor: "pointer", background: "#1970F0", color: "#fff", fontFamily: sans,
                  fontSize: 16, fontWeight: 600, minWidth: 180, whiteSpace: "nowrap",
                  boxShadow: "0 18px 50px rgba(25,112,240,0.45)" }}>
                Explore the FX suite
              </button>
            </div>
          </div>
        </section>

        {/* platform section: blank video slot for now */}
        <section style={{ padding: "96px 32px", background: "linear-gradient(to bottom, #070D1F, #020409)" }}>
          <div style={{ maxWidth: 896, margin: "0 auto" }}>
            <h2 style={{ fontSize: "clamp(28px, 3vw, 36px)", fontWeight: 700, textAlign: "center",
              margin: "0 0 64px", color: "#fff" }}>
              Discover how the platform works
            </h2>
            <div style={{ aspectRatio: "16 / 9", background: "#000", borderRadius: 8,
              boxShadow: "0 25px 50px rgba(0,0,0,0.5)", display: "flex", alignItems: "center",
              justifyContent: "center", color: "rgba(255,255,255,0.25)", fontSize: 14,
              fontFamily: mono }}>
              Video coming soon
            </div>
          </div>
        </section>

        {/* bank standards section */}
        <section style={{ padding: "80px 32px", background: "#040815" }}>
          <div style={{ maxWidth: 1280, margin: "0 auto" }}>
            <h3 style={{ textAlign: "center", color: "rgba(255,255,255,0.60)", fontSize: 15,
              fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em",
              margin: "0 0 48px" }}>
              Built to the standards of the major banks
            </h3>
            <div style={{ display: "grid", gap: 24, alignItems: "center",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
              {[
                ["Goldman Sachs", "goldman_sachs.png", { fontFamily: "Georgia, serif", fontWeight: 800, fontSize: 19, color: "#0B0B0B" }],
                ["J.P.Morgan", "jpmorgan.png", { fontFamily: "Georgia, serif", fontWeight: 600, fontSize: 19, color: "#101010" }],
                ["Morgan Stanley", "morgan_stanley.png", { fontWeight: 700, fontSize: 17, color: "#0B0B0B" }],
                ["BANK OF AMERICA", "bank_of_america.png", { fontWeight: 800, fontSize: 12.5, color: "#012169", letterSpacing: "0.06em" }],
                ["citi", "citi.png", { fontWeight: 700, fontSize: 26, color: "#004685" }],
                ["BARCLAYS", "barclays.png", { fontWeight: 700, fontSize: 16, color: "#00AEEF", letterSpacing: "0.10em" }],
                ["UBS", "ubs.png", { fontFamily: "Georgia, serif", fontWeight: 800, fontSize: 24, color: "#E60000" }],
                ["NOMURA", "nomura.png", { fontWeight: 800, fontSize: 18, color: "#B60005", letterSpacing: "0.06em" }],
                ["CA CIB", "cacib.png", { fontWeight: 800, fontSize: 19, color: "#57585B" }],
                ["SOCIETE GENERALE", "societe_generale.png", { fontWeight: 800, fontSize: 11.5, color: "#1A1A1A", letterSpacing: "0.04em" }],
                ["BNP PARIBAS", "bnp_paribas.png", { fontWeight: 800, fontSize: 14, color: "#00915A", letterSpacing: "0.04em" }],
                ["NATIXIS", "natixis.png", { fontWeight: 800, fontSize: 16, color: "#6E1E78", letterSpacing: "0.08em" }],
              ].map(([nm, file, st]) => (
                <div key={nm} style={{ height: 96, background: "#FFFFFF", borderRadius: 8,
                  boxShadow: "0 10px 26px rgba(0,0,0,0.35)", display: "flex", alignItems: "center",
                  justifyContent: "center", padding: "0 12px", textAlign: "center" }}>
                  <img src={BANK_LOGOS[file] || "/logos/" + file} alt={nm + " logo"}
                    style={{ maxHeight: 56, maxWidth: "86%", objectFit: "contain", display: "block" }}
                    onError={e => {
                      e.currentTarget.style.display = "none";
                      e.currentTarget.nextSibling.style.display = "block";
                    }} />
                  <span style={{ fontFamily: sans, display: "none", ...st }}>{nm}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* footer */}
        <footer style={{ background: "#060B1D", borderTop: "1px solid rgba(255,255,255,0.10)" }}>
          <div style={{ maxWidth: 1280, margin: "0 auto", padding: "48px 32px" }}>
            <div style={{ display: "grid", gap: 32,
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                  <Glyph size={34} />
                  <Wordmark size={22} />
                </div>
                <p style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", lineHeight: 1.6, margin: 0 }}>
                  The future of structured products.
                </p>
              </div>
              <div>
                <h4 style={{ color: "#fff", fontWeight: 600, fontSize: 15, margin: "0 0 16px" }}>Navigation</h4>
                {[["Home", "home"], ["FX Structured Products", "fx"], ["Rates", "rates"], ["Equity", "equity"], ["Hybrid", "hybrid"]].map(([lab, id]) => (
                  <div key={id} onClick={() => { nav(id); window.scrollTo(0, 0); }}
                    style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", marginBottom: 10,
                      cursor: "pointer" }}
                    onMouseEnter={e => e.currentTarget.style.color = "#5A8DF0"}
                    onMouseLeave={e => e.currentTarget.style.color = "rgba(255,255,255,0.55)"}>
                    {lab}
                  </div>
                ))}
              </div>
              <div>
                <h4 style={{ color: "#fff", fontWeight: 600, fontSize: 15, margin: "0 0 16px" }}>Resources</h4>
                {["Terms of use", "Legal notice", "Privacy policy"].map(t => (
                  <div key={t} style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", marginBottom: 10 }}>{t}</div>
                ))}
              </div>
              <div>
                <h4 style={{ color: "#fff", fontWeight: 600, fontSize: 15, margin: "0 0 16px" }}>Contact</h4>
                <div style={{ fontSize: 14, color: "rgba(255,255,255,0.55)", marginBottom: 10 }}>
                  <span style={{ color: "#1970F0", marginRight: 8 }}>✉</span>contact@ratex.app
                </div>
                <div style={{ fontSize: 14, color: "rgba(255,255,255,0.55)" }}>
                  <span style={{ color: "#1970F0", marginRight: 8 }}>in</span>LinkedIn
                </div>
              </div>
            </div>
            <p style={{ textAlign: "center", fontSize: 14, color: "rgba(255,255,255,0.35)",
              margin: "48px 0 0" }}>
              © 2026 Ratex. All rights reserved.
            </p>
          </div>
        </footer>
      </div>
    );
  }

  /* ————————————————— RATES (placeholder) ————————————————— */
  if (page === "rates") {
    return (
      <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
        <style>{GLOBAL_CSS}</style>
        <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
          <SiteHeader active="rates" onNav={nav} />
          <div style={{ ...card, maxWidth: 640, margin: "12vh auto 0", textAlign: "center", padding: 44 }}>
            <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.02em" }}>
              Rates <span style={{ backgroundImage: "linear-gradient(90deg, #4A7DF0, #6E8EF7)",
                WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Structured Products</span>
            </div>
            <div style={{ marginTop: 16, fontSize: 14.5, color: C.mute, lineHeight: 1.65 }}>
              Coming soon: swaps and asset swap packages, caps & floors, swaptions,
              CMS products and callable notes, priced on the same engine and design language
              as the FX suite.
            </div>
            <button onClick={() => nav("fx")} className="sp-btn"
              style={{ marginTop: 30, padding: "13px 34px", border: "none", borderRadius: 12,
                cursor: "pointer", background: "linear-gradient(135deg, #4A7DF0 0%, #6E8EF7 140%)",
                color: "#fff", fontFamily: sans, fontSize: 14.5, fontWeight: 700,
                boxShadow: "0 8px 26px rgba(74,125,240,0.35)" }}>
              Go to FX Structured Products →
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ————————————————— EQUITY (placeholder) ————————————————— */
  if (page === "equity") {
    return (
      <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
        <style>{GLOBAL_CSS}</style>
        <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
          <SiteHeader active="equity" onNav={nav} />
          <div style={{ ...card, maxWidth: 640, margin: "12vh auto 0", textAlign: "center", padding: 44 }}>
            <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.02em" }}>
              Equity <span style={{ backgroundImage: "linear-gradient(90deg, #4A7DF0, #6E8EF7)",
                WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Structured Products</span>
            </div>
            <div style={{ marginTop: 16, fontSize: 14.5, color: C.mute, lineHeight: 1.65 }}>
              Coming soon: autocalls and phoenix notes, reverse convertibles, bonus and discount
              certificates, priced on the same engine and design language as the FX suite.
            </div>
            <button onClick={() => nav("fx")} className="sp-btn"
              style={{ marginTop: 30, padding: "13px 34px", border: "none", borderRadius: 12,
                cursor: "pointer", background: "linear-gradient(135deg, #4A7DF0 0%, #6E8EF7 140%)",
                color: "#fff", fontFamily: sans, fontSize: 14.5, fontWeight: 700,
                boxShadow: "0 8px 26px rgba(74,125,240,0.35)" }}>
              Go to FX Structured Products →
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ————————————————— HYBRID (placeholder) ————————————————— */
  if (page === "hybrid") {
    return (
      <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
        <style>{GLOBAL_CSS}</style>
        <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
          <SiteHeader active="hybrid" onNav={nav} />
          <div style={{ ...card, maxWidth: 640, margin: "12vh auto 0", textAlign: "center", padding: 44 }}>
            <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.02em" }}>
              Hybrid <span style={{ backgroundImage: "linear-gradient(90deg, #4A7DF0, #6E8EF7)",
                WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Structured Products</span>
            </div>
            <div style={{ marginTop: 16, fontSize: 14.5, color: C.mute, lineHeight: 1.65 }}>
              Coming soon: cross asset payoffs mixing FX, rates and equity, PRDCs, dual digitals
              and best of worst of baskets, priced on the same engine and design language as the FX suite.
            </div>
            <button onClick={() => nav("fx")} className="sp-btn"
              style={{ marginTop: 30, padding: "13px 34px", border: "none", borderRadius: 12,
                cursor: "pointer", background: "linear-gradient(135deg, #4A7DF0 0%, #6E8EF7 140%)",
                color: "#fff", fontFamily: sans, fontSize: 14.5, fontWeight: 700,
                boxShadow: "0 8px 26px rgba(74,125,240,0.35)" }}>
              Go to FX Structured Products →
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ————————————————— FX PRODUCT CATALOG ————————————————— */
  if (page === "products") {
    const pick = (prod, tt) => {
      setProduct(prod);
      if (tt) setTarfType(tt);
      setPage("setup");
    };
    return (
      <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
        <style>{GLOBAL_CSS}</style>
        <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
          <SiteHeader active="fx" onNav={nav} />
          <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", marginBottom: 6 }}>
            FX Structured{" "}
            <span style={{ backgroundImage: "linear-gradient(90deg, #4A7DF0, #6E8EF7)",
              WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Products</span>
          </div>
          <div style={{ fontSize: 13.5, color: C.mute, marginBottom: 26 }}>
            Choose a product to open the pricing ticket.
          </div>

          {/* TARF family */}
          <div style={{ ...card, marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 6 }}>
              <div style={{ fontSize: 17, fontWeight: 700 }}>TARF</div>
              <div style={{ fontSize: 12, color: C.faint }}>
                target redemption forwards · strip of leveraged forwards knocked out on accumulated gains
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginTop: 14 }}>
              {tarfTypes.map(tt => (
                <div key={tt.id} onClick={() => pick("TARF", tt.id)} className="sp-click"
                  style={{ background: C.card2, border: `1.5px solid ${C.line}`, borderRadius: 12,
                    padding: "14px 16px", cursor: "pointer" }}>
                  <div style={{ fontWeight: 700, fontSize: 14 }}>{tt.name}</div>
                  <div style={{ fontSize: 11.5, color: C.mute, marginTop: 5, lineHeight: 1.45 }}>{tt.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* other products */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
            {products.filter(pr => pr.id !== "TARF").map(pr => (
              <div key={pr.id} onClick={() => pick(pr.id)} className="sp-click"
                style={{ ...card, cursor: "pointer" }}>
                <div style={{ fontSize: 16, fontWeight: 700 }}>{pr.name}</div>
                <div style={{ fontSize: 12, color: C.mute, marginTop: 8, lineHeight: 1.5 }}>{pr.desc}</div>
                <div style={{ marginTop: 16, fontSize: 12.5, fontWeight: 700, color: C.blue }}>
                  Open ticket →
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ————————————————— PAGE 1 : SETUP (desk layout) ————————————————— */
  if (page === "setup") {
    const famName = products.find(pr => pr.id === product)?.name || product;
    const tarfLabel = (tarfTypes.find(tt => tt.id === tarfType) || tarfTypes[0]).name;
    const setTarfLabel = lab => {
      const e = tarfTypes.find(tt => tt.name === lab);
      if (e) setTarfType(e.id);
    };
    const isMC = product === "TARF" || product === "ACCU";
    const isDirStrike = (product === "TARF" && !isPivotFam) || product === "ACCU";
    const wayShown = isDirStrike; // pivot family is two sided, DCD/VAN have their own selectors
    return (
      <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
        <style>{GLOBAL_CSS}</style>
        <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
          <SiteHeader active="fx" onNav={nav} />
          <button onClick={() => setPage("products")} className="sp-btn"
            style={{ background: C.card2, border: `1px solid ${C.line}`, color: C.text,
              borderRadius: 10, padding: "8px 15px", fontSize: 13, cursor: "pointer",
              fontFamily: sans, marginBottom: 18 }}>
            ← All products
          </button>

          <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1fr", gap: 20, alignItems: "start" }}>

            {/* ————— GLOBAL CHARACTERISTICS ————— */}
            <div style={card}>
              <PanelTitle>Global Characteristics</PanelTitle>

              <Row name="Product Type"
                caption={product === "TARF" ? "switch between TARF variants here · other products via ← All products" : undefined}>
                {product === "TARF" ? (
                  <Sel v={tarfLabel} set={setTarfLabel} opts={tarfTypes.map(tt => tt.name)} />
                ) : (
                  <div style={{ ...input, background: "transparent",
                    border: `1px dashed ${C.inpLine}`, color: C.text, fontWeight: 600 }}>
                    {famName}
                  </div>
                )}
              </Row>
              <Row name="Currency Pair">
                <Sel v={pair} set={changePair} opts={Object.keys(PAIRS)} />
              </Row>

              {wayShown && (
                <Row name="Customer Way"
                  caption={omega === 1 ? `client buys ${BASE} / sells ${QUOTE} at the strike` : `client buys ${QUOTE} / sells ${BASE} at the strike`}>
                  <Sel v={buyCcy === BASE ? "Buys " + BASE : "Buys " + QUOTE}
                    set={v => setBuyCcy(v === "Buys " + BASE ? BASE : QUOTE)}
                    opts={["Buys " + BASE, "Buys " + QUOTE]} />
                </Row>
              )}

              {product === "VAN" && (<>
                <Row name="Type" caption={vanType === "Call" ? `${BASE} call / ${QUOTE} put` : `${BASE} put / ${QUOTE} call`}>
                  <Sel v={vanType} set={setVanType} opts={["Call", "Put"]} />
                </Row>
                <Row name="Position">
                  <Sel v={vanSide} set={setVanSide} opts={["Buy", "Sell"]} />
                </Row>
              </>)}

              {(product === "DCD" || product === "SHARK") && (
                <Row name="Deposit Currency">
                  <Sel v={depCcy} set={setDepCcy} opts={[BASE, QUOTE]} />
                </Row>
              )}
              {product === "SHARK" && (
                <Row name="Direction"
                  caption={sharkDir === "Bullish"
                    ? `participation if ${BASE} rises, knocked out above the barrier`
                    : `participation if ${BASE} falls, knocked out below the barrier`}>
                  <Sel v={sharkDir === "Bullish" ? `Bullish ${BASE} · call up & out` : `Bearish ${BASE} · put down & out`}
                    set={v => setSharkDir(v.startsWith("Bullish") ? "Bullish" : "Bearish")}
                    opts={[`Bullish ${BASE} · call up & out`, `Bearish ${BASE} · put down & out`]} />
                </Row>
              )}

              <Row name={product === "DCD" ? "Deposit Notional" : "Notional"}
                caption={product === "DCD" ? "placed for the full term"
                  : product === "VAN"
                  ? (notionalCcy === QUOTE ? `= ${fmtBig((+notional || 0) / (+vanStrike || 1))} ${BASE} at the strike` : `${BASE} notional of the option`)
                  : notionalMode === "Total"
                  ? `= ${fmtBig((+notional || 0) / Math.max(1, Math.round(+nFix || 1)))} ${notionalCcy} per fixing`
                  : `= ${fmtBig((+notional || 0) * Math.max(1, Math.round(+nFix || 1)))} ${notionalCcy} total`}>
                {isMC ? (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 128px 88px", gap: 10 }}>
                    <NotionalInput v={notional} set={setNotional} />
                    <Sel v={notionalMode} set={setNotionalMode} opts={["Total", "Per fixing"]} />
                    <Sel v={notionalCcy} set={setNotionalCcy} opts={[BASE, QUOTE]} />
                  </div>
                ) : product === "VAN" ? (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 88px", gap: 10 }}>
                    <NotionalInput v={notional} set={setNotional} />
                    <Sel v={notionalCcy} set={setNotionalCcy} opts={[BASE, QUOTE]} />
                  </div>
                ) : (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 88px", gap: 10 }}>
                    <NotionalInput v={notional} set={setNotional} />
                    <div style={{ ...input, textAlign: "center", color: C.mute }}>{depCcy}</div>
                  </div>
                )}
              </Row>

              {product === "TARF" && !isCount && (
                <Row name={<span>CIV Target{" "}
                    <span onClick={() => setShowCivHelp(v => !v)}
                      style={{ display: "inline-flex", alignItems: "center", justifyContent: "center",
                        width: 15, height: 15, borderRadius: 8, marginLeft: 4, cursor: "pointer",
                        border: `1px solid ${showCivHelp ? C.blue : C.faint}`,
                        color: showCivHelp ? C.blue : C.faint, fontSize: 10, fontWeight: 700,
                        verticalAlign: "middle" }}>?</span></span>}
                  caption={`= ${fmt((+civ || 0) * 100, 0)} figures of accumulated gain`}>
                  <Num v={civ} set={setCiv} step="0.05" min="0" />
                  {showCivHelp && (
                    <div style={{ marginTop: 8, padding: "10px 12px", background: "rgba(74,125,240,0.07)",
                      border: `1px solid rgba(74,125,240,0.35)`, borderRadius: 10,
                      fontSize: 11.5, color: C.mute, lineHeight: 1.55 }}>
                      1 CIV = 100 figures and 1 figure = 100 pips ({(PC.pip * 100).toFixed(PC.dec)} in {pair}
                      rate terms). The trade knocks out once the sum of in the money fixings reaches
                      CIV × 100 figures of accumulated intrinsic value.
                    </div>
                  )}
                </Row>
              )}
              {isCount && (
                <Row name="Target # of Gains" caption="trade stops after this many ITM fixings">
                  <Num v={countTarget} set={setCountTarget} step="1" min="1" />
                </Row>
              )}
              {isCapLoss && (
                <Row name="Loss Cap (CIV)"
                  caption={`= ${fmt((+civLoss || 0) * 100, 0)} figures of accumulated leveraged loss`}>
                  <Num v={civLoss} set={setCivLoss} step="0.05" min="0" />
                </Row>
              )}
              {product === "TARF" && (
                <Row name={isCapLoss ? "Gain KO Settlement" : "KO Settlement"}>
                  <Sel v={koConv} set={setKoConv} opts={["full", "capped", "none"]} />
                </Row>
              )}
              {isCapLoss && (
                <Row name="Loss KO Settlement"
                  caption="payment at the fixing that breaches the loss cap">
                  <Sel v={koConvS} set={setKoConvS} opts={["full", "capped", "none"]} />
                </Row>
              )}

              {isDirStrike && (
                <Row name="Strike"
                  caption={product === "ACCU"
                    ? (omega === 1 ? "below market, client accumulates here" : "above market")
                    : undefined}>
                  <RateNum v={strike} set={setStrike} />
                </Row>
              )}

              {isPivotFam && (<>
                <Row name="Low Strike (Call)"
                  caption={`client buys ${BASE} here below pivot · high strike mirrors automatically`}>
                  <RateNum v={kLow} set={v => {
                    setKLow(v);
                    if (v !== "" && isFinite(+v) && isFinite(+pivotLvl))
                      setKHigh(+(2 * +pivotLvl - +v).toFixed(PC.dec));
                  }} /></Row>
                <Row name="Pivot Level" caption="both strikes stay symmetric around the pivot">
                  <RateNum v={pivotLvl} set={v => {
                    const w = (isFinite(+kHigh) && isFinite(+kLow)) ? Math.abs(+kHigh - +kLow) / 2 : 0;
                    setPivotLvl(v);
                    if (v !== "" && isFinite(+v) && w > 0) {
                      setKLow(+(+v - w).toFixed(PC.dec));
                      setKHigh(+(+v + w).toFixed(PC.dec));
                    }
                  }} /></Row>
                <Row name="High Strike (Put)"
                  caption={`client sells ${BASE} here above pivot · low strike mirrors automatically`}>
                  <RateNum v={kHigh} set={v => {
                    setKHigh(v);
                    if (v !== "" && isFinite(+v) && isFinite(+pivotLvl))
                      setKLow(+(2 * +pivotLvl - +v).toFixed(PC.dec));
                  }} /></Row>
              </>)}
              {isPivotEKI && (<>
                <Row name="Low KI Barrier" caption="below Low Strike · high barrier mirrors automatically">
                  <RateNum v={eLowBar} set={v => {
                    setELowBar(v);
                    if (v !== "" && isFinite(+v) && isFinite(+pivotLvl))
                      setEHighBar(+(2 * +pivotLvl - +v).toFixed(PC.dec));
                  }} /></Row>
                <Row name="High KI Barrier" caption="above High Strike · low barrier mirrors automatically">
                  <RateNum v={eHighBar} set={v => {
                    setEHighBar(v);
                    if (v !== "" && isFinite(+v) && isFinite(+pivotLvl))
                      setELowBar(+(2 * +pivotLvl - +v).toFixed(PC.dec));
                  }} /></Row>
              </>)}

              {(product === "TARF" || product === "ACCU") && (
                <Row name="Leverage Ratio"><Num v={leverage} set={setLeverage} step="0.5" min="1" /></Row>
              )}
              {isAccel && (
                <Row name="F"
                  caption={`improved strike = K ${omega === 1 ? "−" : "+"} F × (K − fixing)² on gaining fixings`}>
                  <Num v={accelFA} set={setAccelFA} step="0.25" min="0" />
                </Row>
              )}

              {product === "TARF" && !isEKI && !isPivotFam && (
                <Row name="Leverage Barrier"
                  caption={!levBarSameAsStrike ? undefined : "leverage applies beyond the strike"}>
                  <div style={{ display: "grid", gridTemplateColumns: levBarSameAsStrike ? "1fr" : "auto 1fr", gap: 10, alignItems: "center" }}>
                    <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5,
                      color: C.mute, cursor: "pointer", height: 41 }}>
                      <input type="checkbox" checked={levBarSameAsStrike}
                        onChange={e => setLevBarSameAsStrike(e.target.checked)} />
                      same as strike
                    </label>
                    {!levBarSameAsStrike && <RateNum v={levBar} set={setLevBar} />}
                  </div>
                </Row>
              )}

              {isLKO && (<>
                <Row name="LKO Barrier" caption={omega === 1 ? "below strike" : "above strike"}>
                  <RateNum v={lkoBar} set={setLkoBar} /></Row>
                <Row name="LKO Observation">
                  <Sel v={lkoStyle} set={setLkoStyle} opts={["European", "American"]} /></Row>
                <Row name="LKO Variant"
                  caption={lkoVariant === "Standard"
                    ? "if triggered, loss side leverage drops to 0x for the remaining fixings"
                    : "if triggered, the strategy terminates and the outstanding target is paid upfront"}>
                  <Sel v={lkoVariant} set={setLkoVariant} opts={["Standard", "Accelerated"]} /></Row>
              </>)}

              {isEKI && (
                <Row name="KI Barrier"
                  caption={(omega === 1 ? "below strike · " : "above strike · ") +
                    "no obligation between strike and barrier; beyond it, leveraged at the strike"}>
                  <RateNum v={ekiBar} set={setEkiBar} />
                </Row>
              )}

              {product === "ACCU" && (<>
                <Row name="KO Barrier" caption={omega === 1 ? "above strike, cancels the trade" : "below strike, cancels the trade"}>
                  <RateNum v={accKO} set={setAccKO} /></Row>
                <Row name="KO Observation">
                  <Sel v={accKoStyle} set={setAccKoStyle} opts={["European", "American"]} /></Row>
              </>)}

              {product === "DCD" && (<>
                <Row name="Conversion Strike"
                  caption={depCcy === BASE ? `converted into ${QUOTE} if fixing above` : `converted into ${BASE} if fixing below`}>
                  <RateNum v={dcdStrike} set={setDcdStrike} /></Row>
                <Row name="Bank Margin (% p.a.)" caption="deducted from the coupon">
                  <Num v={dcdMargin} set={setDcdMargin} step="0.05" min="0" /></Row>
              </>)}

              {product === "VAN" && (
                <Row name="Strike"><RateNum v={vanStrike} set={setVanStrike} /></Row>
              )}
              {product === "SHARK" && (<>
                <Row name="Payout Currency"
                  caption={sharkPayCcy === QUOTE
                    ? `natural settlement: the payoff ${sharkDir === "Bullish" ? "(S − K)" : "(K − S)"} is a ${QUOTE} amount`
                    : `self quanto: the ${QUOTE} payoff is settled in ${BASE}, changing PV and participation`}>
                  <Sel v={sharkPayCcy === BASE ? BASE : QUOTE}
                    set={v => setSharkPayCcy(v)} opts={[QUOTE, BASE]} />
                </Row>
                {sharkPayCcy === BASE && (
                  <Row name="Quanto Conversion"
                    caption={sharkQConv === "Strike K"
                      ? "payoff divided by K, settled in " + BASE + ": the standard self quanto"
                      : sharkQConv === "Spot at T0"
                      ? "payoff divided by today's spot, fixed at inception"
                      : "converted at the expiry fixing: economically identical to the " + QUOTE + " payout, no quanto adjustment"}>
                    <Sel v={sharkQConv} set={setSharkQConv}
                      opts={["Strike K", "Spot at T0", "Spot at expiry"]} />
                  </Row>
                )}
                <Row name="Strike" caption="participation starts here">
                  <RateNum v={sharkStrike} set={setSharkStrike} /></Row>
                <Row name="KO Barrier"
                  caption={sharkDir === "Bullish" ? "above the strike, cancels the participation" : "below the strike, cancels the participation"}>
                  <RateNum v={sharkBar} set={setSharkBar} /></Row>
                <Row name="Barrier Observation"
                  caption={sharkObs === "European" ? "checked at maturity only" : "monitored continuously until maturity"}>
                  <Sel v={sharkObs} set={setSharkObs} opts={["European", "American"]} /></Row>
                <Row name="Rebate if KO (%)" caption="paid at maturity on top of capital if knocked out">
                  <Num v={sharkRebate} set={setSharkRebate} step="0.05" min="0" /></Row>
                <Row name="Bank Margin (% p.a.)" caption="deducted from the interest budget">
                  <Num v={sharkMargin} set={setSharkMargin} step="0.05" min="0" /></Row>
              </>)}

              <div style={{ fontSize: 12, color: C.faint, marginTop: 16, lineHeight: 1.55,
                paddingTop: 14, borderTop: `1px solid ${C.line}` }}>
                {product === "SHARK"
                  ? <>Capital is fully protected: the deposit interest, net of the margin, buys a{" "}
                    {sharkDir === "Bullish" ? "call up & out" : "put down & out"} on {pair}. The participation is
                    solved so the package is fair. If the barrier is {sharkObs === "European" ? "beyond at maturity" : "touched at any time"},
                    the coupon is replaced by the rebate. Priced by Monte Carlo with exact settlement in the
                    deposit currency.</>
                  : product === "DCD"
                  ? <>The client places a {depCcy} deposit and implicitly sells the bank a{" "}
                    {depCcy === BASE ? `${BASE} call / ${QUOTE} put` : `${BASE} put / ${QUOTE} call`} struck at the conversion
                    strike; the premium is returned as an enhanced coupon. At maturity the bank repays the
                    weaker currency.</>
                  : product === "VAN"
                  ? <>European exercise, cash settled at expiry, priced closed form under Garman Kohlhagen.
                    The client {vanSide === "Buy" ? "pays" : "receives"} the premium today.</>
                  : product === "ACCU"
                  ? <>100% of the per fixing notional on favourable fixings, ×{leverage || "…"} on unfavourable
                    ones. If the {accKoStyle === "European" ? "fixing" : "spot path"} crosses the KO barrier,
                    that fixing and all remaining fixings are cancelled.</>
                  : isPivotFam
                  ? <>Above the pivot the client sells {BASE} at the High Strike; below it the client buys {BASE} at
                    the Low Strike.{" "}
                    {isPivotEKI ? "Obligations only knock in beyond the KI barriers; participation in between."
                      : `Beyond either strike the obligation runs on ×${leverage || "…"} of the notional.`}{" "}
                    Gains accrue toward the target; losses never count. {QUOTE} notionals convert at the pivot.</>
                  : isAccel
                  ? <>Every gaining fixing trades at the improved strike K {omega === 1 ? "−" : "+"} F × (K − fixing)²:
                    the client receives d + F × d² per fixing and the CIV target fills faster, so the trade
                    knocks out sooner. Losing fixings stay a standard ×{leverage || "…"} at the original strike.</>
                  : isCapLoss
                  ? <>Two accumulators run in parallel: gains accrue toward the long CIV target, leveraged
                    losses accrue toward the loss cap. The trade knocks out when either side is reached, so the
                    client's aggregate downside is bounded at the cap.</>
                  : isCount
                  ? <>Every favourable fixing pays 1x in full and counts one unit; the trade knocks out after
                    the target number of gains. Unfavourable fixings pay ×{leverage || "…"} and never count.</>
                  : <>Gains accrue toward the CIV target; losses pay ×{leverage || "…"} beyond{" "}
                    <span style={{ fontFamily: mono }}>{fmtRate(+effLevBar || 0)}</span> and never count toward it.</>}
              </div>
            </div>

            {/* ————— MARKET + DATES ————— */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div style={card}>
                <PanelTitle right={
                  <button onClick={() => { fetchSpot(false); fetchCurves(); }} className="sp-btn"
                    style={{ background: "linear-gradient(135deg, #4A7DF0 0%, #6E8EF7 140%)", border: "none",
                      color: "#fff", borderRadius: 8, padding: "5px 14px", fontSize: 11.5, fontWeight: 700,
                      cursor: "pointer", fontFamily: sans }}>Reload All</button>
                }>Market</PanelTitle>

                <Row name="Spot Rate"
                  caption={spotLive === undefined ? `fetching live ${pair}…`
                    : spotLive && spotLive.rate ? `live ${spotLive.rate.toFixed(RATE_DEC)} · ${spotLive.src}`
                    : "live feed unavailable · manual"}>
                  <RateNum v={spot} set={setSpot} />
                </Row>
                <Row name={QUOTE + " Rate (%)"}>
                  <Num v={rUSD} set={v => { touchedUSD.current = true; setRUSD(v); }} step="0.05" />
                </Row>
                <Row name={BASE + " Rate (%)"}
                  caption={curves.loading ? "fetching yield curves…"
                    : `${BASE} ${(BASE === "EUR" && curves.eurSrc) || (BASE === "USD" && curves.usdSrc) ? ((BASE === "EUR" ? curves.eurSrc : curves.usdSrc) + " @ " + fmt(matYears, 2) + "y") : "manual"} · ${QUOTE} ${(QUOTE === "EUR" && curves.eurSrc) || (QUOTE === "USD" && curves.usdSrc) ? (QUOTE === "EUR" ? curves.eurSrc : curves.usdSrc) : "manual"}`}>
                  <Num v={rEUR} set={v => { touchedEUR.current = true; setREUR(v); }} step="0.05" />
                </Row>
                <Row name="Vol Input">
                  <Sel v={volMode} set={setVolMode} opts={["Flat", "Smile"]} />
                </Row>
                {volMode === "Flat" ? (
                  <Row name="Vol σ (%)"><Num v={sigma} set={setSigma} step="0.25" /></Row>
                ) : (<>
                  <Row name="ATM σ (%)"><Num v={sigma} set={setSigma} step="0.25" /></Row>
                  <Row name="25Δ RR (%)"><Num v={rr25} set={setRR25} step="0.05" /></Row>
                  <Row name="25Δ BF (%)"
                    caption="Malz smile · each product priced at σ(strike)">
                    <Num v={bf25} set={setBF25} step="0.05" /></Row>
                </>)}
                {(isMC || product === "SHARK") && (
                  <Row name="MC Paths" caption={`${fmt(Math.max(0, Math.round(+nPaths || 0)), 0)} simulated paths, antithetic`}>
                    <NotionalInput v={nPaths} set={setNPaths} />
                  </Row>
                )}
              </div>

              <div style={card}>
                <PanelTitle>Dates</PanelTitle>
                <Row name="Trade Date">
                  <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                    color: C.mute }}>{fmtDate(new Date())}</div>
                </Row>
                <Row name="Start Date">
                  <input type="date" className="sp-input" value={startDate} style={input}
                    onChange={e => setStartDate(e.target.value)} />
                </Row>
                {isMC && (<>
                  <Row name="Frequency"><Sel v={freq} set={setFreq} opts={["Weekly", "Biweekly", "Monthly"]} /></Row>
                  <Row name="Nb Fixings"><Num v={nFix} set={setNFix} step="1" min="1" /></Row>
                  <Row name="Settlement Mode"
                    caption={payTiming === "Rolling" ? "each fixing settles on its date" : "all flows accumulate and pay at maturity"}>
                    <Sel v={payTiming} set={setPayTiming} opts={["Rolling", "At maturity (ZC)"]} /></Row>
                  <Row name="Maturity">
                    <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                      color: C.blue, fontWeight: 600 }}>{fmtDate(maturity)}</div>
                  </Row>
                </>)}
                {(product === "DCD" || product === "SHARK") && (<>
                  <Row name="Deposit Term">
                    <Sel v={dcdTerm} set={setDcdTerm} opts={["2W", "1M", "2M", "3M", "6M", "12M"]} /></Row>
                  <Row name="Day Count" caption="deposit interest accrual">
                    <Sel v={dcdDayCount} set={setDcdDayCount} opts={["ACT/365", "ACT/360", "30/360", "ACT/ACT"]} /></Row>
                  <Row name="Maturity">
                    <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                      color: C.blue, fontWeight: 600 }}>{fmtDate(dcdMatDate(startDate, dcdTerm))}</div>
                  </Row>
                </>)}
                {product === "VAN" && (<>
                  <Row name="Expiry Term">
                    <Sel v={vanTerm} set={setVanTerm} opts={["1W", "2W", "1M", "2M", "3M", "6M", "12M"]} /></Row>
                  <Row name="Expiry Date">
                    <div style={{ ...input, background: "transparent", border: `1px dashed ${C.inpLine}`,
                      color: C.blue, fontWeight: 600 }}>{fmtDate(dcdMatDate(startDate, vanTerm))}</div>
                  </Row>
                </>)}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, marginTop: 26 }}>
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
    { id: "delta", name: "Delta", unit: res.base + " equivalent", color: C.blue, val: fmtBigSigned(res.delta) },
    res.kind === "VAN"
      ? { id: "deltaU", name: "Delta", unit: "unitless, (1.00) to 1.00", color: C.text,
          val: fmtSigned(res.delta / res.eurN, 4) }
      : { id: "delta", name: "Delta %", unit: "of " + res.base + " notional", color: C.text, val: fmtSigned(res.deltaPct, 1) + "%" },
    { id: "gamma", name: "Gamma", unit: "Δdelta per 1 pip", color: C.text, val: fmtBigSigned(res.gammaPip) },
    { id: "vega", name: "Vega", unit: res.base + " per 1 vol pt", color: C.blue, val: fmtBigSigned(res.vega / res.S0) },
    { id: "theta", name: "Theta", unit: res.base + " per day", color: C.text, val: fmtBigSigned(res.theta / res.S0) },
    { id: "rho", name: "Rho " + res.quote + " / " + res.base, unit: res.base + " per 1 bp", color: C.text,
      val: `${fmtBigSigned(res.rhoUSD / res.S0)} / ${fmtBigSigned(res.rhoEUR / res.S0)}` },
  ] : [];
  const greekMeta = {
    delta: { title: "Delta vs spot", unit: res ? res.base + " equivalent" : "", keys: [["v", C.blue, "Delta"]] },
    deltaU: { title: "Delta vs spot", unit: "unitless, from (1.00) to 1.00", keys: [["v", C.blue, "Delta"]] },
    gamma: { title: "Gamma vs spot", unit: "Δdelta per 1 pip", keys: [["v", C.amber, "Gamma"]] },
    vega:  { title: "Vega vs spot", unit: res.base + " per 1 vol pt", keys: [["v", C.blue, "Vega"]] },
    theta: { title: "Theta vs spot", unit: res.base + " per day", keys: [["v", C.amber, "Theta"]] },
    rho:   { title: "Rho vs spot", unit: res.base + " per 1 bp",
             keys: [["vUSD", C.blue, "Rho USD"], ["vEUR", C.red, "Rho EUR"]] },
  };
  const tickFmt = v => fmtBig(v);

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: sans, padding: "28px 24px 60px" }}>
      <style>{GLOBAL_CSS}</style>
      <div className="sp-fade" style={{ maxWidth: 1180, margin: "0 auto" }}>
        <SiteHeader active="fx" onNav={nav} />
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 18 }}>
          <button onClick={() => setPage("setup")} className="sp-btn"
            style={{ background: C.card2, border: `1px solid ${C.line}`, color: C.text,
              borderRadius: 10, padding: "9px 16px", fontSize: 13.5, cursor: "pointer", fontFamily: sans }}>
            ← Edit parameters
          </button>
          <div style={{ fontSize: 20, fontWeight: 800 }}>
            {res ? res.name : (product === "DCD" ? "Dual Currency Deposit"
              : product === "ACCU" ? "Accumulator"
              : product === "VAN" ? "Vanilla Option"
              : product === "SHARK" ? "Sharkfin Note" : tarfName)}{" "}
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

        {res && res.kind === "SHARK" && (
          <div style={{ ...card, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, marginBottom: 18, padding: 0 }}>
            <div style={{ padding: "20px 22px" }}>
              <div style={colHead}>Participation</div>
              <div style={{ fontFamily: mono, fontSize: 30, marginTop: 6, color: C.green,
                fontVariantNumeric: "tabular-nums" }}>
                {fmt(res.partPct, 1)}<span style={{ fontSize: 16, color: C.mute }}>%</span>
              </div>
              <div style={{ fontSize: 11, color: res.shortBudget ? C.amber : C.faint, marginTop: 4 }}>
                {res.shortBudget ? "interest budget short of the rebate cost"
                  : "of " + res.base + " performance beyond the strike, paid in " + res.payCcy}
              </div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Max coupon</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8, color: C.green }}>
                +{fmt(res.maxCpnPct, 2)}%
              </div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4, fontFamily: mono }}>
                just before the barrier at {fmtRate(res.H)}</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Knock out</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8, color: C.amber }}>
                {fmt(res.pKOPct, 1)}%
              </div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4, fontFamily: mono }}>
                P(KO) · rebate {fmt(res.rebPct, 2)}% if hit</div>
            </div>
          </div>
        )}

        {res && res.kind === "SHARK" && (
          <div style={{ ...card, marginBottom: 18 }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 14 }}>Note economics</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
              {[
                ["Deposit", fmtBig(res.N) + " " + res.depCcy + " · " + res.term + " · " + res.expiry],
                ["Interest budget", fmt(res.rDepPct, 2) + "% − " + fmt(res.margPct, 2) + "% margin = " + fmt(res.budPct, 2) + "% at maturity"],
                ["Strike / Barrier", fmtRate(res.K) + " / " + fmtRate(res.H) + " · " + res.obs + " KO"],
                ["Capital floor", "100% guaranteed" + (res.rebPct > 0 ? " + " + fmt(res.rebPct, 2) + "% rebate if KO" : "")],
                ["Direction", res.om === 1 ? "Bullish " + res.base + " · call up & out" : "Bearish " + res.base + " · put down & out"],
                ["Payout", res.payCcy + (res.payCcy !== res.quote
                  ? " · " + (res.qConv === "Spot at expiry" ? "converted at the expiry fixing, no quanto adjustment"
                    : res.qConv === "Spot at T0" ? "self quanto, converted at inception spot"
                    : "self quanto, converted at the strike") : " · natural settlement")],
                ["Pricing", "Monte Carlo · " + fmt(res.nPathsUsed, 0) + " paths" + (res.obs === "American" ? " · bridge KO monitoring" : " · maturity fixing") + (res.selfQuanto ? " · S_T weighted expectation" : "")],
              ].map(([k, v]) => (
                <div key={k} style={{ background: C.card2, border: `1px solid ${C.line}`,
                  borderRadius: 10, padding: "11px 14px" }}>
                  <div style={colHead}>{k}</div>
                  <div style={{ fontFamily: mono, fontSize: 13, marginTop: 6 }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {res && res.kind === "VAN" && (
          <div style={{ ...card, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, marginBottom: 18, padding: 0 }}>
            <div style={{ padding: "20px 22px" }}>
              <div style={colHead}>Premium</div>
              <div style={{ fontFamily: mono, fontSize: 30, marginTop: 6, fontVariantNumeric: "tabular-nums",
                color: res.side === "Buy" ? C.red : C.green }}>
                {res.side === "Buy" ? `(${bigRaw(res.premEUR)})` : "+" + bigRaw(res.premEUR)}{" "}
                <span style={{ fontSize: 14, color: C.mute }}>{res.base}</span>
              </div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>
                client {res.side === "Buy" ? "pays" : "receives"} today</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Premium % / pips</div>
              <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8 }}>{fmt(res.premPct, 3)}%</div>
              <div style={{ fontSize: 11, color: C.faint, marginTop: 4, fontFamily: mono }}>
                of {res.base} notional · {fmt(res.pips, 1)} {res.quote} pips per {res.base}</div>
            </div>
            <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
              <div style={colHead}>Premium in {res.quote}</div>
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
                ["Contract", (res.type === "Call" ? `${res.base} call / ${res.quote} put` : `${res.base} put / ${res.quote} call`) + " · European"],
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

        {(!res || (res.kind !== "DCD" && res.kind !== "VAN" && res.kind !== "SHARK")) && (
        <div style={{ ...card, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, marginBottom: 18, padding: 0 }}>
          <div style={{ padding: "20px 22px" }}>
            <div style={colHead}>PV (client side)</div>
            <div style={{ fontFamily: mono, fontSize: 30, marginTop: 6, fontVariantNumeric: "tabular-nums",
              color: res && res.pvEUR >= 0 ? C.green : C.red }}>
              {res ? fmtBigSigned(res.pvEUR) : "…"} <span style={{ fontSize: 14, color: C.mute }}>{res ? res.base : ""}</span>
            </div>
          </div>
          <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
            <div style={colHead}>PV % of notional</div>
            <div style={{ fontFamily: mono, fontSize: 22, marginTop: 8,
              color: res && res.pvPct >= 0 ? C.green : C.red }}>
              {res ? fmtSigned(res.pvPct, 3) + "%" : "…"}</div>
            <div style={{ fontSize: 11, color: C.faint, marginTop: 4 }}>of {res.base} notional</div>
          </div>
          <div style={{ padding: "20px 22px", borderLeft: `1px solid ${C.line}` }}>
            <div style={colHead}>PV in {res ? res.quote : "quote ccy"}</div>
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
                    {["#", "Fixing date", "τ (yrs)", "Forward", "DF", `Amount ${res.base}`, "P live",
                      ...(res.accOn ? ["P KO"] : []), `E[CF] ${res.base}`, "Σ E[CF]"].map((h, i) => (
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
                      tickFormatter={v => v.toFixed(RATE_DEC)} stroke={C.line} />
                    <YAxis tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                      tickFormatter={tickFmt} stroke={C.line} width={56} />
                    <Tooltip contentStyle={{ background: C.card2, border: `1px solid ${C.line}`,
                        fontFamily: mono, fontSize: 12, borderRadius: 8 }}
                      labelFormatter={v => "Spot " + (+v).toFixed(RATE_DEC)}
                      formatter={(v, nm) => [fmt(v, 0), nm]} />
                    <ReferenceLine y={0} stroke={C.mute} strokeWidth={1} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.S0} stroke={C.blue} strokeDasharray="4 3"
                      label={{ value: "Spot " + res.S0.toFixed(RATE_DEC), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top" }} />
                    {!res.pivotOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.K} stroke={C.amber} strokeDasharray="4 3"
                        label={{ value: "K " + res.K.toFixed(RATE_DEC), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 15 }} />
                    )}
                    {res.pivotOn && (<>
                      <ReferenceLine ifOverflow="extendDomain" x={res.kLow} stroke={C.amber} strokeDasharray="4 3"
                        label={{ value: "KL " + res.kLow.toFixed(RATE_DEC), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 15 }} />
                      <ReferenceLine ifOverflow="extendDomain" x={res.pivotL} stroke={C.blue} strokeDasharray="4 3"
                        label={{ value: "P " + res.pivotL.toFixed(RATE_DEC), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                      <ReferenceLine ifOverflow="extendDomain" x={res.kHigh} stroke={C.amber} strokeDasharray="4 3"
                        label={{ value: "KH " + res.kHigh.toFixed(RATE_DEC), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 15 }} />
                    </>)}
                    {res.lkoOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.H} stroke={C.violet} strokeDasharray="4 3"
                        label={{ value: "LKO " + res.H.toFixed(RATE_DEC), fill: C.violet, fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                    )}
                    {res.accOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.koLevel} stroke={C.red} strokeDasharray="4 3"
                        label={{ value: "KO " + res.koLevel.toFixed(RATE_DEC), fill: C.red, fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                    )}
                    {res.ekiOn && (
                      <ReferenceLine ifOverflow="extendDomain" x={res.E} stroke="#C89A4B" strokeDasharray="4 3"
                        label={{ value: "KI " + res.E.toFixed(RATE_DEC), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                    )}
                    {res.pivotEkiOn && (<>
                      <ReferenceLine ifOverflow="extendDomain" x={res.eLow} stroke="#C89A4B" strokeDasharray="4 3"
                        label={{ value: "KI " + res.eLow.toFixed(RATE_DEC), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                      <ReferenceLine ifOverflow="extendDomain" x={res.eHigh} stroke="#C89A4B" strokeDasharray="4 3"
                        label={{ value: "KI " + res.eHigh.toFixed(RATE_DEC), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                    </>)}
                    {greekMeta[selGreek].keys.map(([k, col, nm]) => (
                      <Line key={k} dataKey={k} name={nm} dot={false} stroke={col}
                        strokeWidth={2.2} type="monotone" connectNulls />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              {selGreek === "vega" && res && res.kind === "SHARK" && res.prof.vegaVol && (
                <div style={{ marginTop: 18 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                    <div style={{ fontWeight: 700, fontSize: 14.5 }}>
                      Vega vs volatility
                      <span style={{ color: C.faint, fontWeight: 400, fontSize: 12.5 }}>
                        {" "}· {res.base} per 1 vol pt, spot held at {fmtRate(res.S0)}</span>
                    </div>
                  </div>
                  <div style={{ height: 240 }}>
                    <ResponsiveContainer>
                      <ComposedChart data={res.prof.vegaVol} margin={{ top: 20, right: 16, bottom: 4, left: 8 }}>
                        <CartesianGrid stroke={C.line} strokeDasharray="2 4" vertical={false} />
                        <XAxis dataKey="sg" type="number" domain={["dataMin", "dataMax"]}
                          tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                          tickFormatter={v => fmt(v, 0) + "%"} stroke={C.line} />
                        <YAxis tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                          tickFormatter={tickFmt} stroke={C.line} width={56} />
                        <Tooltip contentStyle={{ background: C.card2, border: `1px solid ${C.line}`,
                            fontFamily: mono, fontSize: 12, borderRadius: 8 }}
                          labelFormatter={v => "σ " + fmt(+v, 2) + "%"}
                          formatter={(v, nm) => [fmt(v, 0), nm]} />
                        <ReferenceLine y={0} stroke={C.mute} strokeWidth={1} />
                        <ReferenceLine ifOverflow="extendDomain" x={res.sigUsed} stroke={C.blue} strokeDasharray="4 3"
                          label={{ value: "σ " + fmt(res.sigUsed, 2) + "%", fill: C.blue, fontSize: 10, fontFamily: mono, position: "top" }} />
                        <Line type="monotone" dataKey="v" name="Vega" stroke={C.violet} strokeWidth={2.4} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ fontSize: 11.5, color: C.faint, marginTop: 8, lineHeight: 1.55 }}>
                    Two competing forces: volatility raises the option value but also the knock out probability.
                    At low vol the barrier is far away in probability space and vega is positive; as vol rises
                    P(KO) grows faster than the upside, vega peaks, declines through zero, and at high vol the
                    barrier dominates completely. Where the marked σ sits on this curve decides the sign of the
                    vega you see at spot in the chart above.
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div style={{ ...card, marginBottom: 18, paddingBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>Payoff diagram
              <span style={{ color: C.faint, fontWeight: 400, fontSize: 12.5 }}>
                {res && res.kind === "DCD" ? " · final redemption at maturity"
                  : res && res.kind === "VAN" ? " · at expiry, net of premium"
                  : res && res.kind === "SHARK" ? " · redemption at maturity, % of nominal"
                  : " · per fixing, client perspective"}</span>
            </div>
            {res && res.kind !== "DCD" && res.kind !== "VAN" && res.kind !== "SHARK" && !res.accOn && !res.countOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              gain leg stops once {fmt(res.targetFig, 0)} figures accumulated
            </div>}
            {res && res.countOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              knocks out after {res.targetCount} gaining fixings
            </div>}
            {res && res.capLossOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              loss cap {fmt(res.targetSFig, 0)} fig · P(loss KO) {fmt(res.lossKoProb, 1)}%
            </div>}
            {res && res.accOn && <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint }}>
              {res.accStyle} KO at {fmtRate(res.koLevel)} · {res.payAtMat ? "ZC settlement" : "rolling settlement"}
            </div>}
          </div>
          {res && res.kind === "SHARK"
            ? <SharkfinDiagram res={res} C={C} mono={mono} sans={sans} />
            : res && res.kind === "VAN"
            ? <VanillaDiagram res={res} C={C} mono={mono} sans={sans} />
            : res && res.kind === "DCD"
            ? <DCDDiagram res={res} C={C} mono={mono} sans={sans} />
            : res && res.pivotOn
            ? <PivotPayoffDiagram res={res} C={C} mono={mono} sans={sans} />
            : <PayoffDiagram res={res} C={C} mono={mono} sans={sans} />}
          <div style={{ fontSize: 11, color: C.faint, marginTop: 2 }}>
            {res && res.kind === "SHARK"
              ? `Capital is repaid at 100% everywhere. The fin: participation of ${fmt(res.partPct, 1)}% on the ${res.base} performance beyond the strike, growing until the barrier. ${res.obs === "European" ? "If the fixing at maturity is beyond the barrier" : "If spot touches the barrier at any time"}, the coupon is replaced by the ${fmt(res.rebPct, 2)}% rebate.`
              : res && res.kind === "VAN"
              ? `Net position at expiry: intrinsic value ${res.side === "Buy" ? "minus the premium paid" : "minus intrinsic, plus the premium received"}. Breakeven is the strike ${res.type === "Call" ? "plus" : "minus"} the premium in ${res.quote} pips per ${res.base}.`
              : res && res.kind === "DCD"
              ? `Nominal redemption only, coupon excluded (the coupon of ${fmtBig(res.N * res.cpn)} ${res.depCcy} is paid regardless). Flat leg: nominal repaid in ${res.depCcy}. Sloped leg: the bank repays the alternative currency converted at the strike, so the ${res.depCcy} value of the nominal falls with the fixing. The amber marker is the breakeven including the coupon.`
              : res && res.accelOn
              ? `The gain leg is a curve, not a line: each favourable fixing trades at the improved strike (F = ${fmt(res.accelFA, 2)}), paying d + F × d² and filling the target quicker, which pulls the KO closer to the strike. The loss side is a standard ×${res.L} at the original strike.`
              : res && res.capLossOn
              ? `Two knock outs: the gain leg stops at the long target (KO marker) and the leveraged loss leg is bounded by the loss cap: once accumulated leveraged losses reach ${fmt(res.targetSFig, 0)} figures the trade terminates, flooring the per fixing loss at the dashed level (CAP marker).`
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
                    tickFormatter={v => v.toFixed(RATE_DEC)} stroke={C.line} />
                  <YAxis tick={{ fill: C.mute, fontSize: 10, fontFamily: mono }}
                    tickFormatter={tickFmt} stroke={C.line} width={56} />
                  <Tooltip contentStyle={{ background: C.card2, border: `1px solid ${C.line}`,
                      fontFamily: mono, fontSize: 12, borderRadius: 8 }}
                    labelFormatter={v => "Spot " + (+v).toFixed(RATE_DEC)}
                    formatter={v => [fmt(v, 0) + " EUR", "PV"]} />
                  <ReferenceLine y={0} stroke={C.mute} strokeWidth={1} />
                  <ReferenceLine ifOverflow="extendDomain" x={res.S0} stroke={C.blue} strokeDasharray="4 3"
                    label={{ value: "Spot " + res.S0.toFixed(RATE_DEC), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top" }} />
                  {!res.pivotOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.K} stroke={C.amber} strokeDasharray="4 3"
                      label={{ value: "K " + res.K.toFixed(RATE_DEC), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 15 }} />
                  )}
                  {res.pivotOn && (<>
                    <ReferenceLine ifOverflow="extendDomain" x={res.kLow} stroke={C.amber} strokeDasharray="4 3"
                      label={{ value: "KL " + res.kLow.toFixed(RATE_DEC), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 15 }} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.pivotL} stroke={C.blue} strokeDasharray="4 3"
                      label={{ value: "P " + res.pivotL.toFixed(RATE_DEC), fill: C.blue, fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.kHigh} stroke={C.amber} strokeDasharray="4 3"
                      label={{ value: "KH " + res.kHigh.toFixed(RATE_DEC), fill: C.amber, fontSize: 10, fontFamily: mono, position: "top", dy: 15 }} />
                  </>)}
                  {res.lkoOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.H} stroke={C.violet} strokeDasharray="4 3"
                      label={{ value: "LKO " + res.H.toFixed(RATE_DEC), fill: C.violet, fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                  )}
                  {res.accOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.koLevel} stroke={C.red} strokeDasharray="4 3"
                      label={{ value: "KO " + res.koLevel.toFixed(RATE_DEC), fill: C.red, fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                  )}
                  {res.ekiOn && (
                    <ReferenceLine ifOverflow="extendDomain" x={res.E} stroke="#C89A4B" strokeDasharray="4 3"
                      label={{ value: "KI " + res.E.toFixed(RATE_DEC), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                  )}
                  {res.pivotEkiOn && (<>
                    <ReferenceLine ifOverflow="extendDomain" x={res.eLow} stroke="#C89A4B" strokeDasharray="4 3"
                      label={{ value: "KI " + res.eLow.toFixed(RATE_DEC), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
                    <ReferenceLine ifOverflow="extendDomain" x={res.eHigh} stroke="#C89A4B" strokeDasharray="4 3"
                      label={{ value: "KI " + res.eHigh.toFixed(RATE_DEC), fill: "#C89A4B", fontSize: 10, fontFamily: mono, position: "top", dy: 30 }} />
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
            {res && res.kind === "SHARK"
              ? ` Value of the participation and rebate legs in ${res.base} (the guaranteed deposit is excluded): the classic sharkfin hump, rising with spot then collapsing toward the barrier where the coupon dies, negative gamma concentrated at the fin edge.`
              : res && res.kind === "VAN"
              ? ` Mark to market of the position in ${res.base} before expiry: the smooth curve above the expiry hockey stick is time value, largest at the strike where gamma and vega ${res.side === "Buy" ? "peak" : "trough"}.`
              : res && res.kind === "DCD"
              ? ` Mark to market of the embedded short option in ${res.base} (the deposit leg itself is excluded): most negative where the option is deepest in the bank's favour, with vega concentrated near the strike.`
              : res && res.accelOn
              ? " Accelerated gains cut both ways: the upside per fixing is richer, but the target fills sooner so the trade dies earlier when spot runs favourably, while the leveraged downside is untouched: expected life drops and the KO probability rises versus the vanilla TARF."
              : res && res.capLossOn
              ? " The loss cap puts a floor under the downside: unlike the vanilla TARF whose PV dives linearly, here cumulative losses cannot exceed the cap, so the left tail of the profile flattens toward a bounded worst case."
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
