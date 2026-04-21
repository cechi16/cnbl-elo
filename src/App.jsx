import { Analytics } from "@vercel/analytics/react"
import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import SimWorker from "./simWorker.js?worker";

const K = 16;
const BASE_ELO = 1000;
const HOME_ADV = 50;

function expectedScore(ra, rb) {
  return 1 / (1 + Math.pow(10, (rb - ra) / 400));
}


function fairOdds(probHome) {
  const pH = probHome / 100;
  const pA = 1 - pH;
  const oddsH = pH > 0 ? (1 / pH).toFixed(2) : "—";
  const oddsA = pA > 0 ? (1 / pA).toFixed(2) : "—";
  return { oddsH, oddsA };
}

// 538-style ELO: margin-of-victory multiplier + home court advantage
// decay: after each match pull ratings toward baseElo (0 = off, 0.02 = mild, 0.05 = strong)
// movK: MoV sensitivity constant — higher = bigger swings for blowouts (538 default: 2.2)
function updateElo538(homeRa, awayRb, homeScore, awayScore, K, homeAdv, decay = 0, baseElo = BASE_ELO, movK = 2.2) {
  const expectedHome = expectedScore(homeRa + homeAdv, awayRb);
  const diff = homeScore - awayScore;
  if (diff === 0) return [homeRa, awayRb];
  const homeWon = diff > 0;
  const winnerElo = homeWon ? homeRa : awayRb;
  const loserElo  = homeWon ? awayRb  : homeRa;
  const movMult = Math.log(Math.abs(diff) + 1) * (movK / ((winnerElo - loserElo) * 0.001 + movK));
  const delta = K * movMult * ((homeWon ? 1 : 0) - expectedHome);
  // Apply decay: gently pull ratings back toward baseline after every match
  const nh = (homeRa + delta) * (1 - decay) + baseElo * decay;
  const na = (awayRb - delta) * (1 - decay) + baseElo * decay;
  return [Math.round(nh), Math.round(na)];
}

const SAMPLE_TEAMS = [
  "ERA Basketball Nymburk", "BK KVIS Pardubice", "PUMPA Basket Brno",
  "SLUNETA Ústí nad Labem", "Sršni Photomate Písek", "BK Opava",
  "BK ARMEX ENERGY Děčín", "NH Ostrava", "BK Olomoucko",
  "USK Praha", "SK Slavia Praha", "BK GAPA Hradec Králové"
];

function generateSampleResults() {
  const results = [];
  const teams = [...SAMPLE_TEAMS];
  const now = new Date();
  for (let w = 20; w >= 1; w--) {
    const matchCount = Math.floor(Math.random() * 4) + 4;
    for (let m = 0; m < matchCount; m++) {
      const shuffled = [...teams].sort(() => Math.random() - 0.5);
      const home = shuffled[0], away = shuffled[1];
      const homeWin = Math.random() > 0.45;
      const d = new Date(now);
      d.setDate(d.getDate() - w * 7 - Math.floor(Math.random() * 3));
      results.push({
        Date: d.toISOString().slice(0, 10), Home: home, Away: away,
        HomeTeam: home, AwayTeam: away,
        HomeScore: homeWin ? Math.floor(Math.random()*30)+75 : Math.floor(Math.random()*25)+60,
        AwayScore: homeWin ? Math.floor(Math.random()*25)+60 : Math.floor(Math.random()*30)+75,
      });
    }
  }
  return results.sort((a, b) => a.Date.localeCompare(b.Date));
}

function computeElo(results) {
  const ratings = {};
  const history = {};
  const historyDated = {};
  const changes = [];

  results.forEach(r => {
    const home = r.HomeTeam ?? r.home ?? r.Home ?? "";
    const away = r.AwayTeam ?? r.away ?? r.Away ?? "";
    const homeScore = r.HomeScore ?? r.homeScore ?? 0;
    const awayScore = r.AwayScore ?? r.awayScore ?? 0;
    const date = r.Date ?? r.date ?? "";
    if (!home || !away) return;

    if (!ratings[home]) { ratings[home] = BASE_ELO; history[home] = [BASE_ELO]; historyDated[home] = [{ elo: BASE_ELO, date: "" }]; }
    if (!ratings[away]) { ratings[away] = BASE_ELO; history[away] = [BASE_ELO]; historyDated[away] = [{ elo: BASE_ELO, date: "" }]; }

    const prev = ratings[home];
    const [nh, na] = updateElo538(ratings[home], ratings[away], homeScore, awayScore, K, HOME_ADV);
    ratings[home] = nh; ratings[away] = na;
    changes.push(nh - prev);
    history[home].push(nh);
    history[away].push(na);
    historyDated[home].push({ elo: nh, date });
    historyDated[away].push({ elo: na, date });
  });

  return { ratings, history, historyDated, changes };
}

function computeEloParametric(results, K, baseElo, homeAdv, decay = 0, movK = 2.2) {
  const ratings = {};
  results.forEach(r => {
    const home = r.HomeTeam ?? r.Home ?? "";
    const away = r.AwayTeam ?? r.Away ?? "";
    const hs = r.HomeScore ?? 0, as_ = r.AwayScore ?? 0;
    if (!home || !away) return;
    if (!ratings[home]) ratings[home] = baseElo;
    if (!ratings[away]) ratings[away] = baseElo;
    const [nh, na] = updateElo538(ratings[home], ratings[away], hs, as_, K, homeAdv, decay, baseElo, movK);
    ratings[home] = nh; ratings[away] = na;
  });
  return ratings;
}

// Walk-forward validation: 5 expanding folds, each tests a 10% chunk unseen during training
function runWalkForward(results, K, baseElo, homeAdv, decay = 0, movK = 2.2) {
  const FOLDS = [
    [0.40, 0.50], [0.50, 0.60], [0.60, 0.70],
    [0.70, 0.75], [0.75, 0.80], [0.80, 0.85],
    [0.85, 0.90], [0.90, 0.95], [0.95, 1.00],
  ];
  const eps = 1e-7;
  const foldResults = [];

  for (const [trainFrac, testFrac] of FOLDS) {
    const n = results.length;
    const trainData = results.slice(0, Math.floor(n * trainFrac));
    const testData  = results.slice(Math.floor(n * trainFrac), Math.floor(n * testFrac));
    if (testData.length < 3) continue;

    const ratings = {};
    const init = (t) => { if (!ratings[t]) ratings[t] = baseElo; };
    let homeWinsTrain = 0;

    trainData.forEach(r => {
      const home = r.HomeTeam ?? r.Home ?? "", away = r.AwayTeam ?? r.Away ?? "";
      const hs = r.HomeScore ?? 0, as_ = r.AwayScore ?? 0;
      if (!home || !away) return;
      init(home); init(away);
      if (hs > as_) homeWinsTrain++;
      const [nh, na] = updateElo538(ratings[home], ratings[away], hs, as_, K, homeAdv, decay, baseElo, movK);
      ratings[home] = nh; ratings[away] = na;
    });

    const pHR = trainData.length > 0 ? homeWinsTrain / trainData.length : 0.5;
    let eloAcc = 0, eloBrier = 0, eloLL = 0;
    let haAcc = 0, haBrier = 0, hrAcc = 0, hrBrier = 0, count = 0;

    testData.forEach(r => {
      const home = r.HomeTeam ?? r.Home ?? "", away = r.AwayTeam ?? r.Away ?? "";
      const hs = r.HomeScore ?? 0, as_ = r.AwayScore ?? 0;
      if (!home || !away) return;
      init(home); init(away);
      const y = hs > as_ ? 1 : 0;
      const p = expectedScore(ratings[home] + homeAdv, ratings[away]);

      if ((p >= 0.5) === (y === 1)) eloAcc++;
      eloBrier += (p - y) ** 2;
      eloLL += -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));

      if (y === 1) haAcc++;
      haBrier += (1 - y) ** 2;

      if ((pHR >= 0.5) === (y === 1)) hrAcc++;
      hrBrier += (pHR - y) ** 2;

      count++;
      const [nh, na] = updateElo538(ratings[home], ratings[away], hs, as_, K, homeAdv, decay, baseElo, movK);
      ratings[home] = nh; ratings[away] = na;
    });

    if (count === 0) continue;
    foldResults.push({
      n: count,
      elo: { acc: eloAcc / count, brier: eloBrier / count, ll: eloLL / count },
      homeAlways: { acc: haAcc / count, brier: haBrier / count },
      homeRate:   { acc: hrAcc / count, brier: hrBrier / count },
    });
  }

  if (!foldResults.length) return null;
  const mean = (key, sub) => foldResults.reduce((s, f) => s + f[key][sub], 0) / foldResults.length;

  return {
    foldCount: foldResults.length,
    totalGames: foldResults.reduce((s, f) => s + f.n, 0),
    accuracy:  (mean('elo', 'acc') * 100).toFixed(1),
    brier:      mean('elo', 'brier').toFixed(4),
    logLoss:    mean('elo', 'll').toFixed(4),
    baselines: {
      homeAlways: { accuracy: (mean('homeAlways', 'acc') * 100).toFixed(1), brier: mean('homeAlways', 'brier').toFixed(4) },
      homeRate:   { accuracy: (mean('homeRate',   'acc') * 100).toFixed(1), brier: mean('homeRate',   'brier').toFixed(4) },
      coin:       { accuracy: "50.0", brier: "0.2500" },
    },
  };
}

// Collect per-game predictions from walk-forward folds for calibration + P&L simulation
function collectWalkForwardPredictions(results, K, baseElo, homeAdv, decay = 0, movK = 2.2) {
  const FOLDS = [
    [0.40, 0.50], [0.50, 0.60], [0.60, 0.70],
    [0.70, 0.75], [0.75, 0.80], [0.80, 0.85],
    [0.85, 0.90], [0.90, 0.95], [0.95, 1.00],
  ];
  const preds = [];
  for (const [trainFrac, testFrac] of FOLDS) {
    const n = results.length;
    const trainData = results.slice(0, Math.floor(n * trainFrac));
    const testData  = results.slice(Math.floor(n * trainFrac), Math.floor(n * testFrac));
    const ratings = {};
    const init = t => { if (!ratings[t]) ratings[t] = baseElo; };
    trainData.forEach(r => {
      const home = r.HomeTeam ?? r.Home ?? "", away = r.AwayTeam ?? r.Away ?? "";
      if (!home || !away) return;
      init(home); init(away);
      const [nh, na] = updateElo538(ratings[home], ratings[away], r.HomeScore ?? 0, r.AwayScore ?? 0, K, homeAdv, decay, baseElo, movK);
      ratings[home] = nh; ratings[away] = na;
    });
    testData.forEach(r => {
      const home = r.HomeTeam ?? r.Home ?? "", away = r.AwayTeam ?? r.Away ?? "";
      const hs = r.HomeScore ?? 0, as_ = r.AwayScore ?? 0;
      if (!home || !away) return;
      init(home); init(away);
      const p = expectedScore(ratings[home] + homeAdv, ratings[away]);
      preds.push({ p, y: hs > as_ ? 1 : 0, date: r.Date ?? "" });
      const [nh, na] = updateElo538(ratings[home], ratings[away], hs, as_, K, homeAdv, decay, baseElo, movK);
      ratings[home] = nh; ratings[away] = na;
    });
  }
  return preds.sort((a, b) => a.date.localeCompare(b.date));
}

// Calibration: bucket by model confidence, compare predicted % vs actual win %
function computeCalibration(preds) {
  const bins = [[0.50,0.55],[0.55,0.60],[0.60,0.65],[0.65,0.70],[0.70,0.75],[0.75,0.80],[0.80,1.01]];
  return bins.map(([lo, hi]) => {
    const sub = preds.filter(({p}) => { const c = Math.max(p, 1-p); return c >= lo && c < hi; });
    const wins = sub.filter(({p, y}) => (p >= 0.5 ? y === 1 : y === 0)).length;
    return { label: `${Math.round(lo*100)}–${hi >= 1 ? '100' : Math.round(hi*100)}%`, n: sub.length, pred: ((lo+hi)/2*100).toFixed(0), actual: sub.length ? (wins/sub.length*100).toFixed(1) : null };
  });
}

function getUpcoming() {
  return [
    { HomeTeam: "PUMPA Basket Brno",       AwayTeam: "ERA Basketball Nymburk", Date: "2026-04-18" },
    { HomeTeam: "BK ARMEX ENERGY Děčín",   AwayTeam: "SLUNETA Ústí nad Labem", Date: "2026-04-18" },
    { HomeTeam: "BK Opava",                AwayTeam: "NH Ostrava",              Date: "2026-04-18" },
    { HomeTeam: "BK KVIS Pardubice",       AwayTeam: "Sršni Photomate Písek",   Date: "2026-04-18" },
  ];
}

function MiniChart({ data, color }) {
  if (!data || data.length < 2) return null;
  const w = 80, h = 32, pad = 2;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - pad * 2);
    const y = pad + ((max - v) / range) * (h - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function normWords(str) {
  return str.toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9 ]/g, ' ')
    .trim().split(/\s+/).filter(w => w.length > 2);
}
function teamsMatch(full, short) {
  const fw = normWords(full), sw = normWords(short);
  return sw.some(w => fw.includes(w));
}
function findMatchOdds(home, away, matches, matchDate) {
  // For playoff series the SAME teams play multiple games with home/away swapping
  // between games — so reverse-matched odds from another date would be wrong.
  // Require exact date match when matchDate is provided; fall back to no-date
  // matching only when the caller didn't give us a date.
  const sameDate = matchDate
    ? matches.filter(m => m.Date === matchDate)
    : matches;

  const direct = sameDate.find(m => teamsMatch(home, m.HomeTeam) && teamsMatch(away, m.AwayTeam));
  if (direct) return direct;
  const rev = sameDate.find(m => teamsMatch(home, m.AwayTeam) && teamsMatch(away, m.HomeTeam));
  if (!rev) return null;
  // Flip bookmaker odds to match caller's home/away orientation
  return {
    ...rev,
    Bookmakers: Object.fromEntries(
      Object.entries(rev.Bookmakers ?? {}).map(([k, o]) => [k, { HomeOdds: o.AwayOdds, AwayOdds: o.HomeOdds }])
    ),
  };
}

function getOpeningOdds(oddsHistory, home, away, matchDate) {
  for (const snap of oddsHistory) {
    const m = findMatchOdds(home, away, snap.Matches ?? [], matchDate);
    if (m && Object.keys(m.Bookmakers ?? {}).length) return m;
  }
  return null;
}

const TAB_COLORS = { leaderboard: "#378ADD", predictions: "#1D9E75", results: "#D85A30", backtest: "#7C52C8", oddshistory: "#B07A10", historical: "#2E7D5E", simulace: "#C0392B" };

export default function App() {
  const [tab, setTab] = useState("leaderboard");
  const [results, setResults] = useState([]);
  const [ratings, setRatings] = useState({});
  const [history, setHistory] = useState({});
  const [upcoming, setUpcoming] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [btK, setBtK] = useState(16);
  const [btBase, setBtBase] = useState(1000);
  const [btHomeAdv, setBtHomeAdv] = useState(50);
  const [gsBest, setGsBest] = useState(null);
  const [changes, setChanges] = useState([]);
  const [historyDated, setHistoryDated] = useState({});
  const [oddsHistory, setOddsHistory] = useState([]);
  const [selectedMatchKey, setSelectedMatchKey] = useState(null);
  const [historicalOdds, setHistoricalOdds] = useState([]);
  const [histBk, setHistBk] = useState("Tipsport");
  const [simBk, setSimBk] = useState("Tipsport");
  const [simOdds, setSimOdds] = useState("close");
  const [simEdge, setSimEdge] = useState(0);

  const latestOdds = oddsHistory.length ? (oddsHistory[oddsHistory.length - 1].Matches ?? []) : [];

  const [btDecay, setBtDecay] = useState(0);
  const [btMovK, setBtMovK] = useState(2.2);

  const btResult = useMemo(() => runWalkForward(results, btK, btBase, btHomeAdv, btDecay, btMovK), [results, btK, btBase, btHomeAdv, btDecay, btMovK]);
  const btRatings = useMemo(() => computeEloParametric(results, btK, btBase, btHomeAdv, btDecay, btMovK), [results, btK, btBase, btHomeAdv, btDecay, btMovK]);
  const btPreds   = useMemo(() => collectWalkForwardPredictions(results, btK, btBase, btHomeAdv, btDecay, btMovK), [results, btK, btBase, btHomeAdv, btDecay, btMovK]);
  const btCalib   = useMemo(() => computeCalibration(btPreds), [btPreds]);
  const btRanked = useMemo(
    () => Object.entries(btRatings).sort((a, b) => b[1] - a[1]).map(([team, elo], i) => ({ team, elo, rank: i + 1 })),
    [btRatings]
  );

  const eloP = useCallback((home, away) => {
    const eH = btRatings[home] ?? Object.entries(btRatings).find(([t]) => teamsMatch(t, home))?.[1] ?? btBase;
    const eA = btRatings[away] ?? Object.entries(btRatings).find(([t]) => teamsMatch(t, away))?.[1] ?? btBase;
    return expectedScore(eH + btHomeAdv, eA);
  }, [btRatings, btBase, btHomeAdv]);

  const [monteCarlo,   setMonteCarlo]   = useState(null);
  const [bettingMC,    setBettingMC]    = useState(null);
  const [roiComp,      setRoiComp]      = useState(null);
  const [nextSeason,   setNextSeason]   = useState(null);
  const [simRunning,   setSimRunning]   = useState(false);
  const workerRef = useRef(null);
  const [scrapeStatus, setScrapeStatus] = useState(null); // null | "running" | "ok" | "error"
  const [lastUpdated,  setLastUpdated]  = useState(null);

  const triggerScrape = useCallback(async () => {
    const pwd = window.prompt("Heslo pro aktualizaci:");
    if (pwd !== "123123") return;
    setScrapeStatus("running");
    try {
      const r = await fetch("/api/trigger-scrape", { method: "POST" });
      setScrapeStatus(r.ok ? "ok" : "error");
    } catch {
      setScrapeStatus("error");
    }
  }, []);

  const runAllSims = useCallback(() => {
    if (!results.length || !Object.keys(btRatings).length) return;
    setSimRunning(true);
    setMonteCarlo(null); setBettingMC(null); setRoiComp(null); setNextSeason(null);

    if (workerRef.current) workerRef.current.terminate();
    const worker = new SimWorker();
    workerRef.current = worker;

    let done = 0;
    const total = 4;
    worker.onmessage = ({ data }) => {
      if (data.type === "monteCarlo")    setMonteCarlo(data.result);
      if (data.type === "bettingMC")     setBettingMC(data.result);
      if (data.type === "roiComparison") setRoiComp(data.result);
      if (data.type === "nextSeason")    setNextSeason(data.result);
      if (++done === total) { setSimRunning(false); worker.terminate(); }
    };

    worker.postMessage({ type: "monteCarlo",    payload: { results, upcoming, btRatings, btBase, btHomeAdv } });
    worker.postMessage({ type: "bettingMC",     payload: { historicalOdds, btRatings, btBase, btHomeAdv, simBk, simOdds, simEdge } });
    worker.postMessage({ type: "roiComparison", payload: { historicalOdds, btRatings, btBase, btHomeAdv, simBk } });
    worker.postMessage({ type: "nextSeason",    payload: { btRatings, btBase, btHomeAdv } });
  }, [results, upcoming, historicalOdds, btRatings, btBase, btHomeAdv, simBk, simOdds, simEdge]);

  const runGridSearch = useCallback(() => {
    let best = null;
    for (let k = 4; k <= 64; k += 4) {
      for (const hca of [0, 25, 50, 75, 100, 125, 150]) {
        for (const decay of [0, 0.01, 0.02, 0.03]) {
          for (const movK of [1.5, 2.2, 3.0]) {
            const res = runWalkForward(results, k, btBase, hca, decay, movK);
            const score = r => parseFloat(r.accuracy) / 100 - parseFloat(r.logLoss);
            if (res && (!best || score(res) > score(best))) {
              best = { k, hca, decay, movK, ...res };
            }
          }
        }
      }
    }
    if (best) { setBtK(best.k); setBtHomeAdv(best.hca); setBtDecay(best.decay); setBtMovK(best.movK); setGsBest(best); }
  }, [results, btBase]);

  useEffect(() => {
    // Load results
    fetch("/results.json")
      .then(r => r.json())
      .then(data => {
        const { ratings: r, history: h, historyDated: hd, changes: c } = computeElo(data);
        setResults(data); setRatings(r); setHistory(h); setHistoryDated(hd); setChanges(c);
        setLoading(false);
      })
      .catch(() => {
        const res = generateSampleResults();
        const { ratings: r, history: h, changes: c } = computeElo(res);
        setResults(res); setRatings(r); setHistory(h); setChanges(c);
        setLoading(false);
      });

    // Load upcoming matches
    fetch("/upcoming.json")
      .then(r => {
        const lm = r.headers.get("Last-Modified");
        if (lm) setLastUpdated(new Date(lm));
        return r.json();
      })
      .then(data => setUpcoming(data))
      .catch(() => setUpcoming(getUpcoming()));

    // Load full odds history
    fetch("/odds_history.json")
      .then(r => r.ok ? r.json() : [])
      .then(hist => { if (hist.length) setOddsHistory(hist); })
      .catch(() => {});

    // Load historical odds
    fetch("/historical_odds.json")
      .then(r => r.ok ? r.json() : [])
      .then(data => { if (data.length) setHistoricalOdds(data); })
      .catch(() => {});
  }, []);

  const ranked = Object.entries(ratings)
    .sort((a, b) => b[1] - a[1])
    .map(([team, elo], i) => {
      const teamResults = results.filter(r => {
        const h = r.HomeTeam ?? r.home ?? r.Home ?? "";
        const a = r.AwayTeam ?? r.away ?? r.Away ?? "";
        return h === team || a === team;
      });
      const wins = teamResults.filter(r => {
        const h = r.HomeTeam ?? r.home ?? r.Home ?? "";
        const hs = r.HomeScore ?? r.homeScore ?? 0;
        const as_ = r.AwayScore ?? r.awayScore ?? 0;
        return (h === team && hs > as_) || (h !== team && as_ > hs);
      }).length;
      const last5 = teamResults.slice(-5);
      const form = last5.map(r => {
        const h = r.HomeTeam ?? r.home ?? r.Home ?? "";
        const hs = r.HomeScore ?? r.homeScore ?? 0;
        const as_ = r.AwayScore ?? r.awayScore ?? 0;
        return (h === team && hs > as_) || (h !== team && as_ > hs) ? "W" : "L";
      });
      const hist = history[team] ?? [];
      const prev = hist.length > 1 ? hist[hist.length - 2] : elo;
      return { team, elo, rank: i + 1, wins, gp: teamResults.length, losses: teamResults.length - wins, form, diff: elo - prev, hist };
    });

  const recentResults = [...results].reverse().slice(0, 20);

  const s = {
    wrap: { maxWidth: 960, margin: "0 auto", fontFamily: "system-ui, sans-serif", color: "#111" },
    header: { padding: "20px 24px 0", borderBottom: "1px solid #eee", background: "#fff" },
    title: { fontSize: 24, fontWeight: 600, margin: "0 0 4px", textAlign: "center" },
    sub: { fontSize: 13, color: "#888", margin: "0 0 14px", textAlign: "center" },
    tabs: { display: "flex", gap: 0 },
    tab: (active, key) => ({
      padding: "10px 20px", fontSize: 13, fontWeight: active ? 600 : 400, cursor: "pointer",
      border: "none", background: "transparent",
      borderBottom: active ? `2px solid ${TAB_COLORS[key]}` : "2px solid transparent",
      color: active ? TAB_COLORS[key] : "#888", transition: "all .15s"
    }),
    body: { padding: "20px 24px", background: "#f9f9f9", minHeight: "80vh" },
    table: { width: "100%", borderCollapse: "collapse", fontSize: 13, background: "#fff", borderRadius: 8, overflow: "hidden" },
    th: { textAlign: "left", padding: "10px 10px", color: "#888", fontWeight: 500, borderBottom: "1px solid #eee", fontSize: 12, background: "#fafafa" },
    td: { padding: "10px 10px", borderBottom: "1px solid #f0f0f0", verticalAlign: "middle" },
    badge: (win) => ({
      display: "inline-block", width: 18, height: 18, borderRadius: 3, fontSize: 10, fontWeight: 600,
      textAlign: "center", lineHeight: "18px", marginRight: 2,
      background: win ? "#EAF3DE" : "#FCEBEB", color: win ? "#3B6D11" : "#A32D2D"
    }),
    card: { background: "#fff", border: "1px solid #eee", borderRadius: 10, padding: "16px 18px", marginBottom: 12, boxShadow: "0 1px 3px rgba(0,0,0,.05)" },
    diff: (d) => ({ fontSize: 11, color: d > 0 ? "#3B6D11" : d < 0 ? "#A32D2D" : "#888", fontWeight: 600 }),
    teamBtn: (sel) => ({ background: "none", border: "none", cursor: "pointer", textAlign: "left", fontWeight: sel ? 600 : 400, color: sel ? "#185FA5" : "#111", padding: 0, fontSize: 13 }),
  };

  if (loading) return <div style={{ padding: 60, textAlign: "center", color: "#888" }}>Načítám ELO data…</div>;

  return (
    <div style={s.wrap}>
      <div style={s.header}>
        <h1 style={s.title}>ČNBL ELO Dashboard</h1>
        <p style={s.sub}>Czech Basketball League · 538-style ELO · K={btK} · HCA={btHomeAdv} · Base={btBase} · Decay={btDecay > 0 ? btDecay.toFixed(3) : "off"} · MoV={btMovK.toFixed(1)}</p>
        <div style={s.tabs}>
          {[["leaderboard", "Leaderboard"], ["predictions", "Predictions"], ["results", "Results"], ["backtest", "Backtest"], ["oddshistory", "Odds History"], ["historical", "Historické kurzy"], ["simulace", "Monte Carlo"]].map(([k, l]) => (
            <button key={k} style={s.tab(tab === k, k)} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
      </div>

      <div style={s.body}>

        {/* LEADERBOARD */}
        {tab === "leaderboard" && (
          <>
            <table style={s.table}>
              <thead>
                <tr>
                  {["#", "Tým", "ELO", "+/−", "V", "P", "Forma", "Trend"].map(h => <th key={h} style={s.th}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {ranked.map(r => (
                  <tr key={r.team} style={{ background: selectedTeam === r.team ? "#f0f7ff" : "#fff" }}>
                    <td style={{ ...s.td, color: "#aaa", width: 28 }}>{r.rank}</td>
                    <td style={s.td}>
                      <button style={s.teamBtn(selectedTeam === r.team)} onClick={() => setSelectedTeam(selectedTeam === r.team ? null : r.team)}>
                        {r.team}
                      </button>
                    </td>
                    <td style={{ ...s.td, fontWeight: 600 }}>{r.elo}</td>
                    <td style={s.td}><span style={s.diff(r.diff)}>{r.diff > 0 ? "+" : ""}{r.diff}</span></td>
                    <td style={{ ...s.td, color: "#3B6D11", fontWeight: 500 }}>{r.wins}</td>
                    <td style={{ ...s.td, color: "#A32D2D", fontWeight: 500 }}>{r.losses}</td>
                    <td style={s.td}>{r.form.map((f, i) => <span key={i} style={s.badge(f === "W")}>{f}</span>)}</td>
                    <td style={s.td}><MiniChart data={r.hist.slice(-15)} color={r.diff >= 0 ? "#1D9E75" : "#D85A30"} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {/* ELO history chart for selected team */}
            {selectedTeam && (() => {
              const pts = historyDated[selectedTeam] ?? [];
              if (pts.length < 2) return null;
              const teamRank = ranked.find(r => r.team === selectedTeam);
              const color = teamRank?.diff >= 0 ? "#1D9E75" : "#D85A30";

              const W = 600, H = 180, pL = 50, pR = 16, pT = 14, pB = 30;
              const pw = W - pL - pR, ph = H - pT - pB;

              const elos = pts.map(p => p.elo);
              const minE = Math.min(...elos) - 20, maxE = Math.max(...elos) + 20;
              const eRange = maxE - minE;

              const xOf = i => pL + (i / (pts.length - 1)) * pw;
              const yOf = v => pT + ((maxE - v) / eRange) * ph;

              // Y ticks: ~4 evenly spaced
              const rawStep = eRange / 4;
              const step = Math.ceil(rawStep / 10) * 10;
              const yTicks = [];
              for (let v = Math.ceil(minE / step) * step; v <= maxE; v += step) yTicks.push(v);

              // X ticks: show up to 6 date labels
              const xStep = Math.ceil(pts.length / 6);
              const xTicks = pts.filter((_, i) => i % xStep === 0 || i === pts.length - 1);

              const polyPts = pts.map((p, i) => `${xOf(i).toFixed(1)},${yOf(p.elo).toFixed(1)}`).join(" ");
              const baseY = yOf(BASE_ELO);

              return (
                <div style={{ ...s.card, marginTop: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{selectedTeam}</span>
                    <span style={{ fontSize: 11, color: "#aaa" }}>{pts.length - 1} zapasu · ELO {pts[pts.length - 1].elo}</span>
                  </div>
                  <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
                    {/* Grid + Y labels */}
                    {yTicks.map(v => (
                      <g key={v}>
                        <line x1={pL} y1={yOf(v)} x2={W - pR} y2={yOf(v)} stroke="#f4f4f4" strokeWidth="1" />
                        <text x={pL - 4} y={yOf(v) + 3.5} textAnchor="end" fontSize="9" fill="#ccc">{v}</text>
                      </g>
                    ))}
                    {/* Base ELO reference */}
                    {baseY >= pT && baseY <= H - pB && (
                      <line x1={pL} y1={baseY} x2={W - pR} y2={baseY} stroke="#ddd" strokeWidth="1" strokeDasharray="4,3" />
                    )}
                    {/* ELO line */}
                    <polyline points={polyPts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
                    {/* End dot */}
                    <circle cx={xOf(pts.length - 1)} cy={yOf(pts[pts.length - 1].elo)} r="3.5" fill={color} />
                    {/* X labels */}
                    {xTicks.map((p, i) => p.date && (
                      <text key={i} x={xOf(pts.indexOf(p))} y={H - 4} textAnchor="middle" fontSize="8" fill="#ccc">
                        {p.date.slice(5)}
                      </text>
                    ))}
                    {/* Axes */}
                    <line x1={pL} y1={pT} x2={pL} y2={H - pB} stroke="#e8e8e8" strokeWidth="1" />
                    <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke="#e8e8e8" strokeWidth="1" />
                  </svg>
                </div>
              );
            })()}
          </>
        )}

        {/* PREDICTIONS */}
        {tab === "predictions" && (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <p style={{ fontSize: 13, color: "#888", margin: 0 }}>FP z ELO · </p>
              <button
                onClick={triggerScrape}
                disabled={scrapeStatus === "running"}
                style={{
                  padding: "5px 14px", fontSize: 12, fontWeight: 600, borderRadius: 6, border: "none",
                  cursor: scrapeStatus === "running" ? "default" : "pointer",
                  background: scrapeStatus === "ok" ? "#1D9E75" : scrapeStatus === "error" ? "#C0392B" : "#378ADD",
                  color: "#fff", opacity: scrapeStatus === "running" ? 0.7 : 1,
                }}
              >
                {scrapeStatus === "running" ? "⏳ Spouštím…" : scrapeStatus === "ok" ? "✓ Spuštěno" : scrapeStatus === "error" ? "✗ Chyba" : "▶ Aktualizovat data"}
              </button>
              {scrapeStatus === "ok" && <span style={{ fontSize: 12, color: "#888" }}>Scraper běží na GitHubu, data budou za ~3 min.</span>}
              {lastUpdated && scrapeStatus !== "ok" && (
                <span style={{ fontSize: 12, color: "#aaa" }}>
                  Aktualizováno: {lastUpdated.toLocaleDateString("cs-CZ")} {lastUpdated.toLocaleTimeString("cs-CZ", { hour: "2-digit", minute: "2-digit" })}
                </span>
              )}
            </div>
            {upcoming.length === 0 && (
              <p style={{ color: "#aaa", fontSize: 13 }}>Žádné nadcházející zápasy. Spusť scraper pro aktualizaci.</p>
            )}
            {upcoming.filter(m => {
              const d = m.Date ?? m.date ?? "";
              return !d || d >= new Date().toISOString().slice(0, 10);
            }).map((m, i) => {
              const home = m.HomeTeam ?? m.home ?? "";
              const away = m.AwayTeam ?? m.away ?? "";
              const date = m.Date ?? m.date ?? "";
              const eloH = btRatings[home] ?? btBase;
              const eloA = btRatings[away] ?? btBase;
              const pRaw = expectedScore(eloH + btHomeAdv, eloA);
              const hp = Math.round(pRaw * 100);
              const ap = 100 - hp;
              const { oddsH, oddsA } = fairOdds(pRaw * 100);
              return (
                <div key={i} style={s.card}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "#aaa" }}>{date}{m.Time ? ` · ${m.Time}` : ""}</span>
                    <span style={{ fontSize: 11, background: "#E6F1FB", color: "#185FA5", borderRadius: 4, padding: "2px 8px", fontWeight: 600 }}>
                      {hp > ap ? home : away} favorizován
                    </span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                    <span style={{ flex: 1, fontWeight: 600, fontSize: 14 }}>{home}</span>
                    <span style={{ color: "#ccc", fontSize: 12 }}>vs</span>
                    <span style={{ flex: 1, fontWeight: 600, fontSize: 14, textAlign: "right" }}>{away}</span>
                  </div>
                  <div style={{ display: "flex", borderRadius: 4, overflow: "hidden", height: 8, marginBottom: 10 }}>
                    <div style={{ flex: hp, background: "#378ADD" }} />
                    <div style={{ flex: ap, background: "#D85A30" }} />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                    <div>
                      <span style={{ color: "#185FA5", fontWeight: 600 }}>{hp}%</span>
                      <span style={{ color: "#aaa", margin: "0 6px" }}>ELO {eloH}</span>
                      <span style={{ background: "#E6F1FB", color: "#185FA5", borderRadius: 4, padding: "1px 8px", fontWeight: 700 }}>
                        {oddsH}
                      </span>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <span style={{ background: "#FCEBEB", color: "#993C1D", borderRadius: 4, padding: "1px 8px", fontWeight: 700 }}>
                        {oddsA}
                      </span>
                      <span style={{ color: "#aaa", margin: "0 6px" }}>ELO {eloA}</span>
                      <span style={{ color: "#993C1D", fontWeight: 600 }}>{ap}%</span>
                    </div>
                  </div>
                  {(() => {
                    const mo = findMatchOdds(home, away, latestOdds, date);
                    if (!mo || !Object.keys(mo.Bookmakers ?? {}).length) return null;
                    const openMatch = getOpeningOdds(oddsHistory, home, away, date);
                    const hasOpen = openMatch && Object.keys(openMatch.Bookmakers ?? {}).length > 0;
                    const thSt = (center) => ({ textAlign: center ? "center" : "left", color: "#bbb", fontWeight: 500, paddingBottom: 3, fontSize: 10 });
                    // Shoda: trh se pohnul směrem k ELO fair hodnotě
                    // ELO fair=1.28, open=1.16→close=1.15: closing dál od ELO → ✗
                    // ELO fair=1.28, open=1.16→close=1.22: closing blíž k ELO → ✓
                    const eloFairH = pRaw > 0 ? 1 / pRaw : null;
                    const agreement = (openH, closeH) => {
                      if (!openH || !closeH || eloFairH === null || Math.abs(closeH - openH) < 0.02) return null;
                      return Math.abs(closeH - eloFairH) < Math.abs(openH - eloFairH);
                    };
                    return (
                      <div style={{ borderTop: "1px solid #f0f0f0", paddingTop: 8, marginTop: 8 }}>
                        <div style={{ color: "#aaa", marginBottom: 6, fontSize: 11 }}>
                          Bookmaker kurzy{hasOpen && <span style={{ color: "#B07A10", marginLeft: 6 }}>· opening → closing · ✓/✗ = trh jde směrem k ELO fair kurzu</span>}
                        </div>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                          <thead>
                            <tr>
                              <th style={thSt(false)}>Bookmaker</th>
                              <th style={thSt(true)}>Domácí</th>
                              <th style={thSt(true)}>Hosté</th>
                              {hasOpen && <th style={thSt(true)}>Shoda</th>}
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(mo.Bookmakers).map(([name, o]) => {
                              const openBk = hasOpen ? openMatch.Bookmakers[name] : null;
                              const movedH = openBk?.HomeOdds && openBk.HomeOdds !== o.HomeOdds;
                              const movedA = openBk?.AwayOdds && openBk.AwayOdds !== o.AwayOdds;
                              const agree = agreement(openBk?.HomeOdds, o.HomeOdds);
                              return (
                                <tr key={name}>
                                  <td style={{ padding: "3px 0", color: "#555" }}>{name}</td>
                                  <td style={{ textAlign: "center" }}>
                                    {movedH && <span style={{ color: "#aaa", fontSize: 10, marginRight: 3 }}>{openBk.HomeOdds} →</span>}
                                    <b style={{ color: "#185FA5" }}>{o.HomeOdds}</b>
                                  </td>
                                  <td style={{ textAlign: "center" }}>
                                    {movedA && <span style={{ color: "#aaa", fontSize: 10, marginRight: 3 }}>{openBk.AwayOdds} →</span>}
                                    <b style={{ color: "#993C1D" }}>{o.AwayOdds}</b>
                                  </td>
                                  {hasOpen && (
                                    <td style={{ textAlign: "center", fontWeight: 700, fontSize: 14, color: agree === true ? "#1D7F3A" : agree === false ? "#A32D2D" : "#ccc" }}>
                                      {agree === true ? "✓" : agree === false ? "✗" : "—"}
                                    </td>
                                  )}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    );
                  })()}
                </div>
              );
            })}
            <p style={{ fontSize: 11, color: "#bbb", marginTop: 8 }}>
              Fair odds = 1/pravděpodobnost. Bookmaker kurzy budou vždy nižší kvůli marži.
            </p>
          </>
        )}

        {/* RESULTS */}
        {tab === "results" && (
          <>
            <p style={{ fontSize: 13, color: "#888", marginBottom: 16 }}>Posledních 20 výsledků · ELO posun po každém zápase</p>
            <table style={s.table}>
              <thead>
                <tr>{["Datum", "Domácí", "Skóre", "Hosté", "ELO posun"].map(h => <th key={h} style={s.th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {recentResults.map((r, i) => {
                  const home = r.HomeTeam ?? r.home ?? r.Home ?? "";
                  const away = r.AwayTeam ?? r.away ?? r.Away ?? "";
                  const hs = r.HomeScore ?? r.homeScore ?? 0;
                  const as_ = r.AwayScore ?? r.awayScore ?? 0;
                  const date = r.Date ?? r.date ?? "";
                  const hw = hs > as_;
                  const idx = results.length - 1 - i;
                  const delta = changes[idx] ?? 0;
                  const homeChange = delta >= 0 ? `+${delta}` : `${delta}`;
                  const awayChange = delta >= 0 ? `−${delta}` : `+${Math.abs(delta)}`;
                  return (
                    <tr key={i}>
                      <td style={{ ...s.td, color: "#aaa", fontSize: 12 }}>{date}</td>
                      <td style={{ ...s.td, fontWeight: hw ? 600 : 400 }}>{home}</td>
                      <td style={{ ...s.td, textAlign: "center", fontWeight: 600 }}>
                        <span style={{ color: hw ? "#3B6D11" : "#A32D2D" }}>{hs}</span>
                        <span style={{ color: "#ccc", margin: "0 6px" }}>–</span>
                        <span style={{ color: !hw ? "#3B6D11" : "#A32D2D" }}>{as_}</span>
                      </td>
                      <td style={{ ...s.td, fontWeight: !hw ? 600 : 400 }}>{away}</td>
                      <td style={{ ...s.td, fontSize: 12 }}>
                        <span style={{ color: delta >= 0 ? "#3B6D11" : "#A32D2D" }}>{homeChange}</span>
                        {" / "}
                        <span style={{ color: delta >= 0 ? "#A32D2D" : "#3B6D11" }}>{awayChange}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}

        {/* BACKTEST */}
        {tab === "backtest" && (
          <>
            {/* Sliders + Grid Search */}
            <div style={{ background: "#fff", borderRadius: 8, padding: "20px 18px", marginBottom: 12, boxShadow: "0 1px 3px rgba(0,0,0,.05)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: "#555" }}>Parametry modelu · walk-forward ({btResult?.foldCount ?? 0} folds)</span>
                <button onClick={runGridSearch} style={{
                  background: "#7C52C8", color: "#fff", border: "none", borderRadius: 6,
                  padding: "6px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer"
                }}>
                  Grid Search
                </button>
              </div>
              {gsBest && (
                <div style={{ background: "#F3EEFF", borderRadius: 6, padding: "8px 12px", marginBottom: 16, fontSize: 12, color: "#5B3FA8" }}>
                  Nejlepší: K={gsBest.k} · HCA={gsBest.hca} · Decay={gsBest.decay} · MoV={gsBest.movK} · LogLoss={gsBest.logLoss} · Acc={gsBest.accuracy}%
                </div>
              )}
              {[
                { label: "K faktor", value: btK, set: setBtK, min: 4, max: 64, step: 4, fmt: v => v },
                { label: "Základní ELO", value: btBase, set: setBtBase, min: 500, max: 2000, step: 100, fmt: v => v },
                { label: "Výhoda domácího hřiště (HCA)", value: btHomeAdv, set: setBtHomeAdv, min: 0, max: 200, step: 25, fmt: v => v },
                { label: "Decay (staré zápasy méně důležité)", value: btDecay, set: setBtDecay, min: 0, max: 0.05, step: 0.005, fmt: v => v === 0 ? "vypnuto" : v.toFixed(3) },
                { label: "MoV citlivost (vliv rozdílu skóre)", value: btMovK, set: setBtMovK, min: 0.5, max: 5, step: 0.5, fmt: v => v.toFixed(1) },
              ].map(({ label, value, set, min, max, step, fmt }) => (
                <div key={label} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>{label}</span>
                    <span style={{ fontSize: 12, color: "#7C52C8", fontWeight: 700 }}>{fmt(value)}</span>
                  </div>
                  <input type="range" min={min} max={max} step={step} value={value}
                    onChange={e => set(+e.target.value)}
                    style={{ width: "100%", accentColor: "#7C52C8" }} />
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#ccc" }}>
                    <span>{fmt(min)}</span><span>{fmt(max)}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Metrics + baselines */}
            {btResult ? (
              <>
                <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                  {[
                    { label: "Přesnost", value: btResult.accuracy + "%", color: "#3B6D11", sub: `${btResult.totalGames} zápasů` },
                    { label: "Brier Score", value: btResult.brier, color: "#7C52C8", sub: "nižší = lepší" },
                    { label: "Log Loss", value: btResult.logLoss, color: "#D85A30", sub: "nižší = lepší" },
                  ].map(({ label, value, color, sub }) => (
                    <div key={label} style={{ flex: 1, background: "#fff", borderRadius: 8, padding: "14px 10px", boxShadow: "0 1px 3px rgba(0,0,0,.05)", textAlign: "center" }}>
                      <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 24, fontWeight: 700, color }}>{value}</div>
                      <div style={{ fontSize: 10, color: "#bbb", marginTop: 3 }}>{sub}</div>
                    </div>
                  ))}
                </div>
                <div style={{ background: "#fff", borderRadius: 8, marginBottom: 16, boxShadow: "0 1px 3px rgba(0,0,0,.05)", overflow: "hidden" }}>
                  <table style={{ ...s.table, fontSize: 12 }}>
                    <thead>
                      <tr>
                        {["Model", "Přesnost", "Brier Score"].map(h => <th key={h} style={s.th}>{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: "ELO (náš model)", acc: btResult.accuracy + "%", brier: btResult.brier, highlight: true },
                        { label: "Vždy domácí (p=1)", acc: btResult.baselines.homeAlways.accuracy + "%", brier: btResult.baselines.homeAlways.brier, highlight: false },
                        { label: "Míra domácích (z tréninku)", acc: btResult.baselines.homeRate.accuracy + "%", brier: btResult.baselines.homeRate.brier, highlight: false },
                        { label: "Náhoda (p=0.5)", acc: "50.0%", brier: "0.2500", highlight: false },
                      ].map(({ label, acc, brier, highlight }) => (
                        <tr key={label} style={{ background: highlight ? "#F3EEFF" : "#fff" }}>
                          <td style={{ ...s.td, fontWeight: highlight ? 600 : 400, color: highlight ? "#5B3FA8" : "#111" }}>{label}</td>
                          <td style={{ ...s.td, fontWeight: 600 }}>{acc}</td>
                          <td style={{ ...s.td, fontWeight: 600 }}>{brier}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p style={{ fontSize: 13, color: "#aaa", textAlign: "center", marginBottom: 16 }}>Nedostatek dat pro backtesting.</p>
            )}

            {/* Calibration + P&L simulation */}
            {btPreds.length > 0 && (
              <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>

                {/* Calibration */}
                <div style={{ flex: 1, minWidth: 280, background: "#fff", borderRadius: 8, padding: "14px 16px", boxShadow: "0 1px 3px rgba(0,0,0,.05)" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#555", marginBottom: 10 }}>Kalibrace — když model řekne X%, jak často to vyjde?</div>
                  {(() => {
                    const filled = btCalib.filter(b => b.actual);
                    if (!filled.length) return null;
                    const W = 240, H = 200, pad = 32;
                    const plot = W - pad * 2;
                    // axis: 50–100%
                    const sc = v => pad + ((v - 50) / 50) * plot;
                    const pts = filled.map(b => ({
                      x: sc(parseFloat(b.pred)),
                      y: W - pad - ((parseFloat(b.actual) - 50) / 50) * plot,
                      delta: parseFloat(b.actual) - parseFloat(b.pred),
                      label: b.label, n: b.n,
                    }));
                    const tickVals = [50, 60, 70, 80, 90, 100];
                    return (
                      <svg viewBox={`0 0 ${W} ${W}`} style={{ width: "100%", height: "auto", display: "block", marginBottom: 10 }}>
                        {/* Grid */}
                        {tickVals.map(v => (
                          <g key={v}>
                            <line x1={sc(v)} y1={pad} x2={sc(v)} y2={W - pad} stroke="#f0f0f0" strokeWidth="1" />
                            <line x1={pad} y1={W - pad - ((v - 50) / 50) * plot} x2={W - pad} y2={W - pad - ((v - 50) / 50) * plot} stroke="#f0f0f0" strokeWidth="1" />
                            <text x={sc(v)} y={W - pad + 12} textAnchor="middle" fontSize="8" fill="#bbb">{v}%</text>
                            <text x={pad - 4} y={W - pad - ((v - 50) / 50) * plot + 3} textAnchor="end" fontSize="8" fill="#bbb">{v}%</text>
                          </g>
                        ))}
                        {/* Perfect calibration diagonal */}
                        <line x1={sc(50)} y1={W - pad} x2={sc(100)} y2={pad} stroke="#ccc" strokeWidth="1" strokeDasharray="4,3" />
                        <text x={sc(78)} y={W - pad - ((78 - 50) / 50) * plot - 5} fontSize="8" fill="#bbb" textAnchor="middle">ideál</text>
                        {/* Model points connected by line */}
                        {pts.length > 1 && (
                          <polyline points={pts.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#7C52C8" strokeWidth="1.5" strokeLinejoin="round" opacity="0.5" />
                        )}
                        {pts.map((p, i) => (
                          <g key={i}>
                            <circle cx={p.x} cy={p.y} r={Math.min(8, Math.max(4, p.n / 3))}
                              fill={p.delta > 3 ? "#1D7F3A" : p.delta < -3 ? "#A32D2D" : "#7C52C8"}
                              opacity="0.85" />
                            <text x={p.x} y={p.y - 10} textAnchor="middle" fontSize="8" fill="#555">{p.n}</text>
                          </g>
                        ))}
                        {/* Axis labels */}
                        <text x={W / 2} y={W - 2} textAnchor="middle" fontSize="9" fill="#aaa">Predikce</text>
                        <text x={8} y={W / 2} textAnchor="middle" fontSize="9" fill="#aaa" transform={`rotate(-90, 8, ${W / 2})`}>Skutecnost</text>
                        <line x1={pad} y1={pad} x2={pad} y2={W - pad} stroke="#e0e0e0" strokeWidth="1" />
                        <line x1={pad} y1={W - pad} x2={W - pad} y2={W - pad} stroke="#e0e0e0" strokeWidth="1" />
                      </svg>
                    );
                  })()}
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                      <tr>{["Jistota", "N", "Pred.", "Skut.", "Δ"].map(h => <th key={h} style={{ ...s.th, padding: "3px 4px" }}>{h}</th>)}</tr>
                    </thead>
                    <tbody>
                      {btCalib.map(b => {
                        if (!b.actual) return null;
                        const delta = (parseFloat(b.actual) - parseFloat(b.pred)).toFixed(1);
                        const dColor = parseFloat(delta) > 3 ? "#1D7F3A" : parseFloat(delta) < -3 ? "#A32D2D" : "#888";
                        return (
                          <tr key={b.label}>
                            <td style={{ ...s.td, padding: "3px 4px" }}>{b.label}</td>
                            <td style={{ ...s.td, padding: "3px 4px", color: "#aaa" }}>{b.n}</td>
                            <td style={{ ...s.td, padding: "3px 4px" }}>{b.pred}%</td>
                            <td style={{ ...s.td, padding: "3px 4px", fontWeight: 600 }}>{b.actual}%</td>
                            <td style={{ ...s.td, padding: "3px 4px", fontWeight: 600, color: dColor }}>{parseFloat(delta) > 0 ? "+" : ""}{delta}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

              </div>
            )}

            {/* Live preview */}
            <div style={{ fontSize: 12, fontWeight: 600, color: "#7C52C8", marginBottom: 10, letterSpacing: "0.03em" }}>
              ŽIVÝ NÁHLED · K={btK} · Base={btBase} · HCA={btHomeAdv}
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>

              {/* Rankings */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <table style={{ ...s.table, fontSize: 12 }}>
                  <thead>
                    <tr>{["#", "Tým", "ELO", "vs. orig."].map(h => <th key={h} style={s.th}>{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {btRanked.map(r => {
                      const origRank = ranked.findIndex(x => x.team === r.team) + 1;
                      const shift = origRank - r.rank;
                      return (
                        <tr key={r.team}>
                          <td style={{ ...s.td, color: "#aaa", width: 22 }}>{r.rank}</td>
                          <td style={{ ...s.td, fontSize: 11 }}>{r.team}</td>
                          <td style={{ ...s.td, fontWeight: 600, color: "#7C52C8" }}>{r.elo}</td>
                          <td style={{ ...s.td, fontSize: 11 }}>
                            {shift > 0
                              ? <span style={{ color: "#3B6D11" }}>▲{shift}</span>
                              : shift < 0
                                ? <span style={{ color: "#A32D2D" }}>▼{Math.abs(shift)}</span>
                                : <span style={{ color: "#bbb" }}>—</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Predictions */}
              <div style={{ flex: 1, minWidth: 0 }}>
                {upcoming.map((m, i) => {
                  const home = m.HomeTeam ?? m.home ?? "";
                  const away = m.AwayTeam ?? m.away ?? "";
                  const eloH = btRatings[home] ?? btBase;
                  const eloA = btRatings[away] ?? btBase;
                  const pRaw = expectedScore(eloH + btHomeAdv, eloA);
                  const hp = Math.round(pRaw * 100);
                  const ap = 100 - hp;
                  const { oddsH, oddsA } = fairOdds(pRaw * 100);
                  return (
                    <div key={i} style={{ ...s.card, padding: "12px 14px", marginBottom: 10 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                        <span style={{ flex: 1, fontWeight: 600, fontSize: 12 }}>{home}</span>
                        <span style={{ color: "#ccc", fontSize: 10 }}>vs</span>
                        <span style={{ flex: 1, fontWeight: 600, fontSize: 12, textAlign: "right" }}>{away}</span>
                      </div>
                      <div style={{ display: "flex", borderRadius: 3, overflow: "hidden", height: 6, marginBottom: 6 }}>
                        <div style={{ flex: hp, background: "#378ADD" }} />
                        <div style={{ flex: ap, background: "#D85A30" }} />
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                        <span style={{ color: "#185FA5", fontWeight: 600 }}>{hp}% <span style={{ color: "#aaa", fontWeight: 400, fontSize: 10 }}>{oddsH}</span></span>
                        <span style={{ color: "#993C1D", fontWeight: 600 }}>{ap}% <span style={{ color: "#aaa", fontWeight: 400, fontSize: 10 }}>{oddsA}</span></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {/* ODDS HISTORY */}
        {tab === "oddshistory" && (() => {
          if (!oddsHistory.length) return (
            <p style={{ color: "#aaa", fontSize: 13 }}>Žádná historia kurzů. Spusť scraper pro aktualizaci.</p>
          );

          const BK_COLORS = { Tipsport: "#0066CC", Fortuna: "#CC2200", Betano: "#7B2FBE", Chance: "#E07A00" };
          const bkColor = name => BK_COLORS[name] ?? "#888";

          // Collect unique matches
          const matchKeys = [];
          const seenKeys = new Set();
          for (const snap of oddsHistory) {
            for (const m of snap.Matches ?? []) {
              const key = `${m.HomeTeam}__${m.AwayTeam}`;
              if (!seenKeys.has(key)) { seenKeys.add(key); matchKeys.push({ key, home: m.HomeTeam, away: m.AwayTeam }); }
            }
          }

          const activeKey = selectedMatchKey ?? matchKeys[0]?.key ?? null;
          const activeMatch = matchKeys.find(m => m.key === activeKey);

          // Build time series for active match
          const series = activeMatch ? oddsHistory
            .map(snap => {
              const m = (snap.Matches ?? []).find(m =>
                teamsMatch(activeMatch.home, m.HomeTeam) && teamsMatch(activeMatch.away, m.AwayTeam)
              );
              if (!m || !Object.keys(m.Bookmakers ?? {}).length) return null;
              return { at: new Date(snap.ScrapedAt), bk: m.Bookmakers };
            })
            .filter(Boolean) : [];

          const bkNames = series.length
            ? [...new Set(series.flatMap(p => Object.keys(p.bk)))]
            : [];

          // ELO fair odds for active match
          const eloFair = activeMatch ? (() => {
            const eH = btRatings[activeMatch.home] ?? btRatings[Object.keys(btRatings).find(t => teamsMatch(t, activeMatch.home)) ?? ""] ?? btBase;
            const eA = btRatings[activeMatch.away] ?? btRatings[Object.keys(btRatings).find(t => teamsMatch(t, activeMatch.away)) ?? ""] ?? btBase;
            const p = expectedScore(eH + btHomeAdv, eA);
            return { fairH: (1 / p).toFixed(2), fairA: (1 / (1 - p)).toFixed(2), pct: Math.round(p * 100) };
          })() : null;

          // SVG chart
          const chart = series.length >= 1 ? (() => {
            const W = 600, H = 200, pL = 44, pR = 12, pT = 12, pB = 38;
            const pw = W - pL - pR, ph = H - pT - pB;

            const allOdds = series.flatMap(p =>
              Object.values(p.bk).flatMap(o => [o.HomeOdds, o.AwayOdds].filter(Boolean))
            );
            const minY = Math.max(1, Math.floor(Math.min(...allOdds) * 10) / 10 - 0.1);
            const maxY = Math.ceil(Math.max(...allOdds) * 10) / 10 + 0.15;
            const yRange = maxY - minY || 1;

            const times = series.map(p => p.at.getTime());
            const minT = Math.min(...times), maxT = Math.max(...times);
            const tRange = maxT - minT || 1;

            const xOf = t => pL + ((t - minT) / tRange) * pw;
            const yOf = v => pT + ((maxY - v) / yRange) * ph;

            // Y ticks
            const yStep = yRange > 2 ? 0.5 : yRange > 1 ? 0.2 : 0.1;
            const yTicks = [];
            for (let v = Math.ceil(minY / yStep) * yStep; v <= maxY + 0.001; v = Math.round((v + yStep) * 100) / 100) yTicks.push(v);

            // X ticks (max 5)
            const step = Math.ceil(series.length / 5);
            const xTicks = series.filter((_, i) => i % step === 0 || i === series.length - 1);

            // Fair odds horizontal lines
            const fairLines = eloFair ? [
              { v: parseFloat(eloFair.fairH), color: "#185FA5", label: `ELO ${activeMatch.home.split(" ").pop()}` },
              { v: parseFloat(eloFair.fairA), color: "#993C1D", label: `ELO ${activeMatch.away.split(" ").pop()}` },
            ].filter(f => f.v >= minY && f.v <= maxY) : [];

            return (
              <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
                {/* Grid + Y labels */}
                {yTicks.map(v => (
                  <g key={v}>
                    <line x1={pL} y1={yOf(v)} x2={W - pR} y2={yOf(v)} stroke="#f0f0f0" strokeWidth="1" />
                    <text x={pL - 4} y={yOf(v) + 3.5} textAnchor="end" fontSize="9" fill="#bbb">{v.toFixed(2)}</text>
                  </g>
                ))}
                {/* ELO fair odds reference lines */}
                {fairLines.map(f => (
                  <g key={f.label}>
                    <line x1={pL} y1={yOf(f.v)} x2={W - pR} y2={yOf(f.v)} stroke={f.color} strokeWidth="1" strokeDasharray="6,4" opacity="0.4" />
                    <text x={W - pR + 2} y={yOf(f.v) + 3.5} fontSize="8" fill={f.color} opacity="0.7">{f.v.toFixed(2)}</text>
                  </g>
                ))}
                {/* Bookmaker lines */}
                {bkNames.flatMap(bk => {
                  const color = bkColor(bk);
                  const hPts = series.filter(p => p.bk[bk]?.HomeOdds).map(p => `${xOf(p.at.getTime()).toFixed(1)},${yOf(p.bk[bk].HomeOdds).toFixed(1)}`).join(" ");
                  const aPts = series.filter(p => p.bk[bk]?.AwayOdds).map(p => `${xOf(p.at.getTime()).toFixed(1)},${yOf(p.bk[bk].AwayOdds).toFixed(1)}`).join(" ");
                  return [
                    hPts && <polyline key={`${bk}-H`} points={hPts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />,
                    aPts && <polyline key={`${bk}-A`} points={aPts} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="5,3" strokeLinejoin="round" strokeLinecap="round" opacity="0.75" />,
                  ].filter(Boolean);
                })}
                {/* Dots for single-point series */}
                {series.length === 1 && bkNames.flatMap(bk => [
                  <circle key={`${bk}-Hd`} cx={xOf(series[0].at.getTime())} cy={yOf(series[0].bk[bk]?.HomeOdds ?? 0)} r="3" fill={bkColor(bk)} />,
                  <circle key={`${bk}-Ad`} cx={xOf(series[0].at.getTime())} cy={yOf(series[0].bk[bk]?.AwayOdds ?? 0)} r="3" fill={bkColor(bk)} opacity="0.6" />,
                ])}
                {/* X axis labels */}
                {xTicks.map((p, i) => (
                  <text key={i} x={xOf(p.at.getTime())} y={H - 4} textAnchor="middle" fontSize="8" fill="#bbb">
                    {p.at.toLocaleString("cs-CZ", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </text>
                ))}
                {/* Axes */}
                <line x1={pL} y1={pT} x2={pL} y2={H - pB} stroke="#e0e0e0" strokeWidth="1" />
                <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke="#e0e0e0" strokeWidth="1" />
              </svg>
            );
          })() : <p style={{ fontSize: 12, color: "#bbb", textAlign: "center", padding: "20px 0" }}>Žádná data pro tento zápas.</p>;

          return (
            <>
              {/* Match selector */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
                {matchKeys.map(({ key, home, away }) => (
                  <button key={key} onClick={() => setSelectedMatchKey(key)} style={{
                    padding: "5px 12px", fontSize: 12, borderRadius: 20, cursor: "pointer", fontWeight: key === activeKey ? 600 : 400,
                    background: key === activeKey ? "#B07A10" : "#fff",
                    color: key === activeKey ? "#fff" : "#555",
                    border: `1px solid ${key === activeKey ? "#B07A10" : "#ddd"}`,
                  }}>
                    {home.split(" ").pop()} vs {away.split(" ").pop()}
                  </button>
                ))}
              </div>

              {activeMatch && (
                <div style={s.card}>
                  <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 2 }}>
                    {activeMatch.home} <span style={{ color: "#aaa", fontWeight: 400 }}>vs</span> {activeMatch.away}
                  </div>
                  {eloFair && (
                    <div style={{ fontSize: 11, color: "#aaa", marginBottom: 12 }}>
                      ELO fair: <b style={{ color: "#185FA5" }}>{eloFair.fairH}</b> domaci / <b style={{ color: "#993C1D" }}>{eloFair.fairA}</b> hoste
                      &nbsp;·&nbsp;{eloFair.pct}% / {100 - eloFair.pct}%
                      &nbsp;·&nbsp;{series.length} snapshot{series.length !== 1 ? "u" : ""}
                    </div>
                  )}
                  {chart}
                  {/* Opening → Closing summary table */}
                  {series.length >= 2 && bkNames.length > 0 && (() => {
                    const openSnap = series[0];
                    const closeSnap = series[series.length - 1];
                    const thSt = { color: "#bbb", fontWeight: 500, paddingBottom: 3, fontSize: 10, textAlign: "center" };
                    return (
                      <div style={{ marginTop: 14 }}>
                        <div style={{ fontSize: 11, color: "#aaa", marginBottom: 6 }}>
                          Opening ({openSnap.at.toLocaleString("cs-CZ", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })})
                          {" → "}
                          Closing ({closeSnap.at.toLocaleString("cs-CZ", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })})
                        </div>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                          <thead>
                            <tr>
                              <th style={{ ...thSt, textAlign: "left" }}>Bookmaker</th>
                              <th style={thSt}>Open dom.</th>
                              <th style={thSt}>Close dom.</th>
                              <th style={thSt}>Pohyb dom.</th>
                              <th style={thSt}>Open host.</th>
                              <th style={thSt}>Close host.</th>
                              <th style={thSt}>Pohyb host.</th>
                            </tr>
                          </thead>
                          <tbody>
                            {bkNames.map(bk => {
                              const open = openSnap.bk[bk];
                              const close = closeSnap.bk[bk];
                              if (!open && !close) return null;
                              const dH = open?.HomeOdds && close?.HomeOdds ? +(close.HomeOdds - open.HomeOdds).toFixed(2) : null;
                              const dA = open?.AwayOdds && close?.AwayOdds ? +(close.AwayOdds - open.AwayOdds).toFixed(2) : null;
                              const fmt = d => (d > 0 ? "+" : "") + d.toFixed(2);
                              // eloFavorsHome = true if ELO predicts home win for this match
                              const eloFavorsHome = eloFair ? eloFair.pct > 50 : null;
                              // agree: market moved same direction as ELO prediction?
                              // home odds dropping = market favors home. If ELO also favors home → agree.
                              const agreeH = dH !== null && Math.abs(dH) >= 0.02 && eloFavorsHome !== null
                                ? (dH < 0) === eloFavorsHome : null;
                              const agreeA = dA !== null && Math.abs(dA) >= 0.02 && eloFavorsHome !== null
                                ? (dA < 0) === !eloFavorsHome : null;
                              return (
                                <tr key={bk}>
                                  <td style={{ padding: "4px 0", color: bkColor(bk), fontWeight: 600 }}>{bk}</td>
                                  <td style={{ textAlign: "center", color: "#185FA5" }}>{open?.HomeOdds ?? "—"}</td>
                                  <td style={{ textAlign: "center", color: "#185FA5", fontWeight: 700 }}>{close?.HomeOdds ?? "—"}</td>
                                  <td style={{ textAlign: "center", fontWeight: 700, fontSize: 13, color: agreeH === true ? "#1D7F3A" : agreeH === false ? "#A32D2D" : "#bbb" }}>
                                    {dH !== null ? fmt(dH) : "—"}{agreeH === true ? " ✓" : agreeH === false ? " ✗" : ""}
                                  </td>
                                  <td style={{ textAlign: "center", color: "#993C1D" }}>{open?.AwayOdds ?? "—"}</td>
                                  <td style={{ textAlign: "center", color: "#993C1D", fontWeight: 700 }}>{close?.AwayOdds ?? "—"}</td>
                                  <td style={{ textAlign: "center", fontWeight: 700, fontSize: 13, color: agreeA === true ? "#1D7F3A" : agreeA === false ? "#A32D2D" : "#bbb" }}>
                                    {dA !== null ? fmt(dA) : "—"}{agreeA === true ? " ✓" : agreeA === false ? " ✗" : ""}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        {eloFair && (
                          <div style={{ fontSize: 11, color: "#888", marginTop: 8, lineHeight: 1.6 }}>
                            ELO model: domácí <b style={{ color: "#185FA5" }}>{eloFair.pct}%</b> ({eloFair.fairH})
                            {" · "}hosté <b style={{ color: "#993C1D" }}>{100 - eloFair.pct}%</b> ({eloFair.fairA})
                            {" · "}Pokud kurz na domácí klesá → trh se sbližuje s ELO (pokud máme domácího jako favorita).
                          </div>
                        )}
                      </div>
                    );
                  })()}
                  {/* Legend */}
                  {bkNames.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 20px", marginTop: 10, fontSize: 11, color: "#555" }}>
                      {bkNames.map(bk => (
                        <div key={bk} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <svg width="24" height="12">
                            <line x1="0" y1="6" x2="24" y2="6" stroke={bkColor(bk)} strokeWidth="2" />
                          </svg>
                          <span>{bk} dom.</span>
                          <svg width="24" height="12">
                            <line x1="0" y1="6" x2="24" y2="6" stroke={bkColor(bk)} strokeWidth="1.5" strokeDasharray="5,3" />
                          </svg>
                          <span>host.</span>
                        </div>
                      ))}
                      {eloFair && (
                        <div style={{ display: "flex", alignItems: "center", gap: 6, color: "#aaa" }}>
                          <svg width="24" height="12">
                            <line x1="0" y1="6" x2="24" y2="6" stroke="#888" strokeWidth="1" strokeDasharray="6,4" opacity="0.6" />
                          </svg>
                          <span>ELO fair odds</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </>
          );
        })()}

        {/* HISTORICAL ODDS */}
        {tab === "historical" && (() => {
          if (!historicalOdds.length) return (
            <p style={{ color: "#aaa", fontSize: 13 }}>Žádná historická data. Spusť scraper s --history.</p>
          );

          const bkNames = ["Tipsport", "Fortuna", "Betano"];
          const seen = new Set();
          const sorted  = [...historicalOdds]
            .filter(m => {
              const key = `${m.Date}|${m.HomeTeam}|${m.AwayTeam}`;
              if (seen.has(key)) return false;
              seen.add(key);
              return true;
            })
            .sort((a, b) => b.Date.localeCompare(a.Date));


          // Summary stats for selected bookmaker
          const withBk = sorted.filter(m => m.Bookmakers?.[histBk]?.OpenHome && m.Bookmakers?.[histBk]?.CloseHome);
          const bkCorrect = withBk.filter(m => {
            const homeWon = m.HomeScore > m.AwayScore;
            const bkFavHome = (m.Bookmakers[histBk].CloseHome ?? 99) < (m.Bookmakers[histBk].CloseAway ?? 99);
            return homeWon === bkFavHome;
          }).length;
          // Drift-to-ELO: did the market move its closing odds closer to the ELO fair odds
          // than the opening line was? If yes, the market ended up agreeing with ELO.
          const driftAligned = withBk.filter(m => {
            const p = eloP(m.HomeTeam, m.AwayTeam);
            const eloH = 1 / p;
            const bk = m.Bookmakers[histBk];
            return Math.abs(bk.CloseHome - eloH) < Math.abs(bk.OpenHome - eloH);
          }).length;

          const thSt = { ...s.th, fontSize: 11 };

          return (
            <>
              {/* Bookmaker selector + summary */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
                {bkNames.map(bk => (
                  <button key={bk} onClick={() => setHistBk(bk)} style={{
                    padding: "5px 14px", fontSize: 12, borderRadius: 20, cursor: "pointer", fontWeight: bk === histBk ? 600 : 400,
                    background: bk === histBk ? "#2E7D5E" : "#fff",
                    color: bk === histBk ? "#fff" : "#555",
                    border: `1px solid ${bk === histBk ? "#2E7D5E" : "#ddd"}`,
                  }}>{bk}</button>
                ))}
                <span style={{ fontSize: 12, color: "#aaa", marginLeft: 8 }}>
                  {withBk.length} zápasů ·{" "}
                  <span style={{ color: "#2E7D5E", fontWeight: 600 }}>{histBk}: {bkCorrect}/{withBk.length} ({withBk.length ? ((bkCorrect / withBk.length) * 100).toFixed(1) : "—"}%)</span>
                  {" · "}
                  <span style={{ color: "#7C52C8", fontWeight: 600 }}>Drift→ELO: {driftAligned}/{withBk.length} ({withBk.length ? ((driftAligned / withBk.length) * 100).toFixed(1) : "—"}%)</span>
                </span>
              </div>

              {/* Table */}
              <div style={{ overflowX: "auto" }}>
                <table style={{ ...s.table, fontSize: 12, minWidth: 700 }}>
                  <thead>
                    <tr>
                      <th style={thSt}>Datum</th>
                      <th style={thSt}>Domácí</th>
                      <th style={{ ...thSt, textAlign: "center" }}>Skóre</th>
                      <th style={thSt}>Hosté</th>
                      <th style={{ ...thSt, textAlign: "center" }}>Open dom.</th>
                      <th style={{ ...thSt, textAlign: "center" }}>Close dom.</th>
                      <th style={{ ...thSt, textAlign: "center" }}>ELO dom.</th>
                      <th style={{ ...thSt, textAlign: "center" }}>Open host.</th>
                      <th style={{ ...thSt, textAlign: "center" }}>Close host.</th>
                      <th style={{ ...thSt, textAlign: "center" }}>ELO host.</th>
                      <th style={{ ...thSt, textAlign: "center", minWidth: 80 }}>Výsledek</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((m, i) => {
                      const bk        = m.Bookmakers?.[histBk];
                      const openH     = bk?.OpenHome  ?? null;
                      const openA     = bk?.OpenAway  ?? null;
                      const closeH    = bk?.CloseHome ?? null;
                      const closeA    = bk?.CloseAway ?? null;
                      const homeWon   = m.HomeScore > m.AwayScore;
                      const p         = eloP(m.HomeTeam, m.AwayTeam);
                      const eloH      = 1 / p;
                      const bkFavHome = closeH && closeA ? closeH < closeA : null;
                      const bkOk      = bkFavHome !== null ? homeWon === bkFavHome : null;
                      const driftOk   = (openH != null && closeH != null)
                        ? Math.abs(closeH - eloH) < Math.abs(openH - eloH)
                        : null;

                      return (
                        <tr key={`${m.Date}|${m.HomeTeam}|${m.AwayTeam}`} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                          <td style={{ ...s.td, color: "#aaa", fontSize: 11 }}>{m.Date}</td>
                          <td style={{ ...s.td, fontWeight: homeWon ? 600 : 400 }}>{m.HomeTeam}</td>
                          <td style={{ ...s.td, textAlign: "center", fontWeight: 700, letterSpacing: 1 }}>
                            <span style={{ color: homeWon ? "#3B6D11" : "#aaa" }}>{m.HomeScore}</span>
                            <span style={{ color: "#ddd", margin: "0 3px" }}>–</span>
                            <span style={{ color: !homeWon ? "#3B6D11" : "#aaa" }}>{m.AwayScore}</span>
                          </td>
                          <td style={{ ...s.td, fontWeight: !homeWon ? 600 : 400 }}>{m.AwayTeam}</td>
                          <td style={{ ...s.td, textAlign: "center", color: "#185FA5" }}>{openH ?? "—"}</td>
                          <td style={{ ...s.td, textAlign: "center", color: "#185FA5", fontWeight: 600 }}>{closeH ?? "—"}</td>
                          <td style={{ ...s.td, textAlign: "center", color: "#185FA5", fontWeight: 600 }}>{(1 / p).toFixed(2)}</td>
                          <td style={{ ...s.td, textAlign: "center", color: "#993C1D" }}>{openA ?? "—"}</td>
                          <td style={{ ...s.td, textAlign: "center", color: "#993C1D", fontWeight: 600 }}>{closeA ?? "—"}</td>
                          <td style={{ ...s.td, textAlign: "center", color: "#993C1D", fontWeight: 600 }}>{(1 / (1 - p)).toFixed(2)}</td>
                          <td style={{ ...s.td, textAlign: "center", fontSize: 13, whiteSpace: "nowrap" }}>
                            {driftOk === null
                              ? <span style={{ color: "#bbb" }}>—</span>
                              : driftOk
                                ? <span style={{ color: "#1D7F3A", fontWeight: 700 }}>✓ drift</span>
                                : <span style={{ color: "#A32D2D", fontWeight: 700 }}>✗ drift</span>}
                            {bkOk !== null && <>{" "}{bkOk
                              ? <span style={{ color: "#888", fontSize: 10 }}>bk✓</span>
                              : <span style={{ color: "#888", fontSize: 10 }}>bk✗</span>}</>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          );
        })()}

        {/* MONTE CARLO SIMULACE */}
        {tab === "simulace" && (() => {
          const thS = { padding: "6px 10px", fontSize: 11, fontWeight: 600, color: "#666", background: "#f4f4f4", borderBottom: "2px solid #eee", textAlign: "center", whiteSpace: "nowrap" };
          const bkNames = ["Tipsport", "Fortuna", "Betano"];
          const pill = (active, label, onClick) => (
            <button onClick={onClick} style={{
              padding: "4px 12px", fontSize: 12, borderRadius: 20, cursor: "pointer",
              fontWeight: active ? 600 : 400, background: active ? "#C0392B" : "#fff",
              color: active ? "#fff" : "#555", border: `1px solid ${active ? "#C0392B" : "#ddd"}`,
            }}>{label}</button>
          );

          return (
            <>
              {/* ── Betting Monte Carlo ── */}
              <div style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                <div>
                  <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 2px" }}>Monte Carlo · Betting simulace</h2>
                  <p style={{ fontSize: 12, color: "#888", margin: 0 }}>
                    1 000 simulací · ELO pravděpodobnosti · bankroll €100
                  </p>
                </div>
                <button onClick={runAllSims} disabled={simRunning} style={{
                  padding: "8px 20px", fontSize: 13, fontWeight: 700, borderRadius: 8, cursor: simRunning ? "wait" : "pointer",
                  background: simRunning ? "#aaa" : "#C0392B", color: "#fff", border: "none",
                }}>
                  {simRunning ? "Počítám…" : "▶ Spustit simulaci"}
                </button>
              </div>

              {/* Controls */}
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16, alignItems: "center" }}>
                {bkNames.map(bk => pill(simBk === bk, bk, () => setSimBk(bk)))}
                <span style={{ color: "#ccc" }}>|</span>
                {pill(simOdds === "close", "Closing kurzy", () => setSimOdds("close"))}
                {pill(simOdds === "open",  "Opening kurzy", () => setSimOdds("open"))}
                <span style={{ color: "#ccc" }}>|</span>
                <span style={{ fontSize: 12, color: "#666" }}>Min. edge:</span>
                <input type="range" min={0} max={20} step={1} value={simEdge}
                  onChange={e => setSimEdge(+e.target.value)}
                  style={{ accentColor: "#C0392B", width: 80 }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: "#C0392B", minWidth: 28 }}>{simEdge}%</span>
              </div>

              {!bettingMC ? <p style={{ color: "#aaa", fontSize: 13 }}>Žádná data pro {simBk}.</p> : (() => {
                const { matchCount, betCount, hitRate, actual, sim, START } = bettingMC;
                const roiColor = v => parseFloat(v) >= 0 ? "#1D7F3A" : "#A32D2D";
                const bar = (end, start) => {
                  const roi = ((end - start) / start * 100);
                  const w = Math.min(Math.abs(roi) / 100 * 80, 80);
                  return <div style={{ display: "inline-block", width: w, height: 8, borderRadius: 4, background: roi >= 0 ? "#1D7F3A" : "#A32D2D", opacity: 0.7, marginLeft: 6, verticalAlign: "middle" }} />;
                };
                return (
                  <>
                    <div style={{ fontSize: 12, color: "#888", marginBottom: 12 }}>
                      {matchCount} zápasů · {betCount} sázek ({hitRate}% výher)
                    </div>
                    <div style={{ overflowX: "auto", marginBottom: 28 }}>
                      <table style={{ ...s.table, fontSize: 12, minWidth: 560 }}>
                        <thead>
                          <tr>
                            <th style={{ ...thS, textAlign: "left", minWidth: 120 }}>Strategie</th>
                            <th style={thS}>Skutečný výsledek</th>
                            <th style={thS}>Konečný bankroll</th>
                            <th style={{ ...thS, color: "#888" }}>P10 simulace</th>
                            <th style={{ ...thS, color: "#C0392B" }}>Medián simulace</th>
                            <th style={{ ...thS, color: "#888" }}>P90 simulace</th>
                            <th style={thS}>P(zisk)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { label: "Flat (1 jednotka)", act: actual.flat, end: actual.flatEnd, s: sim.flat },
                            { label: "Kelly (max 25%)",   act: actual.kelly, end: actual.kellyEnd, s: sim.kelly },
                            { label: "Half Kelly",        act: actual.half,  end: actual.halfEnd,  s: sim.half  },
                          ].map(({ label, act, end, s: sv }, i) => (
                            <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                              <td style={{ ...s.td, fontWeight: 600 }}>{label}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 700, color: roiColor(act) }}>{act}</td>
                              <td style={{ ...s.td, textAlign: "center" }}>
                                €{end}
                                {bar(parseFloat(end), START)}
                              </td>
                              <td style={{ ...s.td, textAlign: "center", color: "#aaa" }}>€{sv.p10}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 700, color: roiColor(sv.p50 - START) }}>€{sv.p50}</td>
                              <td style={{ ...s.td, textAlign: "center", color: "#aaa" }}>€{sv.p90}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 600, color: parseInt(sv.pp) >= 50 ? "#1D7F3A" : "#A32D2D" }}>{sv.pp}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                );
              })()}

              {/* ── ROI Srovnání ── */}
              {roiComp && (() => {
                const rc = roiComp;
                const roiColor = v => parseFloat(v) >= 0 ? "#1D7F3A" : "#A32D2D";
                const rows = [
                  { label: "Opening kurzy (favorite)", d: rc.openFav },
                  { label: "Closing kurzy (favorite)", d: rc.closeFav },
                  { label: "ELO model (closing kurzy)", d: rc.eloModel },
                ];
                return (
                  <>
                    <hr style={{ border: "none", borderTop: "1px solid #eee", margin: "8px 0 20px" }} />
                    <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 4px" }}>ROI srovnání · {simBk} · {rc.matchCount} zápasů</h2>
                    <p style={{ fontSize: 12, color: "#888", margin: "0 0 12px" }}>
                      Vždy sázíme na favorita dané strategie. Flat €10 stake · nebo stake na zisk přesně €10.
                    </p>
                    <div style={{ overflowX: "auto", marginBottom: 24 }}>
                      <table style={{ ...s.table, fontSize: 12, minWidth: 620 }}>
                        <thead>
                          <tr>
                            <th style={{ ...thS, textAlign: "left", minWidth: 200 }}>Strategie</th>
                            <th style={thS}>Sázky</th>
                            <th style={thS}>Hit%</th>
                            <th style={thS}>Flat €10 P&L</th>
                            <th style={{ ...thS, color: "#C0392B" }}>Flat ROI/bet</th>
                            <th style={thS}>Profit €10 P&L</th>
                            <th style={{ ...thS, color: "#C0392B" }}>Profit ROI</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map(({ label, d }, i) => (
                            <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                              <td style={{ ...s.td, fontWeight: 600 }}>{label}</td>
                              <td style={{ ...s.td, textAlign: "center" }}>{d.bets}</td>
                              <td style={{ ...s.td, textAlign: "center" }}>{d.hitRate}%</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 700, color: roiColor(d.plFlat) }}>€{d.plFlat}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 700, color: roiColor(d.roiFlat) }}>{d.roiFlat}%</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 700, color: roiColor(d.plProfit) }}>€{d.plProfit}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 700, color: roiColor(d.roiProfit) }}>{d.roiProfit}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                );
              })()}

              {/* ── Příští sezóna ── */}
              {nextSeason && (() => {
                const { summary, N: nsN, totalGames } = nextSeason;
                const heatColor = p => { const v = Math.min(parseFloat(p) / 60, 1); return `rgba(192,57,43,${(v * 0.7 + 0.05).toFixed(2)})`; };
                return (
                  <>
                    <hr style={{ border: "none", borderTop: "1px solid #eee", margin: "8px 0 20px" }} />
                    <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 4px" }}>Simulace příští sezóny</h2>
                    <p style={{ fontSize: 12, color: "#888", margin: "0 0 12px" }}>
                      {nsN.toLocaleString()} simulací · round-robin {totalGames} zápasů · start z aktuálních ELO ratingů
                    </p>
                    <div style={{ overflowX: "auto", marginBottom: 20 }}>
                      <table style={{ ...s.table, fontSize: 12, minWidth: 560 }}>
                        <thead>
                          <tr>
                            <th style={{ ...thS, textAlign: "left" }}>Tým</th>
                            <th style={thS}>ELO</th>
                            <th style={thS}>Exp. výher</th>
                            <th style={{ ...thS, color: "#C0392B" }}>P(1.)</th>
                            <th style={{ ...thS, color: "#C0392B" }}>P(top 4)</th>
                            <th style={{ ...thS, color: "#C0392B" }}>P(top 8)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {summary.map((t, i) => (
                            <tr key={t.team} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                              <td style={{ ...s.td, fontWeight: 600 }}>{t.team}</td>
                              <td style={{ ...s.td, textAlign: "center", color: "#888" }}>{t.elo}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 600 }}>{t.expWins}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 600, color: "#C0392B" }}>{t.pFirst}%</td>
                              <td style={{ ...s.td, textAlign: "center" }}>{t.pTop4}%</td>
                              <td style={{ ...s.td, textAlign: "center" }}>{t.pTop8}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: "#555" }}>Heat mapa umístění (%)</h3>
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
                        <thead>
                          <tr>
                            <th style={{ ...thS, textAlign: "left", minWidth: 160 }}>Tým</th>
                            {summary.map((_, i) => <th key={i} style={{ ...thS, minWidth: 42 }}>{i + 1}.</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {summary.map(t => (
                            <tr key={t.team}>
                              <td style={{ ...s.td, fontWeight: 600, fontSize: 11, paddingRight: 12 }}>{t.team}</td>
                              {t.posDist.map((p, ci) => (
                                <td key={ci} style={{
                                  ...s.td, textAlign: "center", padding: "4px 6px",
                                  background: parseFloat(p) > 0.5 ? heatColor(p) : "#f9f9f9",
                                  color: parseFloat(p) > 20 ? "#fff" : parseFloat(p) > 0.5 ? "#8B1A0E" : "#ddd",
                                  fontWeight: parseFloat(p) > 10 ? 700 : 400,
                                }}>{parseFloat(p) > 0.5 ? p : ""}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                );
              })()}

              {/* ── Season standings MC (this season) ── */}
              {monteCarlo && (() => {
                const { summary, N: sN, gamesCount } = monteCarlo;
                const heatColor = p => { const v = Math.min(parseFloat(p) / 60, 1); return `rgba(46,125,94,${(v * 0.75 + 0.05).toFixed(2)})`; };
                return (
                  <>
                    <hr style={{ border: "none", borderTop: "1px solid #eee", margin: "8px 0 20px" }} />
                    <div style={{ marginBottom: 12 }}>
                      <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 4px" }}>Monte Carlo · Sezóna</h2>
                      <p style={{ fontSize: 12, color: "#888", margin: 0 }}>{sN.toLocaleString()} simulací · {gamesCount} zbývajících zápasů</p>
                    </div>
                    <div style={{ overflowX: "auto", marginBottom: 20 }}>
                      <table style={{ ...s.table, fontSize: 12, minWidth: 600 }}>
                        <thead>
                          <tr>
                            <th style={{ ...thS, textAlign: "left" }}>Tým</th>
                            <th style={thS}>ELO</th>
                            <th style={thS}>V–P</th>
                            <th style={thS}>Exp. výher</th>
                            <th style={{ ...thS, color: "#C0392B" }}>P(1.)</th>
                            <th style={{ ...thS, color: "#C0392B" }}>P(top 4)</th>
                            <th style={{ ...thS, color: "#C0392B" }}>P(top 8)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {summary.map((t, i) => (
                            <tr key={t.team} style={{ background: i % 2 === 0 ? "#fff" : "#fafafa" }}>
                              <td style={{ ...s.td, fontWeight: 600 }}>{t.team}</td>
                              <td style={{ ...s.td, textAlign: "center", color: "#888" }}>{Math.round(t.elo)}</td>
                              <td style={{ ...s.td, textAlign: "center" }}>
                                <span style={{ color: "#1D7F3A", fontWeight: 600 }}>{t.wins}</span>
                                <span style={{ color: "#ccc" }}>–</span>
                                <span style={{ color: "#A32D2D", fontWeight: 600 }}>{t.losses}</span>
                              </td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 600 }}>{t.expWins}</td>
                              <td style={{ ...s.td, textAlign: "center", fontWeight: 600, color: "#C0392B" }}>{t.pFirst}%</td>
                              <td style={{ ...s.td, textAlign: "center" }}>{t.pTop4}%</td>
                              <td style={{ ...s.td, textAlign: "center" }}>{t.pTop8}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: "#555" }}>Heat mapa umístění (%)</h3>
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
                        <thead>
                          <tr>
                            <th style={{ ...thS, textAlign: "left", minWidth: 160 }}>Tým</th>
                            {summary.map((_, i) => <th key={i} style={{ ...thS, minWidth: 42 }}>{i + 1}.</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {summary.map(t => (
                            <tr key={t.team}>
                              <td style={{ ...s.td, fontWeight: 600, fontSize: 11, paddingRight: 12 }}>{t.team}</td>
                              {t.posDist.map((p, ci) => (
                                <td key={ci} style={{
                                  ...s.td, textAlign: "center", padding: "4px 6px",
                                  background: parseFloat(p) > 0.5 ? heatColor(p) : "#f9f9f9",
                                  color: parseFloat(p) > 20 ? "#fff" : parseFloat(p) > 0.5 ? "#1a5c3e" : "#ddd",
                                  fontWeight: parseFloat(p) > 10 ? 700 : 400,
                                }}>{parseFloat(p) > 0.5 ? p : ""}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                );
              })()}
            </>
          );
        })()}

      </div>
      <Analytics />
    </div>
  );
}