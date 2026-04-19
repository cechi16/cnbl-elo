// Runs in a separate thread — no DOM, no React

function expectedScore(ra, rb) {
  return 1 / (1 + Math.pow(10, (rb - ra) / 400));
}

function teamsMatch(full, short) {
  if (!full || !short) return false;
  const norm = s => s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]/g, "");
  return norm(full).includes(norm(short)) || norm(short).includes(norm(full));
}

function eloP(home, away, ratings, base, homeAdv) {
  const eH = ratings[home] ?? Object.entries(ratings).find(([t]) => teamsMatch(t, home))?.[1] ?? base;
  const eA = ratings[away] ?? Object.entries(ratings).find(([t]) => teamsMatch(t, away))?.[1] ?? base;
  return expectedScore(eH + homeAdv, eA);
}

function computeMonteCarlo({ results, upcoming, btRatings, btBase, btHomeAdv }) {
  const N = 5000;
  const teams = Object.keys(btRatings);
  const nT = teams.length;

  const baseWins = {}, baseLosses = {};
  results.forEach(r => {
    const hw = r.HomeScore > r.AwayScore;
    baseWins[r.HomeTeam]   = (baseWins[r.HomeTeam]   || 0) + (hw ? 1 : 0);
    baseLosses[r.HomeTeam] = (baseLosses[r.HomeTeam] || 0) + (hw ? 0 : 1);
    baseWins[r.AwayTeam]   = (baseWins[r.AwayTeam]   || 0) + (hw ? 0 : 1);
    baseLosses[r.AwayTeam] = (baseLosses[r.AwayTeam] || 0) + (hw ? 1 : 0);
  });

  const resolveTeam = name =>
    teams.find(t => t === name) ?? teams.find(t => teamsMatch(t, name)) ?? name;

  const games = upcoming.map(m => ({
    home: resolveTeam(m.HomeTeam),
    away: resolveTeam(m.AwayTeam),
  }));

  const posCount = {};
  teams.forEach(t => { posCount[t] = new Array(nT).fill(0); });

  for (let s = 0; s < N; s++) {
    const wins = { ...baseWins };
    games.forEach(({ home, away }) => {
      const p = eloP(home, away, btRatings, btBase, btHomeAdv);
      if (Math.random() < p) wins[home] = (wins[home] || 0) + 1;
      else wins[away] = (wins[away] || 0) + 1;
    });
    const ranked = [...teams].sort((a, b) => (wins[b] || 0) - (wins[a] || 0));
    ranked.forEach((t, i) => posCount[t][i]++);
  }

  const summary = teams.map(t => ({
    team: t,
    elo: btRatings[t],
    wins: baseWins[t] || 0,
    losses: baseLosses[t] || 0,
    expWins: ((baseWins[t] || 0) + games.reduce((s, g) => {
      const p = eloP(g.home, g.away, btRatings, btBase, btHomeAdv);
      if (g.home === t) return s + p;
      if (g.away === t) return s + (1 - p);
      return s;
    }, 0)).toFixed(1),
    pTop4:  ((posCount[t].slice(0, 4).reduce((a, b) => a + b, 0) / N) * 100).toFixed(1),
    pTop8:  ((posCount[t].slice(0, 8).reduce((a, b) => a + b, 0) / N) * 100).toFixed(1),
    pFirst: ((posCount[t][0] / N) * 100).toFixed(1),
    posDist: posCount[t].map(c => (c / N * 100).toFixed(1)),
  })).sort((a, b) => b.wins - a.wins || b.elo - a.elo);

  return { summary, nT, N, gamesCount: games.length };
}

function computeBettingMC({ historicalOdds, btRatings, btBase, btHomeAdv, simBk, simOdds, simEdge }) {
  const N = 1000;
  const START = 100;
  const edgeMin = simEdge / 100;

  const matches = historicalOdds.filter(m => {
    const bk = m.Bookmakers?.[simBk];
    if (!bk) return false;
    const oh = simOdds === "open" ? (bk.OpenHome ?? bk.CloseHome) : bk.CloseHome;
    const oa = simOdds === "open" ? (bk.OpenAway ?? bk.CloseAway) : bk.CloseAway;
    return oh && oa && m.HomeScore >= 0;
  });
  if (!matches.length) return null;

  const getOdds = bk => ({
    h: simOdds === "open" ? (bk.OpenHome ?? bk.CloseHome) : bk.CloseHome,
    a: simOdds === "open" ? (bk.OpenAway ?? bk.CloseAway) : bk.CloseAway,
  });

  let actualFlat = START, actualKelly = START, actualHalf = START;
  let actualBets = 0, actualWins = 0;
  matches.forEach(m => {
    const { h, a } = getOdds(m.Bookmakers[simBk]);
    const p = eloP(m.HomeTeam, m.AwayTeam, btRatings, btBase, btHomeAdv);
    const betHome = p >= 0.5;
    const betOdds = betHome ? h : a;
    const betP    = betHome ? p : 1 - p;
    const edge    = betP * betOdds - 1;
    if (edge <= edgeMin) return;
    const kelly = Math.min((betP * betOdds - 1) / (betOdds - 1), 0.25);
    const won   = betHome ? m.HomeScore > m.AwayScore : m.AwayScore > m.HomeScore;
    actualBets++;
    if (won) actualWins++;
    actualFlat += won ? betOdds - 1 : -1;
    const sk = actualKelly * kelly;
    actualKelly += won ? sk * (betOdds - 1) : -sk;
    const sh = actualHalf * kelly * 0.5;
    actualHalf  += won ? sh * (betOdds - 1) : -sh;
  });

  const endFlat = [], endKelly = [], endHalf = [];
  for (let s = 0; s < N; s++) {
    let bkFlat = START, bkKelly = START, bkHalf = START;
    matches.forEach(m => {
      const { h, a } = getOdds(m.Bookmakers[simBk]);
      const p = eloP(m.HomeTeam, m.AwayTeam, btRatings, btBase, btHomeAdv);
      const betHome = p >= 0.5;
      const betOdds = betHome ? h : a;
      const betP    = betHome ? p : 1 - p;
      const edge    = betP * betOdds - 1;
      if (edge <= edgeMin) return;
      const kelly = Math.min((betP * betOdds - 1) / (betOdds - 1), 0.25);
      const won   = Math.random() < betP;
      bkFlat  += won ? betOdds - 1 : -1;
      const sk = bkKelly * kelly;
      bkKelly  = Math.max(bkKelly + (won ? sk * (betOdds - 1) : -sk), 0);
      const sh = bkHalf * kelly * 0.5;
      bkHalf   = Math.max(bkHalf  + (won ? sh * (betOdds - 1) : -sh), 0);
    });
    endFlat.push(bkFlat);
    endKelly.push(bkKelly);
    endHalf.push(bkHalf);
  }

  const pct = (arr, p) => [...arr].sort((a, b) => a - b)[Math.floor(arr.length * p / 100)];
  const roiK = (v, start) => ((v - start) / start * 100).toFixed(1);
  const pp   = arr => (arr.filter(x => x > START).length / arr.length * 100).toFixed(0);

  return {
    matchCount: matches.length,
    betCount: actualBets,
    hitRate: actualBets ? (actualWins / actualBets * 100).toFixed(1) : "—",
    actual: {
      flat:     ((actualFlat - START) / (actualBets || 1) * 100).toFixed(2) + "% / bet",
      kelly:    roiK(actualKelly, START) + "%",
      half:     roiK(actualHalf,  START) + "%",
      flatEnd:  actualFlat.toFixed(1),
      kellyEnd: actualKelly.toFixed(1),
      halfEnd:  actualHalf.toFixed(1),
    },
    sim: {
      flat:  { p10: pct(endFlat,  10).toFixed(1), p50: pct(endFlat,  50).toFixed(1), p90: pct(endFlat,  90).toFixed(1), pp: pp(endFlat)  },
      kelly: { p10: pct(endKelly, 10).toFixed(1), p50: pct(endKelly, 50).toFixed(1), p90: pct(endKelly, 90).toFixed(1), pp: pp(endKelly) },
      half:  { p10: pct(endHalf,  10).toFixed(1), p50: pct(endHalf,  50).toFixed(1), p90: pct(endHalf,  90).toFixed(1), pp: pp(endHalf)  },
    },
    N, START,
  };
}

function computeRoiComparison({ historicalOdds, btRatings, btBase, btHomeAdv, simBk }) {
  const STAKE = 10; // €10 flat stake
  const PROFIT_TARGET = 10; // stake to win €10

  const rows = [];
  const strategies = {
    openFav:  { bets: 0, wins: 0, plFlat: 0, plProfit: 0 },
    closeFav: { bets: 0, wins: 0, plFlat: 0, plProfit: 0 },
    eloModel: { bets: 0, wins: 0, plFlat: 0, plProfit: 0 },
  };

  historicalOdds.forEach(m => {
    const bk = m.Bookmakers?.[simBk];
    if (!bk?.CloseHome || !bk?.CloseAway || m.HomeScore < 0) return;
    const oh = bk.OpenHome ?? bk.CloseHome;
    const oa = bk.OpenAway ?? bk.CloseAway;
    const ch = bk.CloseHome, ca = bk.CloseAway;
    const p  = eloP(m.HomeTeam, m.AwayTeam, btRatings, btBase, btHomeAdv);
    const homeWon = m.HomeScore > m.AwayScore;

    const placeBet = (s, betHome, odds, won) => {
      s.bets++;
      if (won) s.wins++;
      s.plFlat   += won ? STAKE * (odds - 1) : -STAKE;
      const stake = PROFIT_TARGET / (odds - 1);
      s.plProfit += won ? PROFIT_TARGET : -stake;
    };

    // Opening favorite
    placeBet(strategies.openFav,  oh < oa, oh < oa ? oh : oa, oh < oa ? homeWon : !homeWon);
    // Closing favorite
    placeBet(strategies.closeFav, ch < ca, ch < ca ? ch : ca, ch < ca ? homeWon : !homeWon);
    // ELO model
    placeBet(strategies.eloModel, p >= 0.5, p >= 0.5 ? ch : ca, p >= 0.5 ? homeWon : !homeWon);

    rows.push({
      date: m.Date, home: m.HomeTeam, away: m.AwayTeam,
      score: `${m.HomeScore}–${m.AwayScore}`,
      oh, oa, ch, ca,
      eloP: (p * 100).toFixed(0),
      eloOddsH: (1 / p).toFixed(2),
      eloOddsA: (1 / (1 - p)).toFixed(2),
      homeWon,
    });
  });

  const fmt = s => ({
    bets: s.bets,
    wins: s.wins,
    hitRate: s.bets ? (s.wins / s.bets * 100).toFixed(1) : "—",
    roiFlat:   (s.plFlat   / (s.bets * STAKE)   * 100).toFixed(2),
    plFlat:    s.plFlat.toFixed(1),
    roiProfit: (s.plProfit / (s.bets * PROFIT_TARGET) * 100).toFixed(2),
    plProfit:  s.plProfit.toFixed(1),
  });

  return {
    openFav:  fmt(strategies.openFav),
    closeFav: fmt(strategies.closeFav),
    eloModel: fmt(strategies.eloModel),
    matchCount: rows.length,
    rows,
  };
}

function computeNextSeason({ btRatings, btBase, btHomeAdv }) {
  const N = 5000;
  const teams = Object.keys(btRatings);
  const nT = teams.length;

  // Full round-robin: each pair plays home + away
  const schedule = [];
  for (let i = 0; i < nT; i++)
    for (let j = 0; j < nT; j++)
      if (i !== j) schedule.push({ home: teams[i], away: teams[j] });

  const posCount = {};
  teams.forEach(t => { posCount[t] = new Array(nT).fill(0); });

  for (let s = 0; s < N; s++) {
    const wins = {};
    schedule.forEach(({ home, away }) => {
      const p = eloP(home, away, btRatings, btBase, btHomeAdv);
      const winner = Math.random() < p ? home : away;
      wins[winner] = (wins[winner] || 0) + 1;
    });
    const ranked = [...teams].sort((a, b) => (wins[b] || 0) - (wins[a] || 0));
    ranked.forEach((t, i) => posCount[t][i]++);
  }

  const summary = teams.map(t => ({
    team: t,
    elo: Math.round(btRatings[t]),
    expWins: (schedule.reduce((acc, g) => {
      const p = eloP(g.home, g.away, btRatings, btBase, btHomeAdv);
      if (g.home === t) return acc + p;
      if (g.away === t) return acc + (1 - p);
      return acc;
    }, 0)).toFixed(1),
    pFirst: ((posCount[t][0] / N) * 100).toFixed(1),
    pTop4:  ((posCount[t].slice(0, 4).reduce((a, b) => a + b, 0) / N) * 100).toFixed(1),
    pTop8:  ((posCount[t].slice(0, 8).reduce((a, b) => a + b, 0) / N) * 100).toFixed(1),
    posDist: posCount[t].map(c => (c / N * 100).toFixed(1)),
  })).sort((a, b) => parseFloat(b.expWins) - parseFloat(a.expWins));

  return { summary, nT, N, totalGames: schedule.length };
}

self.onmessage = ({ data }) => {
  if (data.type === "monteCarlo")    self.postMessage({ type: "monteCarlo",    result: computeMonteCarlo(data.payload) });
  if (data.type === "bettingMC")     self.postMessage({ type: "bettingMC",     result: computeBettingMC(data.payload) });
  if (data.type === "roiComparison") self.postMessage({ type: "roiComparison", result: computeRoiComparison(data.payload) });
  if (data.type === "nextSeason")    self.postMessage({ type: "nextSeason",    result: computeNextSeason(data.payload) });
};
