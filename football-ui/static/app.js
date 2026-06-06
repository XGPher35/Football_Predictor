/* ═══════════════════════════════════════════════════════════════
   Football Match Predictor — Client-side logic
   ═══════════════════════════════════════════════════════════════ */

const API = "";  // same origin, proxied by Flask

// ── State ──────────────────────────────────────────────────────
let teams = [];
let homeTeam = "";
let awayTeam = "";

// ── DOM refs ───────────────────────────────────────────────────
const homeInput    = document.getElementById("home-input");
const awayInput    = document.getElementById("away-input");
const homeDropdown = document.getElementById("home-dropdown");
const awayDropdown = document.getElementById("away-dropdown");
const homeBadge    = document.getElementById("home-badge");
const awayBadge    = document.getElementById("away-badge");
const matchDate    = document.getElementById("match-date");
const btnPredict   = document.getElementById("btn-predict");
const errorMsg     = document.getElementById("error-msg");
const resultsEl    = document.getElementById("results");

// ── Init ───────────────────────────────────────────────────────
(async function init() {
  // Set default date to today
  matchDate.value = new Date().toISOString().slice(0, 10);

  try {
    const res = await fetch(`${API}/api/teams`);
    const data = await res.json();
    teams = data.teams;
  } catch (e) {
    errorMsg.textContent = "Failed to load team list. Is the server running?";
  }
})();

// ── Autocomplete logic ─────────────────────────────────────────
function setupAutocomplete(input, dropdown, badge, setTeam) {
  let activeIdx = -1;

  function render(list) {
    dropdown.innerHTML = "";
    activeIdx = -1;
    if (list.length === 0) {
      dropdown.classList.remove("open");
      return;
    }
    list.forEach((t, i) => {
      const li = document.createElement("li");
      li.textContent = t;
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        select(t);
      });
      dropdown.appendChild(li);
    });
    dropdown.classList.add("open");
  }

  function filter() {
    const q = input.value.trim().toLowerCase();
    if (!q) {
      render(teams.slice(0, 10));
      return;
    }
    const starts = teams.filter((t) => t.toLowerCase().startsWith(q));
    const contains = teams.filter(
      (t) => t.toLowerCase().includes(q) && !starts.includes(t)
    );
    render([...starts, ...contains].slice(0, 10));
  }

  function select(team) {
    input.value = team;
    badge.textContent = team;
    setTeam(team);
    dropdown.classList.remove("open");
    updateButton();
  }

  function highlightIdx(idx) {
    const items = dropdown.querySelectorAll("li");
    items.forEach((li) => li.classList.remove("active"));
    if (idx >= 0 && idx < items.length) {
      items[idx].classList.add("active");
      items[idx].scrollIntoView({ block: "nearest" });
    }
  }

  input.addEventListener("focus", filter);
  input.addEventListener("input", () => {
    setTeam("");
    badge.textContent = "—";
    updateButton();
    filter();
  });

  input.addEventListener("keydown", (e) => {
    const items = dropdown.querySelectorAll("li");
    if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIdx = Math.min(activeIdx + 1, items.length - 1);
      highlightIdx(activeIdx);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIdx = Math.max(activeIdx - 1, 0);
      highlightIdx(activeIdx);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeIdx >= 0 && activeIdx < items.length) {
        select(items[activeIdx].textContent);
      }
    } else if (e.key === "Escape") {
      dropdown.classList.remove("open");
    }
  });

  input.addEventListener("blur", () => {
    setTimeout(() => dropdown.classList.remove("open"), 150);
  });
}

setupAutocomplete(homeInput, homeDropdown, homeBadge, (t) => (homeTeam = t));
setupAutocomplete(awayInput, awayDropdown, awayBadge, (t) => (awayTeam = t));

// ── Button state ───────────────────────────────────────────────
function updateButton() {
  btnPredict.disabled = !(homeTeam && awayTeam && homeTeam !== awayTeam);
}

// ── Predict ────────────────────────────────────────────────────
btnPredict.addEventListener("click", async () => {
  if (!homeTeam || !awayTeam || homeTeam === awayTeam) return;
  errorMsg.textContent = "";
  btnPredict.classList.add("loading");
  btnPredict.disabled = true;
  resultsEl.hidden = true;

  try {
    // fetch prediction + form in parallel
    const [predRes, homeFormRes, awayFormRes] = await Promise.all([
      fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          date: matchDate.value,
        }),
      }),
      fetch(`${API}/api/team-form/${encodeURIComponent(homeTeam)}?date=${matchDate.value}`),
      fetch(`${API}/api/team-form/${encodeURIComponent(awayTeam)}?date=${matchDate.value}`),
    ]);

    const pred = await predRes.json();
    if (pred.error) throw new Error(pred.error);

    const homeForm = await homeFormRes.json();
    const awayForm = await awayFormRes.json();

    renderResults(pred, homeForm, awayForm);
  } catch (e) {
    errorMsg.textContent = e.message || "Prediction failed.";
  } finally {
    btnPredict.classList.remove("loading");
    updateButton();
  }
});

// ── Render ─────────────────────────────────────────────────────
function renderResults(pred, homeForm, awayForm) {
  resultsEl.hidden = false;

  // ─ Outcome bars ─
  const h = pred.outcome.H;
  const d = pred.outcome.D;
  const a = pred.outcome.A;

  setTimeout(() => {
    document.getElementById("bar-home").style.width = `${h * 100}%`;
    document.getElementById("bar-draw").style.width = `${d * 100}%`;
    document.getElementById("bar-away").style.width = `${a * 100}%`;
  }, 50);

  document.getElementById("pct-home").textContent = `${(h * 100).toFixed(1)}%`;
  document.getElementById("pct-draw").textContent = `${(d * 100).toFixed(1)}%`;
  document.getElementById("pct-away").textContent = `${(a * 100).toFixed(1)}%`;

  // verdict
  let verdict = "";
  const maxP = Math.max(h, d, a);
  if (maxP === h) verdict = `${pred.home_team} favoured to win`;
  else if (maxP === a) verdict = `${pred.away_team} favoured to win`;
  else verdict = "Match predicted as a draw";
  document.getElementById("outcome-verdict").textContent = verdict;

  // ─ Score ─
  const hGoals = Math.round(pred.goals.home);
  const aGoals = Math.round(pred.goals.away);
  document.getElementById("score-home-name").textContent = pred.home_team;
  document.getElementById("score-away-name").textContent = pred.away_team;
  document.getElementById("score-home-val").textContent = hGoals;
  document.getElementById("score-away-val").textContent = aGoals;
  document.getElementById("score-home-raw").textContent = `xG: ${pred.goals.home.toFixed(2)}`;
  document.getElementById("score-away-raw").textContent = `xG: ${pred.goals.away.toFixed(2)}`;

  // ─ Stats table ─
  document.getElementById("stats-home-header").textContent = pred.home_team;
  document.getElementById("stats-away-header").textContent = pred.away_team;
  const statsDef = [
    ["Shots",           "home_shots",           "away_shots"],
    ["Shots on Target", "home_shots_on_target",  "away_shots_on_target"],
    ["Corners",         "home_corners",          "away_corners"],
    ["Fouls",           "home_fouls",            "away_fouls"],
    ["Yellow Cards",    "home_yellows",          "away_yellows"],
    ["Red Cards",       "home_reds",             "away_reds"],
    ["HT Goals",        "home_ht_goals",         "away_ht_goals"],
  ];
  const tbody = document.getElementById("stats-tbody");
  tbody.innerHTML = "";
  statsDef.forEach(([label, hKey, aKey]) => {
    const tr = document.createElement("tr");
    const hVal = pred.stats[hKey] != null ? pred.stats[hKey].toFixed(1) : "—";
    const aVal = pred.stats[aKey] != null ? pred.stats[aKey].toFixed(1) : "—";
    tr.innerHTML = `<td>${hVal}</td><td>${label}</td><td>${aVal}</td>`;
    tbody.appendChild(tr);
  });

  // ─ Form tables ─
  renderFormTable("home", pred.home_team, homeForm.form);
  renderFormTable("away", pred.away_team, awayForm.form);

  // Scroll into view
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFormTable(side, teamName, matches) {
  document.getElementById(`form-${side}-name`).textContent = teamName;
  const tbody = document.getElementById(`form-${side}-tbody`);
  tbody.innerHTML = "";

  if (!matches || matches.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5" style="text-align:center;color:var(--text-dim)">No recent matches</td>`;
    tbody.appendChild(tr);
    return;
  }

  matches.forEach((m) => {
    const tr = document.createElement("tr");
    const opponent = m.home === teamName ? m.away : m.home;
    const venue = m.home === teamName ? "(H)" : "(A)";
    tr.innerHTML = `
      <td>${m.date}</td>
      <td>${opponent} ${venue}</td>
      <td>${m.gf}</td>
      <td>${m.ga}</td>
      <td class="res-${m.result}">${m.result}</td>
    `;
    tbody.appendChild(tr);
  });
}
