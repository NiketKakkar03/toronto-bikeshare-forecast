const stationList = document.querySelector("#stations");
const search = document.querySelector("#search");
const detail = document.querySelector("#detail");
let stations = [];
let selected = null;
let horizon = 30;

const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
})[character]);
const riskClass = (risk) => `risk-${risk === "high" ? "high" : risk === "medium" ? "medium" : "low"}`;

function renderStations() {
  const needle = search.value.trim().toLowerCase();
  stationList.innerHTML = stations.filter((station) =>
    station.name.toLowerCase().includes(needle) || station.station_id.includes(needle)
  ).map((station) => `<button class="station ${selected === station.station_id ? "active" : ""}"
    data-station-id="${escapeHtml(station.station_id)}" role="option">
    <strong>${escapeHtml(station.name)}</strong>
    <small>${station.bikes_available} bikes · ${station.docks_available} docks</small>
  </button>`).join("");
}

async function renderForecast() {
  const response = await fetch(`/api/stations/${encodeURIComponent(selected)}/forecast?horizon=${horizon}`);
  const data = await response.json();
  const current = data.station;
  const rawJson = escapeHtml(JSON.stringify(data, null, 2));
  const horizonButtons = [15, 30, 60].map((minutes) =>
    `<button data-horizon="${minutes}" class="${minutes === horizon ? "active" : ""}">${minutes} min</button>`
  ).join("");
  let outlook = `<p class="notice"><strong>Forecast ${escapeHtml(data.state)}.</strong> ${escapeHtml(data.reason || "Try again shortly.")}</p>`;
  if (data.forecast) {
    const value = data.forecast;
    const guidance = data.guidance;
    const dominantRisk = guidance?.pickup_risk === "high" || guidance?.return_risk === "high" ? "high" : guidance?.pickup_risk === "medium" || guidance?.return_risk === "medium" ? "medium" : "low";
    const guidancePanel = guidance ? `<section class="guidance ${riskClass(dominantRisk)}">
      <p class="label">Outcome</p>
      <h3>${escapeHtml(guidance.headline)}</h3>
      <p class="recommendation">${escapeHtml(guidance.recommendation)}</p>
      <p>${escapeHtml(guidance.explanation)}</p>
      <div class="risk-grid">
        <span class="${riskClass(guidance.pickup_risk)}">Pickup risk <strong>${escapeHtml(guidance.pickup_risk)}</strong></span>
        <span class="${riskClass(guidance.return_risk)}">Return risk <strong>${escapeHtml(guidance.return_risk)}</strong></span>
      </div>
    </section>` : "";
    outlook = `${guidancePanel}<details class="model-details"><summary>Model details</summary><div class="metrics">
      <div class="metric"><span>Expected departures</span><strong>${value.departures_expected}</strong></div>
      <div class="metric"><span>Expected arrivals</span><strong>${value.arrivals_expected}</strong></div>
      <div class="metric"><span>Expected net flow</span><strong>${value.net_flow_expected}</strong><small>arrivals minus departures</small></div>
      <div class="metric"><span>Demand pressure</span><strong>${escapeHtml(value.demand_pressure)}</strong></div>
    </div></details>`;
  }
  const alternatives = data.alternatives.length ? `<h3>Nearby alternatives</h3><ol class="alternatives">${data.alternatives.map((item) =>
    `<li><strong>${escapeHtml(item.name)}</strong> · ${item.distance_metres} m · ${item.bikes_available} bikes · ${item.docks_available} docks</li>`
  ).join("")}</ol>` : "";
  detail.innerHTML = `<h2>${escapeHtml(current.name)}</h2>
    <p>${current.bikes_available} bikes now · ${current.docks_available} docks now</p>
    <p class="meta">Updated ${data.freshness_seconds}s ago</p>
    <div class="horizons" aria-label="Forecast horizon">${horizonButtons}</div>
    ${outlook}${alternatives}
    <details class="json-details"><summary>Raw API JSON</summary><pre>${rawJson}</pre></details>`;
}

stationList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-station-id]");
  if (!button) return;
  selected = button.dataset.stationId;
  renderStations();
  renderForecast();
});
detail.addEventListener("click", (event) => {
  const button = event.target.closest("[data-horizon]");
  if (!button) return;
  horizon = Number(button.dataset.horizon);
  renderForecast();
});
search.addEventListener("input", renderStations);

fetch("/api/stations").then((response) => response.json()).then((data) => {
  stations = data;
  renderStations();
});
