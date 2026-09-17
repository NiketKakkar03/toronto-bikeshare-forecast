const stationList = document.querySelector("#stations");
const search = document.querySelector("#search");
const detail = document.querySelector("#detail");
let stations = [];
let selected = null;
let horizon = 30;

const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
})[character]);
const percent = (value) => `${Math.round(value * 100)}%`;
const riskClass = (value) => value >= .5 ? "risk-high" : value >= .25 ? "risk-medium" : "";

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
  const horizonButtons = [15, 30, 60].map((minutes) =>
    `<button data-horizon="${minutes}" class="${minutes === horizon ? "active" : ""}">${minutes} min</button>`
  ).join("");
  let outlook = `<p class="notice"><strong>Forecast ${escapeHtml(data.state)}.</strong> ${escapeHtml(data.reason || "Try again shortly.")}</p>`;
  if (data.forecast) {
    const value = data.forecast;
    outlook = `<div class="metrics">
      <div class="metric"><span>Expected bikes</span><strong>${value.bikes_expected}</strong><small>range ${value.bikes_interval[0]}–${value.bikes_interval[1]}</small></div>
      <div class="metric"><span>Expected docks</span><strong>${value.docks_expected}</strong><small>range ${value.docks_interval[0]}–${value.docks_interval[1]}</small></div>
      <div class="metric"><span>Empty risk</span><strong class="${riskClass(value.empty_risk)}">${percent(value.empty_risk)}</strong></div>
      <div class="metric"><span>Full risk</span><strong class="${riskClass(value.full_risk)}">${percent(value.full_risk)}</strong></div>
    </div>`;
  }
  const alternatives = data.alternatives.length ? `<h3>Nearby alternatives</h3><ol class="alternatives">${data.alternatives.map((item) =>
    `<li><strong>${escapeHtml(item.name)}</strong> · ${item.distance_metres} m · empty risk ${percent(item.empty_risk)}</li>`
  ).join("")}</ol>` : "";
  detail.innerHTML = `<h2>${escapeHtml(current.name)}</h2>
    <p>${current.bikes_available} bikes now · ${current.docks_available} docks now</p>
    <p class="meta">Updated ${data.freshness_seconds}s ago · data ${escapeHtml(current.data_version)}</p>
    <p class="coordinate">${current.latitude.toFixed(4)}, ${current.longitude.toFixed(4)}</p>
    <div class="horizons" aria-label="Forecast horizon">${horizonButtons}</div>
    ${outlook}${alternatives}`;
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
