"use strict";

const elements = {
  radius: document.querySelector("#radius"),
  innerThickness: document.querySelector("#inner-thickness"),
  outerThickness: document.querySelector("#outer-thickness"),
  innerTemperature: document.querySelector("#inner-temperature"),
  outerTemperature: document.querySelector("#outer-temperature"),
  axialK: document.querySelector("#axial-k"),
  startStrain: document.querySelector("#start-strain"),
  loadJson: document.querySelector("#load-json"),
  jsonFile: document.querySelector("#json-file"),
  jsonName: document.querySelector("#json-name"),
  filePath: document.querySelector("#file-path"),
  pngName: document.querySelector("#png-name"),
  csvName: document.querySelector("#csv-name"),
  jsonExportName: document.querySelector("#json-export-name"),
  savePng: document.querySelector("#save-png"),
  saveCsv: document.querySelector("#save-csv"),
  saveJson: document.querySelector("#save-json"),
  pngState: document.querySelector("#png-state"),
  csvState: document.querySelector("#csv-state"),
  jsonState: document.querySelector("#json-state"),
  pngMessage: document.querySelector("#png-message"),
  csvMessage: document.querySelector("#csv-message"),
  jsonMessage: document.querySelector("#json-message"),
  autoStrainToggle: document.querySelector("#auto-strain-toggle"),
  manualStrainFields: document.querySelector("#manual-strain-fields"),
  maxStrain: document.querySelector("#max-strain"),
  sampleCount: document.querySelector("#sample-count"),
  generate: document.querySelector("#generate"),
  generateGif: document.querySelector("#generate-gif"),
  generateOk: document.querySelector("#generate-ok"),
  outputStatus: document.querySelector("#output-status"),
  errorPanel: document.querySelector("#error-panel"),
  errorMessage: document.querySelector("#error-message"),
  chartWrap: document.querySelector("#chart-wrap"),
  chart: document.querySelector("#result-chart"),
};

const state = {
  temperatureMin: -198.3,
  temperatureMax: 815.6,
  resultId: null,
  chartData: null,
  generating: false,
  importing: false,
  ready: false,
  saving: new Set(),
  maxSampleCount: 5000,
};

const exportControls = {
  png: {button: elements.savePng, status: elements.pngState, name: elements.pngName, message: elements.pngMessage},
  csv: {button: elements.saveCsv, status: elements.csvState, name: elements.csvName, message: elements.csvMessage},
  json: {button: elements.saveJson, status: elements.jsonState, name: elements.jsonExportName, message: elements.jsonMessage},
};

// Original oiiaioiiiai.gif: 67 frames, 5,000 ms per complete loop.
const GIF_LOOP_DURATION_MS = 5000;
let animationRun = 0;

function playWorkingAnimation(image) {
  return new Promise((resolve) => {
    image.onload = () => {
      image.onload = null;
      image.onerror = null;
      // Start the minimum playback time after the browser can paint the image.
      requestAnimationFrame(() => {
        requestAnimationFrame(() => setTimeout(resolve, GIF_LOOP_DURATION_MS));
      });
    };
    image.onerror = () => {
      image.onload = null;
      image.onerror = null;
      resolve();
    };
    animationRun += 1;
    image.src = `/assets/generating.gif?run=${Date.now()}-${animationRun}`;
  });
}

async function waitForOperationAndAnimation(operation, playback) {
  const [outcome] = await Promise.allSettled([operation, playback]);
  if (outcome.status === "rejected") throw outcome.reason;
  return outcome.value;
}

function setHidden(element, hidden) {
  element.classList.toggle("hidden", hidden);
}

function resetSaveStates() {
  for (const control of Object.values(exportControls)) {
    control.status.replaceChildren();
    control.message.textContent = "";
    control.message.classList.remove("error");
  }
}

function updateSaveButtons() {
  for (const [extension, control] of Object.entries(exportControls)) {
    control.button.disabled = state.generating || state.resultId === null || state.saving.has(extension) || state.importing;
  }
  elements.generate.disabled = !state.ready || state.generating || state.saving.size > 0 || state.importing;
}

function updateStrainMode() {
  const automatic = elements.autoStrainToggle.checked;
  setHidden(elements.manualStrainFields, automatic);
  const busy = !state.ready || state.generating || state.importing;
  elements.autoStrainToggle.disabled = busy;
  elements.startStrain.disabled = automatic || busy;
  elements.maxStrain.disabled = automatic || busy;
  elements.sampleCount.disabled = automatic || busy;
}

function makeStatusImage(kind, alt) {
  const image = document.createElement("img");
  image.src = kind === "working" ? "/assets/generating.gif" :
    kind === "success" ? "/assets/cat-ok.png" : "/assets/cat-vegetable.jpg";
  image.alt = alt;
  return image;
}

function updateMode() {
  elements.loadJson.disabled = !state.ready || state.generating || state.saving.size > 0 || state.importing;
  elements.jsonFile.disabled = elements.loadJson.disabled;
  for (const input of modelInputs) input.disabled = !state.ready || state.generating || state.importing;
}

const inputFields = {
  radius_mm: elements.radius, inner_thickness_mm: elements.innerThickness,
  outer_thickness_mm: elements.outerThickness, inner_temperature_c: elements.innerTemperature,
  outer_temperature_c: elements.outerTemperature, k: elements.axialK,
  start_strain_percent: elements.startStrain, max_strain_percent: elements.maxStrain,
  sample_count: elements.sampleCount,
};
const modelInputs = [elements.radius, elements.innerThickness, elements.outerThickness,
  elements.innerTemperature, elements.outerTemperature, elements.axialK];

function parseFiniteNumber(element, label) {
  const value = Number(element.value);
  if (element.value.trim() === "" || !Number.isFinite(value)) {
    throw new Error(`${label} 必須是有限數值。`);
  }
  return value;
}

function collectInputs() {
  return {...Object.fromEntries(Object.entries(inputFields).map(([key, input]) =>
    [key, parseFiniteNumber(input, key)])), auto_strain_range: elements.autoStrainToggle.checked};
}

function applySettings(settings) {
  // Called only after the server validates the whole object; no partial imports.
  for (const [key, input] of Object.entries(inputFields)) input.value = String(settings[key]);
  elements.autoStrainToggle.checked = settings.auto_strain_range;
  updateStrainMode();
}

function invalidateResult() {
  state.resultId = null;
  state.chartData = null;
  setHidden(elements.generateOk, true);
  setHidden(elements.chartWrap, true);
  setHidden(elements.errorPanel, true);
  elements.outputStatus.textContent = "設定已更新，請按 Generate。";
  resetSaveStates();
  updateSaveButtons();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`後端回傳無法解析的內容（HTTP ${response.status}）。`);
  }
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `請求失敗（HTTP ${response.status}）。`);
  }
  return data;
}

function beginGenerate() {
  state.generating = true;
  state.resultId = null;
  state.chartData = null;
  updateSaveButtons();
  updateStrainMode();
  updateMode();
  setHidden(elements.generateGif, false);
  setHidden(elements.generateOk, true);
  setHidden(elements.errorPanel, true);
  setHidden(elements.chartWrap, true);
  elements.outputStatus.textContent = "計算中…";
  resetSaveStates();
  return playWorkingAnimation(elements.generateGif);
}

function finishGenerate() {
  state.generating = false;
  updateSaveButtons();
  updateStrainMode();
  updateMode();
  setHidden(elements.generateGif, true);
}

function showCalculationError(message) {
  state.resultId = null;
  state.chartData = null;
  setHidden(elements.generateOk, true);
  setHidden(elements.chartWrap, true);
  setHidden(elements.errorPanel, false);
  elements.errorMessage.textContent = message;
  elements.outputStatus.textContent = "";
  updateSaveButtons();
}

function niceLabel(value) {
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2);
  }
  return Number(value.toFixed(4)).toString();
}

function drawChart() {
  if (!state.chartData) return;
  const canvas = elements.chart;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;

  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * scale);
  canvas.height = Math.round(rect.height * scale);
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);

  const width = rect.width;
  const height = rect.height;
  const margin = { left: 92, right: 30, top: 28, bottom: 78 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xValues = state.chartData.strain;
  const yValues = state.chartData.pressure;
  const rawXMin = Math.min(...xValues);
  const rawXMax = Math.max(...xValues);
  const rawYMin = Math.min(...yValues);
  const rawYMax = Math.max(...yValues);
  let xMin = rawXMin;
  let xMax = rawXMax;
  let yMin = rawYMin;
  let yMax = rawYMax;
  if (xMin === xMax) [xMin, xMax] = [xMin - 1, xMax + 1];
  if (yMin === yMax) [yMin, yMax] = [yMin - 1, yMax + 1];
  const xPadding = Math.max((xMax - xMin) * 0.02, 0.001);
  const yPadding = Math.max((yMax - yMin) * 0.06, 0.05);
  xMin = rawXMin >= 0 ? 0 : xMin - xPadding;
  xMax += xPadding;
  yMin = rawYMin >= 0 ? 0 : yMin - yPadding;
  yMax += yPadding;

  const xToPixel = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const yToPixel = (value) => margin.top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.font = "14px Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "top";

  const tickCount = 6;
  context.strokeStyle = "rgba(70, 70, 70, 0.18)";
  context.lineWidth = 1;
  context.fillStyle = "#303030";
  for (let index = 0; index <= tickCount; index += 1) {
    const fraction = index / tickCount;
    const x = margin.left + fraction * plotWidth;
    const xValue = xMin + fraction * (xMax - xMin);
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, margin.top + plotHeight);
    context.stroke();
    context.fillText(niceLabel(xValue), x, margin.top + plotHeight + 12);

    const y = margin.top + plotHeight - fraction * plotHeight;
    const yValue = yMin + fraction * (yMax - yMin);
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(niceLabel(yValue), margin.left - 12, y);
    context.textAlign = "center";
    context.textBaseline = "top";
  }

  context.strokeStyle = "#333333";
  context.lineWidth = 1.5;
  context.strokeRect(margin.left, margin.top, plotWidth, plotHeight);

  context.strokeStyle = "#287fbd";
  context.lineWidth = 2.5;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();
  xValues.forEach((xValue, index) => {
    const x = xToPixel(xValue);
    const y = yToPixel(yValues[index]);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();

  context.fillStyle = "#287fbd";
  context.beginPath();
  context.arc(xToPixel(xValues[0]), yToPixel(yValues[0]), 5.5, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = "#222222";
  context.font = "16px Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "bottom";
  context.fillText("Total Hoop Strain ΔR/R (%)", margin.left + plotWidth / 2, height - 10);

  context.save();
  context.translate(22, margin.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.textBaseline = "top";
  context.fillText("Calculated Gas Pressure p_g (MPa)", 0, 0);
  context.restore();
}

async function generate() {
  if (!state.ready || state.generating || state.importing || state.saving.size > 0) return;
  let payload;
  try {
    payload = collectInputs();
  } catch (error) {
    showCalculationError(error.message);
    finishGenerate();
    return;
  }

  const playback = beginGenerate();
  try {
    const result = await waitForOperationAndAnimation(
      postJson("/api/generate", payload), playback,
    );
    state.resultId = result.result_id;
    state.chartData = {
      strain: result.strain_percent,
      pressure: result.pressure_mpa,
    };
    setHidden(elements.errorPanel, true);
    setHidden(elements.chartWrap, false);
    setHidden(elements.generateOk, false);
    elements.outputStatus.textContent =
      `計算完成：${result.point_count} 個${payload.auto_strain_range ? "自動" : "指定"}取樣點，` +
      `範圍 ${niceLabel(result.strain_percent[0])}% 至 ${niceLabel(result.strain_percent.at(-1))}%。`;
    requestAnimationFrame(drawChart);
  } catch (error) {
    showCalculationError(error.message);
  } finally {
    finishGenerate();
  }
}

async function save(extension) {
  if (state.saving.has(extension) || state.generating) return;
  const {status, name, message} = exportControls[extension];
  const filename = name.value.trim();
  const filePath = elements.filePath.value.trim();
  const resultId = state.resultId;

  if (!state.resultId) {
    message.textContent = "目前沒有可儲存的成功計算結果，請先按 Generate。";
    message.classList.add("error");
    return;
  }
  if (!filePath || !filename) {
    message.textContent = "File Path 與檔名都不可留白。";
    message.classList.add("error");
    status.replaceChildren(makeStatusImage("error", "Save error"));
    return;
  }

  state.saving.add(extension);
  updateSaveButtons();
  updateMode();
  const workingImage = makeStatusImage("working", `Saving ${extension.toUpperCase()}`);
  status.replaceChildren(workingImage);
  const playback = playWorkingAnimation(workingImage);
  message.textContent = `正在儲存 ${extension.toUpperCase()}…`;
  message.classList.remove("error");
  try {
    const result = await waitForOperationAndAnimation(
      postJson(`/api/save/${extension}`, {
        result_id: resultId,
        file_path: filePath,
        filename,
      }),
      playback,
    );
    status.replaceChildren(makeStatusImage("success", `${extension.toUpperCase()} saved`));
    message.textContent = `已儲存：${result.saved_path}`;
  } catch (error) {
    status.replaceChildren(makeStatusImage("error", "Save error"));
    message.textContent = error.message;
    message.classList.add("error");
  } finally {
    state.saving.delete(extension);
    updateSaveButtons();
    updateMode();
  }
}

elements.autoStrainToggle.addEventListener("change", () => {
  invalidateResult();
  updateStrainMode();
});
for (const input of Object.values(inputFields)) input.addEventListener("input", invalidateResult);
elements.loadJson.addEventListener("click", () => elements.jsonFile.click());
elements.jsonFile.addEventListener("change", async () => {
  const file = elements.jsonFile.files[0];
  if (!file) return;
  state.importing = true;
  updateSaveButtons();
  updateMode();
  updateStrainMode();
  try {
    if (file.size > 4 * 1024 * 1024) throw new Error("JSON 超過 4 MB。");
    const parsed = JSON.parse(await file.text());
    const result = await postJson("/api/import", parsed);
    applySettings(result.settings);
    invalidateResult();
    elements.jsonName.textContent = file.name;
  } catch (error) {
    showCalculationError(`Import Data: ${error.message}`);
  } finally {
    elements.jsonFile.value = "";
    state.importing = false;
    updateSaveButtons();
    updateMode();
    updateStrainMode();
  }
});
elements.generate.addEventListener("click", generate);
elements.savePng.addEventListener("click", () => save("png"));
elements.saveCsv.addEventListener("click", () => save("csv"));
elements.saveJson.addEventListener("click", () => save("json"));
window.addEventListener("resize", () => {
  if (state.chartData) requestAnimationFrame(drawChart);
});

fetch("/api/config")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((config) => {
    state.ready = true;
    applySettings(config.defaults);
    elements.filePath.value = config.default_output_path;
    for (const input of [elements.pngName, elements.csvName, elements.jsonExportName]) input.value = "feodosyev";
    state.temperatureMin = config.temperature_min_c;
    state.temperatureMax = config.temperature_max_c;
    state.maxSampleCount = config.max_sample_count;
    elements.sampleCount.max = String(state.maxSampleCount);
    elements.innerTemperature.min = String(state.temperatureMin);
    elements.innerTemperature.max = String(state.temperatureMax);
    elements.outerTemperature.min = String(state.temperatureMin);
    elements.outerTemperature.max = String(state.temperatureMax);
    updateSaveButtons();
    updateMode();
  })
  .catch((error) => {
    showCalculationError(`無法讀取後端設定：${error.message}`);
  });

updateMode();
updateStrainMode();
updateSaveButtons();
