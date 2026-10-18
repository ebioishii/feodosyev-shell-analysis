"use strict";

// Test browser-independent form/timing logic with a small DOM and virtual clock.
// The real page layout and GIF playback are also checked in a browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const fields = new Map();
const defaults = {radius_mm: 100, inner_thickness_mm: 1, outer_thickness_mm: 2,
  inner_temperature_c: 538, outer_temperature_c: 20, k: 0.5,
  auto_strain_range: true, start_strain_percent: 0.025, max_strain_percent: 3, sample_count: 481};
let now = 0;
const context = vm.createContext({
  document: {querySelector(selector) {
    if (!fields.has(selector)) fields.set(selector, {
      value: "", checked: selector === "#auto-strain-toggle", classList: {toggle() {}, remove() {}, add() {}},
      addEventListener() {}, replaceChildren() {},
    });
    return fields.get(selector);
  }},
  window: {addEventListener() {}},
  fetch: async () => ({ok: true, json: async () => ({defaults,
    default_output_path: "E:/portable launch", max_sample_count: 5000,
    temperature_min_c: -198.3, temperature_max_c: 815.6})}),
  performance: {now: () => now},
  setTimeout(callback, delay) { now += delay; callback(); },
  requestAnimationFrame(callback) { callback(); },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../Feodosyev html/app.js"), "utf8"), context);

(async () => {
  await new Promise(setImmediate);
  assert.equal(fields.get("#file-path").value, "E:/portable launch");
  assert.equal(fields.get("#auto-strain-toggle").disabled, false);
  for (const [selector, element] of fields) {
    if (selector !== "#file-path") assert.equal(element.value, "", selector);
  }
  const model = {"#radius": 100, "#inner-thickness": 1, "#outer-thickness": 2,
    "#inner-temperature": 538, "#outer-temperature": 20, "#axial-k": 0.5};
  for (const [selector, value] of Object.entries(model)) fields.get(selector).value = String(value);
  const collect = vm.runInContext("collectInputs", context);
  assert.deepEqual(JSON.parse(JSON.stringify(collect())), defaults);
  for (const selector of ["#start-strain", "#max-strain", "#sample-count"]) {
    assert.equal(fields.get(selector).value, ""); // Auto defaults stay out of input boxes.
  }
  fields.get("#auto-strain-toggle").checked = false;
  assert.throws(collect, /start_strain_percent/);
  fields.get("#start-strain").value = "1";
  fields.get("#max-strain").value = "2";
  fields.get("#sample-count").value = "17";
  assert.equal(collect().sample_count, 17);
  assert.equal(collect().auto_strain_range, false);

  const wait = vm.runInContext("waitForOperationAndAnimation", context);
  const timing = () => Promise.resolve({startedAt: 0, loopMs: 1930});
  now = 0;
  assert.equal(await wait(Promise.resolve("fast result"), timing()), "fast result");
  assert.equal(now, 1930); // Exactly one full spin for fast operations.
  now = 5000;
  await wait(Promise.resolve("slow result"), timing());
  assert.equal(now, 5790); // Finish at the next complete cycle, not mid-spin.
  now = 0;
  await assert.rejects(wait(Promise.reject(new Error("solver failure")), timing()), /solver failure/);
  assert.equal(now, 1930);
  now = 0;
  await wait(Promise.resolve("missing image"), Promise.resolve({startedAt: 0, loopMs: 0}));
  assert.equal(now, 0);
  console.log("PASS: blank startup, portable path, Auto/Manual inputs, complete spin timing, errors");
})().catch(error => { console.error(error); process.exitCode = 1; });
