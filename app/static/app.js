"use strict";

/* Text for the page itself (buttons, tips, labels), in both languages.
   Everything medical or safety-related (diagnosis, treatment, caveats, disclaimers) comes from the server,
   which reads it from knowledge_base/treatments_en.json and treatments_hi.json. */
const STRINGS = {
  en: {
    appName: "Tomato Leaf Check",
    languageLabel: "Language",
    heroTitle: "Is your tomato leaf healthy?",
    heroSub: "Upload a photo of one leaf. The app checks the photo, names the disease it finds, and lists what you can do about it.",
    camera: "Take photo",
    choose: "Choose photo",
    dropHint: "or drop a photo here",
    tipsTitle: "For a good photo",
    tip1: "One leaf only",
    tip2: "Plain, light background (a sheet of paper works)",
    tip3: "Daylight and a steady hand, no flash glare",
    analysing: "Checking your photo…",
    again: "Check another photo",
    tryAgain: "Try again",
    yourPhoto: "Your photo",
    heatmap: "Where the model looked",
    heatNote: "Drag the handle. Warmer colours mark the parts of the leaf that influenced the result most.",
    confidence: "Model confidence",
    closest: "Closest match",
    other: "Other possibilities",
    cause: "Cause",
    sources: "Sources",
    techTitle: "How this photo was checked",
    techSim: "Similarity to known tomato leaves",
    techConf: "Model confidence",
    techNeeds: "needs at least",
    uncertainNoTreatment: "No treatment is shown because the result is uncertain. Retake the photo or ask a local agricultural expert.",
    cameraTitle: "Take a photo",
    capture: "Capture",
    cancel: "Cancel",
    cameraError: "The camera could not be started. Allow camera access in your browser, or choose a photo instead.",
    errorNetwork: "Could not reach the app. Check that the server is running, then try again.",
    errorTooBig: "This file is larger than 10 MB. Please choose a smaller photo.",
    errorGeneric: "Something went wrong while checking the photo. Please try again.",
    footer: "Student project trained on the PlantVillage tomato leaf dataset."
  },
  hi: {
    appName: "टमाटर पत्ती जाँच",
    languageLabel: "भाषा",
    heroTitle: "क्या आपके टमाटर का पत्ता स्वस्थ है?",
    heroSub: "एक पत्ते की फोटो अपलोड करें। ऐप फोटो को जाँचता है, मिली बीमारी का नाम बताता है और उसके लिए क्या किया जा सकता है, यह सुझाता है।",
    camera: "फोटो खींचें",
    choose: "फोटो चुनें",
    dropHint: "या फोटो यहाँ छोड़ें",
    tipsTitle: "अच्छी फोटो के लिए",
    tip1: "सिर्फ़ एक पत्ता",
    tip2: "सादी, हल्के रंग की पृष्ठभूमि (सफ़ेद काग़ज भी चलेगा)",
    tip3: "दिन की रोशनी और स्थिर हाथ, फ़्लैश की चमक नहीं",
    analysing: "आपकी फोटो जाँची जा रही है…",
    again: "दूसरी फोटो जाँचें",
    tryAgain: "फिर कोशिश करें",
    yourPhoto: "आपकी फोटो",
    heatmap: "मॉडल ने कहाँ देखा",
    heatNote: "हैंडल को खींचें। गर्म रंग (लाल-पीला) पत्ते के उन हिस्सों को दिखाते हैं जिनका नतीजे पर सबसे ज़्यादा असर रहा।",
    confidence: "मॉडल का भरोसा",
    closest: "सबसे नज़दीकी अनुमान",
    other: "अन्य संभावनाएँ",
    cause: "कारण",
    sources: "स्रोत",
    techTitle: "इस फोटो की जाँच कैसे हुई",
    techSim: "जाने-पहचाने टमाटर के पत्तों से समानता",
    techConf: "मॉडल का भरोसा",
    techNeeds: "कम से कम चाहिए",
    uncertainNoTreatment: "नतीजा अनिश्चित है, इसलिए उपचार नहीं दिखाया गया। फोटो दोबारा लें या किसी स्थानीय कृषि विशेषज्ञ से पूछें।",
    cameraTitle: "फोटो खींचें",
    capture: "फोटो लें",
    cancel: "रद्द करें",
    cameraError: "कैमरा शुरू नहीं हो सका। ब्राउज़र में कैमरे की अनुमति दें, या फोटो चुनें।",
    errorNetwork: "ऐप से संपर्क नहीं हो सका। जाँचें कि सर्वर चालू है, फिर कोशिश करें।",
    errorTooBig: "यह फ़ाइल 10 MB से बड़ी है। कृपया छोटी फोटो चुनें।",
    errorGeneric: "फोटो की जाँच में कुछ गड़बड़ हो गई। कृपया फिर कोशिश करें।",
    footer: "विद्यार्थी परियोजना, PlantVillage टमाटर पत्ती डेटासेट पर प्रशिक्षित।"
  }
};

const ICONS = {
  ok: '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  warn: '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M9.3 9.4a2.7 2.7 0 015.3.6c0 1.8-2.6 2.2-2.6 4M12 18h.01"/></svg>',
  info: '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M12 11v6M12 7.5h.01"/></svg>'
};

const MAX_BYTES = 10 * 1024 * 1024;
const $ = (sel, root = document) => root.querySelector(sel);

let lang = "en";
let lastBlob = null;       // the photo currently on screen (so a language switch can re-check it)
let photoUrl = null;
let inflight = null;

/* ---------- small helpers ---------- */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function t(key) { return (STRINGS[lang] && STRINGS[lang][key]) || STRINGS.en[key] || key; }

/* Writes text into a node and turns source tags like [S01, S07] into small citation chips. */
function richText(node, text) {
  const re = /\[(S\d{2}(?:,\s*S\d{2})*)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) node.append(document.createTextNode(text.slice(last, m.index).replace(/\s+$/, " ")));
    node.append(el("span", "cite", m[1]));
    last = re.lastIndex;
  }
  if (last < text.length) node.append(document.createTextNode(text.slice(last)));
  return node;
}

function applyStatic() {
  document.documentElement.lang = lang;
  document.title = t("appName");
  document.querySelectorAll("[data-i18n]").forEach(n => { n.textContent = t(n.dataset.i18n); });
  document.querySelectorAll("[data-i18n-aria]").forEach(n => { n.setAttribute("aria-label", t(n.dataset.i18nAria)); });
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  $(".brand").setAttribute("aria-label", t("appName"));
}

function setView(v) {
  document.body.dataset.view = v;
  $("#home").hidden = v !== "home";
  $("#result").hidden = v !== "result";
  window.scrollTo(0, 0);
}

/* ---------- photo handling ---------- */
async function prepare(file) {
  // Big phone photos are shrunk before upload (the model works on 224 px anyway).
  if (file.size <= 3 * 1024 * 1024) return file;
  try {
    const bmp = await createImageBitmap(file);
    const scale = Math.min(1, 1600 / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * scale);
    c.height = Math.round(bmp.height * scale);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise(r => c.toBlob(r, "image/jpeg", 0.9));
    return blob || file;
  } catch (_) {
    return file;
  }
}

function showPhoto(blob) {
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(blob);
  const img = $("#img-orig");
  const frame = $("#frame");
  $("#viewer").hidden = false;
  img.onload = () => { frame.style.setProperty("--ar", (img.naturalWidth / img.naturalHeight).toFixed(4)); };
  img.onerror = () => { $("#viewer").hidden = true; };      // not a readable image: show only the message
  img.src = photoUrl;
  resetSlider();
}

function resetSlider() {
  const frame = $("#frame");
  frame.style.setProperty("--pos", "100%");
  $("#img-heat").hidden = true; $("#img-heat").removeAttribute("src");
  $("#handle").hidden = true; $("#slider").hidden = true;
  $("#tag-l").hidden = true; $("#tag-r").hidden = true; $("#heat-note").hidden = true;
}

function enableSlider(b64) {
  const frame = $("#frame"), slider = $("#slider");
  const heat = $("#img-heat");
  heat.src = "data:image/png;base64," + b64;
  heat.hidden = false;
  $("#handle").hidden = false; slider.hidden = false;
  $("#tag-l").textContent = t("yourPhoto"); $("#tag-r").textContent = t("heatmap");
  $("#tag-l").hidden = false; $("#tag-r").hidden = false;
  $("#heat-note").textContent = t("heatNote"); $("#heat-note").hidden = false;
  slider.setAttribute("aria-label", t("heatmap"));
  slider.value = 100;
  const set = v => { frame.style.setProperty("--pos", v + "%"); slider.value = v; };
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { set(50); return; }
  // one short reveal so people notice there are two layers
  const from = 100, to = 52, dur = 900, t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    set(from + (to - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------- camera ----------
   Phones and tablets: the file input with capture="environment" opens the camera app.
   Laptops ignore that, so there we show a live camera view (works on localhost). */
let camStream = null;

function stopCamera() {
  if (camStream) { camStream.getTracks().forEach(tr => tr.stop()); camStream = null; }
  $("#cam-video").srcObject = null;
}

async function openCamera() {
  const touch = window.matchMedia("(pointer: coarse)").matches;
  if (touch || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { $("#file-camera").click(); return; }
  const dlg = $("#cam"), msg = $("#cam-msg"), shot = $("#cam-shot");
  msg.hidden = true; shot.disabled = true;
  dlg.showModal();
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    $("#cam-video").srcObject = camStream;
    shot.disabled = false;
  } catch (_) {
    msg.textContent = t("cameraError"); msg.hidden = false;
  }
}

function capturePhoto() {
  const v = $("#cam-video");
  if (!v.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(b => { $("#cam").close(); if (b) handleFile(b); }, "image/jpeg", 0.92);
}

/* ---------- talking to the backend ---------- */
async function handleFile(file) {
  if (!file) return;
  const blob = await prepare(file);
  lastBlob = blob;
  setView("result");
  showPhoto(blob);
  await analyze();
}

async function analyze() {
  if (!lastBlob) return;
  if (inflight) inflight.abort();
  inflight = new AbortController();
  const frame = $("#frame");
  frame.dataset.busy = "true";
  resetSlider();
  renderBusy();
  const fd = new FormData();
  fd.append("file", lastBlob, "photo.jpg");
  fd.append("lang", lang);
  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd, signal: inflight.signal });
    if (r.status === 413) throw new Error("errorTooBig");
    if (!r.ok) throw new Error("errorGeneric");
    const res = await r.json();
    frame.dataset.busy = "false";
    renderResult(res);
  } catch (err) {
    if (err.name === "AbortError") return;
    frame.dataset.busy = "false";
    renderError(err instanceof TypeError ? "errorNetwork" : (STRINGS.en[err.message] ? err.message : "errorGeneric"));
  }
}

/* ---------- rendering ---------- */
function banner(kind, title, lines) {
  const b = el("div", "banner s-" + kind);
  const ico = el("div", "ico"); ico.innerHTML = ICONS[kind];
  const body = el("div");
  body.append(el("h2", null, title));
  (lines || []).forEach(l => { if (l) body.append(el("p", null, l)); });
  b.append(ico, body);
  return b;
}

function renderBusy() {
  const v = $("#verdict");
  v.replaceChildren();
  v.append(banner("info", t("analysing")));
  const sk = el("div", "skel");
  for (let i = 0; i < 4; i++) sk.append(el("i"));
  v.append(sk);
}

function renderError(key) {
  const v = $("#verdict");
  v.replaceChildren(banner("info", t(key)));
  const again = el("button", "btn primary again", t("tryAgain"));
  again.type = "button";
  again.addEventListener("click", analyze);
  v.append(again);
}

function listSection(parent, heading, items, open) {
  if (!items || !items.length) return;
  const d = el("details"); if (open) d.open = true;
  d.append(el("summary", null, heading));
  const body = el("div", "body"), ul = el("ul");
  items.forEach(txt => ul.append(richText(el("li"), txt)));
  body.append(ul); d.append(body); parent.append(d);
}

function renderResult(res) {
  const v = $("#verdict");
  v.replaceChildren();
  const m = res.messages || {};
  const ui = {
    kind: { reliable: "ok", uncertain: "warn", unsupported: "info", quality_failed: "info" }[res.status] || "info"
  };

  // 1. status banner
  if (res.status === "quality_failed") {
    const msgs = res.quality.messages || [];
    v.append(banner("info", msgs[0] || res.headline, [...msgs.slice(1), m.scope]));
  } else if (res.status === "reliable") {
    v.append(banner("ok", res.headline));
  } else {
    v.append(banner(ui.kind, res.headline, [m.scope]));
  }

  // 2. the photo viewer gets its heat map
  if (res.gradcam_png_base64) enableSlider(res.gradcam_png_base64);

  // 3. diagnosis
  const p = res.prediction, tr = res.treatment;
  if (p) {
    const diag = el("div", "diag");
    if (res.status === "uncertain") diag.append(el("p", "closest", t("closest")));
    diag.append(el("h2", null, p.class_name));
    if (tr && tr.pathogen) diag.append(el("p", "cause", t("cause") + ": " + tr.pathogen));
    const conf = el("div", "conf");
    const row = el("div", "conf-row");
    row.append(el("span", "conf-num", p.confidence_text), el("span", "conf-label", t("confidence")));
    const bar = el("div", "bar"), fill = el("i");
    fill.style.width = (Math.min(p.confidence, 0.99) * 100).toFixed(1) + "%";
    bar.append(fill); conf.append(row, bar); diag.append(conf);
    if (tr && tr.summary) diag.append(el("p", "summary", tr.summary));
    v.append(diag);
  }

  if (res.status === "uncertain") {
    if (res.alternatives && res.alternatives.length) {
      const box = el("div", "alts");
      box.append(el("h3", null, t("other")));
      const ul = el("ul"); ul.style.cssText = "list-style:none;padding:0;margin:0";
      res.alternatives.forEach(a => {
        const li = el("li"); li.append(el("b", null, a.class_name), el("span", null, a.confidence_text));
        ul.append(li);
      });
      box.append(ul); v.append(box);
    }
    v.append(el("p", "nonote", t("uncertainNoTreatment")));
  }

  // 4. treatment (reliable only)
  if (tr) {
    const ui2 = window.__ui || {};
    const acc = el("div", "acc");
    const mg = tr.management || {};
    listSection(acc, ui2.section_symptoms || "Symptoms", tr.symptoms, true);
    listSection(acc, ui2.section_cultural || "Cultural practices", mg.cultural, true);
    listSection(acc, ui2.section_biological || "Biological options", mg.biological, false);
    listSection(acc, ui2.section_chemical || "Chemical groups", mg.chemical_groups, false);
    listSection(acc, ui2.section_conditions || "Favourable conditions", tr.favorable_conditions, false);
    if (mg.resistant_varieties) listSection(acc, ui2.section_varieties || "Resistant varieties", [mg.resistant_varieties], false);
    if (acc.children.length) v.append(acc);
  }

  // 5. notes: caveat, expert review, disclaimer
  const notes = el("div", "notes");
  ["caveat", "expert_review", "disclaimer"].forEach(k => { if (m[k]) notes.append(el("p", null, m[k])); });
  if (notes.children.length) v.append(notes);

  // 6. sources
  if (tr && tr.sources && tr.sources.length) {
    const s = el("div", "src");
    s.append(el("h3", null, t("sources")));
    const dl = el("dl");
    tr.sources.forEach(x => { const dt = el("dt"); dt.append(el("span", "cite", x.id)); dl.append(dt, el("dd", null, x.name)); });
    s.append(dl); v.append(s);
  }

  // 7. how the photo was checked
  if (res.gate) {
    const d = el("details", "tech");
    d.append(el("summary", null, t("techTitle")));
    const dl = el("dl");
    const g = res.gate;
    dl.append(el("dt", null, t("techSim")),
              el("dd", null, g.knn_similarity.toFixed(3) + " (" + t("techNeeds") + " " + g.tau_knn.toFixed(3) + ")"),
              el("dt", null, t("techConf")),
              el("dd", null, (g.confidence * 100).toFixed(1) + "% (" + t("techNeeds") + " " + (g.tau_conf * 100).toFixed(1) + "%)"));
    d.append(dl); v.append(d);
  }

  const again = el("button", "btn ghost again", t("again"));
  again.type = "button";
  again.addEventListener("click", reset);
  v.append(again);
}

function reset() {
  if (inflight) inflight.abort();
  lastBlob = null;
  if (photoUrl) { URL.revokeObjectURL(photoUrl); photoUrl = null; }
  $("#file-choose").value = ""; $("#file-camera").value = "";
  resetSlider();
  setView("home");
}

/* ---------- language ---------- */
async function loadUi() {
  try {
    const r = await fetch("/api/ui?lang=" + lang);
    if (r.ok) window.__ui = await r.json();
  } catch (_) { /* section headings fall back to English */ }
}

async function setLang(next) {
  if (next === lang) return;
  lang = next;
  try { localStorage.setItem("lang", lang); } catch (_) {}
  applyStatic();
  await loadUi();
  if (lastBlob) await analyze();      // re-check so the diagnosis and treatment come back in the new language
}

/* ---------- start-up ---------- */
function init() {
  let stored = null;
  try { stored = localStorage.getItem("lang"); } catch (_) {}
  lang = (stored === "en" || stored === "hi") ? stored : ((navigator.language || "").toLowerCase().startsWith("hi") ? "hi" : "en");
  applyStatic();
  loadUi();

  document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));
  $("#btn-choose").addEventListener("click", () => $("#file-choose").click());
  $("#btn-camera").addEventListener("click", openCamera);
  $("#cam-shot").addEventListener("click", capturePhoto);
  $("#cam-cancel").addEventListener("click", () => $("#cam").close());
  $("#cam").addEventListener("close", stopCamera);      // also runs on Esc, so the camera light always goes off
  $("#btn-back").addEventListener("click", reset);
  [$("#file-choose"), $("#file-camera")].forEach(inp => inp.addEventListener("change", () => handleFile(inp.files && inp.files[0])));

  const drop = $("#drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => handleFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]));
  ["dragover", "drop"].forEach(ev => window.addEventListener(ev, e => e.preventDefault()));   // don't navigate away on a stray drop

  const slider = $("#slider");
  slider.addEventListener("input", () => $("#frame").style.setProperty("--pos", slider.value + "%"));
}
init();