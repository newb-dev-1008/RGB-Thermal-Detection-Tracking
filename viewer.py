#!/usr/bin/env python3
"""Zero-dependency local web viewer for Challenge 2: Day RGB + Thermal IR ISR."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
import urllib.parse
import webbrowser
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_CATEGORIES: list[dict[str, Any]] = [
    {"id": 0, "name": "armored_vehicle"},
    {"id": 1, "name": "vehicle"},
    {"id": 2, "name": "person"},
]
SMALL_AREA_PX = 32 * 32

_EXTRA_MIME_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
}

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Challenge 2 Viewer — RGB + Thermal IR ISR</title>
<style>
  :root {
    --bg: #0b1117;
    --panel: #131c26;
    --panel-alt: #182330;
    --border: #273647;
    --text: #e6edf3;
    --muted: #8b9bb0;
    --rgb: #38bdf8;
    --thermal: #fb7185;
    --good: #10b981;
    --warn: #f59e0b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
  }
  header h1 { margin: 0; font-size: 18px; }
  header .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .tabs { display: flex; gap: 8px; }
  .tab-btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 7px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
  }
  .tab-btn.active { background: var(--rgb); color: #062129; border-color: var(--rgb); }
  .container { padding: 16px 20px; max-width: 1560px; margin: 0 auto; }
  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
  }
  .stat-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .stat-val { font-size: 18px; font-weight: 700; margin-top: 3px; }
  .banner {
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.45);
    color: #fde68a;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 13px;
    display: none;
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 14px;
  }
  .toolbar label { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  select, input[type="text"], input[type="range"] {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 5px 8px;
    font-size: 13px;
  }
  button.btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 5px 10px;
    cursor: pointer;
    font-size: 12px;
  }
  button.btn:hover { border-color: var(--rgb); }
  .split-layout {
    display: grid;
    grid-template-columns: minmax(320px, 1.05fr) minmax(380px, 1.35fr);
    gap: 14px;
    align-items: start;
  }
  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 4px;
  }
  .thumb-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 6px;
    cursor: pointer;
  }
  .thumb-card.selected { border-color: var(--rgb); box-shadow: 0 0 0 1px var(--rgb); }
  .thumb-card img {
    width: 100%;
    height: 92px;
    object-fit: cover;
    border-radius: 4px;
    background: #05080c;
    display: block;
  }
  .thumb-meta {
    margin-top: 5px;
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    gap: 4px;
    color: var(--muted);
  }
  .badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--panel-alt);
    color: var(--text);
  }
  .badge.rgb { border-color: var(--rgb); color: var(--rgb); }
  .badge.thermal { border-color: var(--thermal); color: var(--thermal); }
  .badge.pred { border-color: var(--good); color: var(--good); }
  .inspector {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }
  .canvas-wrap {
    position: relative;
    background: #05080c;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 360px;
  }
  canvas { max-width: 100%; max-height: 62vh; display: block; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 12px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
  th, td { border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: left; }
  th { color: var(--muted); font-weight: 600; }
  .pager { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 12px; color: var(--muted); }
  .mot-grid { display: grid; grid-template-columns: 290px 1fr; gap: 14px; }
  .seq-list { display: flex; flex-direction: column; gap: 8px; max-height: 75vh; overflow-y: auto; }
  .seq-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px;
    cursor: pointer;
  }
  .seq-card.selected { border-color: var(--rgb); box-shadow: 0 0 0 1px var(--rgb); }
  .video-stage {
    position: relative;
    display: inline-block;
    max-width: 100%;
    background: #05080c;
    border-radius: 6px;
    overflow: hidden;
  }
  .video-stage video { display: block; max-width: 100%; max-height: 62vh; }
  .video-stage canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Challenge 2: Day RGB + Thermal IR ISR Detection &amp; Multi-Stream Tracking</h1>
    <div class="sub">Classes: <code>armored_vehicle</code>, <code>vehicle</code>, <code>person</code> &middot; Balanced EO/IR Evaluation</div>
  </div>
  <div class="tabs">
    <button class="tab-btn active" id="tabBtnDet" type="button">RGB / Thermal Explorer</button>
    <button class="tab-btn" id="tabBtnMot" type="button">Multi-Stream Tracking</button>
  </div>
</header>

<div class="container">
  <div id="detView">
    <div class="banner" id="gtHiddenBanner">Held-out <code>test</code> split: ground-truth boxes are stripped in public archives so teams only evaluate locally on <code>valid</code>.</div>
    <div class="stats-row" id="detStatsRow"></div>
    <div class="toolbar">
      <label>Split <select id="splitSelect"></select></label>
      <label>Modality
        <select id="modalityFilter">
          <option value="all">All Modalities</option>
          <option value="rgb">RGB Day</option>
          <option value="thermal">Thermal IR</option>
        </select>
      </label>
      <label>Class
        <select id="classFilter">
          <option value="all">All images</option>
          <option value="0">armored_vehicle</option>
          <option value="1">vehicle</option>
          <option value="2">person</option>
          <option value="neg">Empty / Negative</option>
          <option value="has_pred">Has Predictions (--preds)</option>
        </select>
      </label>
      <label>Display Filter
        <select id="displayFilter">
          <option value="normal">Normal</option>
          <option value="contrast">Thermal Contrast Boost</option>
          <option value="invert">Thermal Polarity Invert</option>
        </select>
      </label>
      <label>Search <input id="searchBox" type="text" placeholder="c2_0001" /></label>
      <label><input id="showGt" type="checkbox" checked /> GT</label>
      <label><input id="showPred" type="checkbox" checked /> Preds</label>
      <label>Thr <input id="scoreSlider" type="range" min="0" max="0.95" step="0.05" value="0.25" /> <span id="scoreVal">0.25</span></label>
    </div>

    <div class="split-layout">
      <div>
        <div class="thumb-grid" id="thumbGrid"></div>
        <div class="pager">
          <button class="btn" id="prevPage" type="button">Prev</button>
          <span id="pageInfo">Page 1</span>
          <button class="btn" id="nextPage" type="button">Next</button>
        </div>
      </div>
      <div class="inspector">
        <div class="canvas-wrap"><canvas id="detCanvas"></canvas></div>
        <div class="legend">
          <span><i class="dot" style="background:#38bdf8"></i>armored_vehicle</span>
          <span><i class="dot" style="background:#f59e0b"></i>vehicle</span>
          <span><i class="dot" style="background:#f43f5e"></i>person</span>
          <span><i class="dot" style="background:#10b981"></i>Prediction (dashed)</span>
        </div>
        <div id="detMetaSummary" style="margin-top:8px;font-size:12px;color:var(--muted);"></div>
        <table id="detBoxTable"><thead><tr><th>Source</th><th>Class</th><th>BBox [x, y, w, h]</th><th>Area</th><th>Track / Score</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <div id="motView" style="display:none;">
    <div class="toolbar">
      <label>Stream Modality
        <select id="streamModFilter">
          <option value="all">All (RGB + Thermal)</option>
          <option value="thermal">Thermal IR</option>
          <option value="rgb">RGB Day</option>
        </select>
      </label>
      <label>Thermal Display
        <select id="motDisplayFilter">
          <option value="normal">Normal</option>
          <option value="contrast">Contrast Boost</option>
          <option value="invert">Polarity Invert</option>
        </select>
      </label>
      <label><input id="motShowGt" type="checkbox" checked /> Show MOT GT</label>
      <label><input id="motShowPred" type="checkbox" checked /> Show MOT Preds</label>
      <button class="btn" id="stepBackBtn" type="button">-1 Frame</button>
      <button class="btn" id="playPauseBtn" type="button">Play / Pause</button>
      <button class="btn" id="stepFwdBtn" type="button">+1 Frame</button>
      <span id="motFrameLabel" style="font-size:13px;color:var(--rgb);font-weight:600;">Frame 1</span>
    </div>
    <div class="mot-grid">
      <div class="seq-list" id="seqList"></div>
      <div class="inspector">
        <div id="motHeaderInfo" style="margin-bottom:8px;font-size:13px;color:var(--muted);"></div>
        <div class="video-stage">
          <video id="motVideo" controls preload="metadata" playsinline></video>
          <canvas id="motCanvas"></canvas>
        </div>
        <table id="motFrameTable"><thead><tr><th>Source</th><th>Track ID</th><th>Class</th><th>BBox [x, y, w, h]</th><th>Conf</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
</div>

<script>
(() => {
  const CLASS_COLORS = {0: "#38bdf8", 1: "#f59e0b", 2: "#f43f5e"};
  const CLASS_NAMES = {0: "armored_vehicle", 1: "vehicle", 2: "person"};
  const PAGE_SIZE = 36;
  let state = null;
  let filteredImages = [];
  let page = 0;
  let selectedId = null;
  let currentSeq = null;
  let currentSeqData = null;

  function el(id) { return document.getElementById(id); }

  function mediaUrl(rel) {
    return "/media/" + String(rel).split("/").map(encodeURIComponent).join("/");
  }

  function cssFilterFor(mode) {
    if (mode === "contrast") return "contrast(1.45) brightness(1.08)";
    if (mode === "invert") return "invert(1) contrast(1.25)";
    return "none";
  }

  function renderTabs(which) {
    const detActive = which === "det";
    el("detView").style.display = detActive ? "block" : "none";
    el("motView").style.display = detActive ? "none" : "block";
    el("tabBtnDet").classList.toggle("active", detActive);
    el("tabBtnMot").classList.toggle("active", !detActive);
  }

  function populateSplits() {
    const sel = el("splitSelect");
    sel.innerHTML = "";
    const order = ["valid", "train", "test", "calibration"];
    const keys = Object.keys(state.splits).sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    for (const k of keys) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${k} (${state.splits[k].counts.image_count})`;
      sel.appendChild(opt);
    }
  }

  function renderStats(splitObj) {
    const c = splitObj.counts;
    const mc = c.modality_counts || {};
    const rgbC = mc.rgb || {images: 0, boxes: 0};
    const irC = mc.thermal || {images: 0, boxes: 0};
    const modHidden = !splitObj.gt_available && (rgbC.images + irC.images === 0);
    const row = el("detStatsRow");
    row.innerHTML = "";
    const cards = [
      ["Total Images", c.image_count],
      ["RGB Day", modHidden ? "Hidden (test)" : `${rgbC.images} imgs / ${splitObj.gt_available ? rgbC.boxes + " boxes" : "hidden"}`],
      ["Thermal IR", modHidden ? "Hidden (test)" : `${irC.images} imgs / ${splitObj.gt_available ? irC.boxes + " boxes" : "hidden"}`],
      ["armored_vehicle", splitObj.gt_available ? (c.class_boxes.armored_vehicle || 0) : "—"],
      ["vehicle / person", splitObj.gt_available ? `${c.class_boxes.vehicle || 0} / ${c.class_boxes.person || 0}` : "—"],
      ["Loaded Preds", c.pred_count || 0],
    ];
    for (const [lbl, val] of cards) {
      const d = document.createElement("div");
      d.className = "stat-card";
      const l = document.createElement("div");
      l.className = "stat-label";
      l.textContent = lbl;
      const v = document.createElement("div");
      v.className = "stat-val";
      v.textContent = String(val);
      d.appendChild(l);
      d.appendChild(v);
      row.appendChild(d);
    }
    el("gtHiddenBanner").style.display = (!splitObj.gt_available && el("splitSelect").value === "test") ? "block" : "none";
  }

  function applyDetFilters() {
    const splitName = el("splitSelect").value;
    const splitObj = state.splits[splitName];
    if (!splitObj) return;
    renderStats(splitObj);

    const modVal = el("modalityFilter").value;
    const clsVal = el("classFilter").value;
    const q = el("searchBox").value.trim().toLowerCase();
    const thr = parseFloat(el("scoreSlider").value);
    el("scoreVal").textContent = thr.toFixed(2);

    filteredImages = splitObj.images.filter((im) => {
      if (modVal !== "all" && im.modality !== modVal) return false;
      if (q && !im.file_name.toLowerCase().includes(q) && !String(im.id).includes(q)) return false;
      if (clsVal === "neg") return im.boxes.length === 0;
      if (clsVal === "has_pred") return (im.preds || []).some((p) => p.score >= thr);
      if (clsVal === "0" || clsVal === "1" || clsVal === "2") {
        const target = parseInt(clsVal, 10);
        return im.boxes.some((b) => b.category_id === target);
      }
      return true;
    });
    page = 0;
    renderPage();
  }

  function renderPage() {
    const grid = el("thumbGrid");
    grid.innerHTML = "";
    const totalPages = Math.max(1, Math.ceil(filteredImages.length / PAGE_SIZE));
    page = Math.min(Math.max(0, page), totalPages - 1);
    const slice = filteredImages.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    el("pageInfo").textContent = `Page ${page + 1} / ${totalPages} (${filteredImages.length} items)`;

    if (slice.length > 0 && !slice.some((x) => x.id === selectedId)) {
      selectedId = slice[0].id;
    }

    for (const im of slice) {
      const card = document.createElement("div");
      card.className = "thumb-card" + (im.id === selectedId ? " selected" : "");
      card.onclick = () => { selectedId = im.id; renderPage(); drawDetInspector(im); };

      const imgEl = document.createElement("img");
      imgEl.loading = "lazy";
      imgEl.src = mediaUrl(im.media_src);
      imgEl.alt = im.file_name;
      card.appendChild(imgEl);

      const meta = document.createElement("div");
      meta.className = "thumb-meta";
      const nameSpan = document.createElement("span");
      nameSpan.textContent = im.file_name;
      const bSpan = document.createElement("span");
      const mClass = im.modality === "thermal" ? "thermal" : "rgb";
      bSpan.className = "badge " + mClass;
      bSpan.textContent = `${String(im.modality || "?").toUpperCase()} · ${im.boxes.length}`;
      meta.appendChild(nameSpan);
      meta.appendChild(bSpan);
      card.appendChild(meta);
      grid.appendChild(card);
    }

    const active = slice.find((x) => x.id === selectedId) || slice[0];
    if (active) drawDetInspector(active);
  }

  function appendCell(tr, text, color) {
    const td = document.createElement("td");
    td.textContent = text;
    if (color) td.style.color = color;
    tr.appendChild(td);
  }

  function drawDetInspector(im) {
    const canvas = el("detCanvas");
    const ctx = canvas.getContext("2d");
    const imgObj = new Image();
    const showGt = el("showGt").checked;
    const showPred = el("showPred").checked;
    const thr = parseFloat(el("scoreSlider").value);
    const filterCss = cssFilterFor(el("displayFilter").value);

    imgObj.onload = () => {
      canvas.width = imgObj.naturalWidth || im.width || 640;
      canvas.height = imgObj.naturalHeight || im.height || 480;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.filter = filterCss;
      ctx.drawImage(imgObj, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      const lw = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 240));
      ctx.font = `bold ${Math.max(12, lw * 6)}px ui-monospace, monospace`;

      if (showGt) {
        for (const box of im.boxes) {
          const [x, y, w, h] = box.bbox;
          const col = CLASS_COLORS[box.category_id] || "#38bdf8";
          ctx.strokeStyle = col;
          ctx.lineWidth = lw;
          ctx.setLineDash([]);
          ctx.strokeRect(x, y, w, h);
          const lbl = `${box.category_name}`;
          ctx.fillStyle = "rgba(11,17,23,0.78)";
          ctx.fillRect(x, Math.max(0, y - 20), ctx.measureText(lbl).width + 8, 18);
          ctx.fillStyle = col;
          ctx.fillText(lbl, x + 4, Math.max(14, y - 5));
        }
      }

      if (showPred) {
        for (const pred of (im.preds || [])) {
          if (pred.score < thr) continue;
          const [x, y, w, h] = pred.bbox;
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = lw;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(x, y, w, h);
          const lbl = `pred:${pred.category_name} ${pred.score.toFixed(2)}`;
          ctx.fillStyle = "rgba(6,40,30,0.85)";
          ctx.fillRect(x, Math.min(canvas.height - 18, y + h), ctx.measureText(lbl).width + 8, 18);
          ctx.fillStyle = "#6ee7b7";
          ctx.fillText(lbl, x + 4, Math.min(canvas.height - 4, y + h + 14));
        }
      }
      ctx.setLineDash([]);
    };
    imgObj.src = mediaUrl(im.media_src);

    el("detMetaSummary").textContent =
      `File: ${im.file_name} · Modality: ${String(im.modality).toUpperCase()} · ID: ${im.id} · Size: ${im.width}x${im.height}`;

    const tbody = el("detBoxTable").querySelector("tbody");
    tbody.innerHTML = "";
    for (const b of im.boxes) {
      const tr = document.createElement("tr");
      appendCell(tr, "GT");
      appendCell(tr, String(b.category_name));
      appendCell(tr, b.bbox.map((v) => v.toFixed(1)).join(", "));
      appendCell(tr, `${Math.round(b.area)} px²`);
      appendCell(tr, b.track_id != null ? "track #" + b.track_id : "—");
      tbody.appendChild(tr);
    }
    for (const p of (im.preds || [])) {
      if (p.score < thr) continue;
      const tr = document.createElement("tr");
      appendCell(tr, "Pred", "#10b981");
      appendCell(tr, String(p.category_name));
      appendCell(tr, p.bbox.map((v) => v.toFixed(1)).join(", "));
      appendCell(tr, `${Math.round(p.bbox[2] * p.bbox[3])} px²`);
      appendCell(tr, p.score.toFixed(3));
      tbody.appendChild(tr);
    }
  }

  function renderSeqList() {
    const list = el("seqList");
    list.innerHTML = "";
    const modF = el("streamModFilter").value;
    const matching = state.sequences.filter((s) => modF === "all" || s.modality === modF);
    for (const seq of matching) {
      const card = document.createElement("div");
      card.className = "seq-card" + (currentSeq && currentSeq.sequence === seq.sequence ? " selected" : "");
      card.onclick = () => selectSequence(seq);

      const title = document.createElement("div");
      title.style.fontWeight = "700";
      title.textContent = `${seq.sequence} (${seq.split})`;
      const sub = document.createElement("div");
      sub.style.fontSize = "12px";
      sub.style.color = "#8b9bb0";
      sub.style.marginTop = "4px";
      const clsStr = (seq.classes && seq.classes.length) ? seq.classes.join(", ") : "held-out test";
      sub.textContent = `${seq.width}x${seq.height} · ${seq.frame_count} frames · ${clsStr}`;

      const badges = document.createElement("div");
      badges.style.marginTop = "6px";
      const bm = document.createElement("span");
      bm.className = "badge " + (seq.modality === "thermal" ? "thermal" : "rgb");
      bm.textContent = String(seq.modality || "rgb").toUpperCase();
      badges.appendChild(bm);

      const bg = document.createElement("span");
      bg.className = "badge";
      bg.style.marginLeft = "6px";
      bg.textContent = seq.gt_available ? "GT Loaded" : "GT Hidden";
      badges.appendChild(bg);

      if (seq.pred_available) {
        const bp = document.createElement("span");
        bp.className = "badge pred";
        bp.style.marginLeft = "6px";
        bp.textContent = "Preds";
        badges.appendChild(bp);
      }
      card.appendChild(title);
      card.appendChild(sub);
      card.appendChild(badges);
      list.appendChild(card);
    }
  }

  async function selectSequence(seq) {
    currentSeq = seq;
    renderSeqList();
    const resp = await fetch("/api/sequence/" + encodeURIComponent(seq.sequence));
    currentSeqData = await resp.json();
    const vid = el("motVideo");
    vid.src = mediaUrl(seq.media_src);
    vid.style.filter = cssFilterFor(el("motDisplayFilter").value);
    vid.load();
    el("motHeaderInfo").textContent =
      `${seq.sequence} · modality=${String(seq.modality).toUpperCase()} · split=${seq.split} · ` +
      (currentSeqData.gt_available ? "GT tracks active" : "Held-out test stream (GT stripped)");
    drawMotFrame();
  }

  function drawMotFrame() {
    const vid = el("motVideo");
    const canvas = el("motCanvas");
    if (!currentSeq || !currentSeqData) return;
    const fps = currentSeqData.fps || 30;
    const maxF = currentSeqData.frame_count || 150;
    const frameIdx = Math.min(maxF, Math.max(1, Math.floor((vid.currentTime || 0) * fps + 1e-3) + 1));
    el("motFrameLabel").textContent = `Frame ${frameIdx} / ${maxF}`;

    const natW = vid.videoWidth || currentSeq.width || 640;
    const natH = vid.videoHeight || currentSeq.height || 480;
    if (canvas.width !== natW || canvas.height !== natH) {
      canvas.width = natW;
      canvas.height = natH;
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const gItems = (el("motShowGt").checked && currentSeqData.gt_available) ? (currentSeqData.gt_frames[String(frameIdx)] || []) : [];
    const pItems = (el("motShowPred").checked && currentSeqData.pred_available) ? (currentSeqData.pred_frames[String(frameIdx)] || []) : [];
    const lw = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 220));
    ctx.font = `bold ${Math.max(12, lw * 6)}px ui-monospace, monospace`;

    for (const item of gItems) {
      const [x, y, w, h] = item.bbox;
      const col = CLASS_COLORS[item.category_id] || "#38bdf8";
      ctx.strokeStyle = col;
      ctx.lineWidth = lw;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, w, h);
      const lbl = `#${item.track_id} ${CLASS_NAMES[item.category_id] || "obj"}`;
      ctx.fillStyle = "rgba(11,17,23,0.82)";
      ctx.fillRect(x, Math.max(0, y - 20), ctx.measureText(lbl).width + 8, 18);
      ctx.fillStyle = col;
      ctx.fillText(lbl, x + 4, Math.max(14, y - 5));
    }
    for (const item of pItems) {
      const [x, y, w, h] = item.bbox;
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = lw;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      const lbl = `P#${item.track_id} ${item.conf.toFixed(2)}`;
      ctx.fillStyle = "rgba(6,40,30,0.85)";
      ctx.fillRect(x, Math.min(canvas.height - 18, y + h), ctx.measureText(lbl).width + 8, 18);
      ctx.fillStyle = "#6ee7b7";
      ctx.fillText(lbl, x + 4, Math.min(canvas.height - 4, y + h + 14));
    }
    ctx.setLineDash([]);

    const tbody = el("motFrameTable").querySelector("tbody");
    tbody.innerHTML = "";
    for (const item of gItems) {
      const tr = document.createElement("tr");
      appendCell(tr, "GT");
      appendCell(tr, "#" + item.track_id);
      appendCell(tr, String(CLASS_NAMES[item.category_id] || item.category_id));
      appendCell(tr, item.bbox.map((v) => v.toFixed(1)).join(", "));
      appendCell(tr, item.conf.toFixed(2));
      tbody.appendChild(tr);
    }
    for (const item of pItems) {
      const tr = document.createElement("tr");
      appendCell(tr, "Pred", "#10b981");
      appendCell(tr, "#" + item.track_id);
      appendCell(tr, String(CLASS_NAMES[item.category_id] || item.category_id));
      appendCell(tr, item.bbox.map((v) => v.toFixed(1)).join(", "));
      appendCell(tr, item.conf.toFixed(2));
      tbody.appendChild(tr);
    }
  }

  function stepVideo(deltaFrames) {
    const vid = el("motVideo");
    if (!currentSeqData) return;
    vid.pause();
    const fps = currentSeqData.fps || 30;
    vid.currentTime = Math.max(0, vid.currentTime + deltaFrames / fps);
  }

  async function init() {
    const resp = await fetch("/api/state");
    state = await resp.json();
    populateSplits();
    applyDetFilters();
    renderSeqList();
    if (state.sequences.length > 0) selectSequence(state.sequences[0]);

    el("tabBtnDet").onclick = () => renderTabs("det");
    el("tabBtnMot").onclick = () => renderTabs("mot");
    el("splitSelect").onchange = applyDetFilters;
    el("modalityFilter").onchange = applyDetFilters;
    el("classFilter").onchange = applyDetFilters;
    el("displayFilter").onchange = renderPage;
    el("searchBox").oninput = applyDetFilters;
    el("scoreSlider").oninput = () => { el("scoreVal").textContent = parseFloat(el("scoreSlider").value).toFixed(2); applyDetFilters(); };
    el("showGt").onchange = renderPage;
    el("showPred").onchange = renderPage;
    el("prevPage").onclick = () => { if (page > 0) { page--; renderPage(); } };
    el("nextPage").onclick = () => { page++; renderPage(); };

    el("streamModFilter").onchange = renderSeqList;
    el("motDisplayFilter").onchange = () => { el("motVideo").style.filter = cssFilterFor(el("motDisplayFilter").value); };
    el("motShowGt").onchange = drawMotFrame;
    el("motShowPred").onchange = drawMotFrame;
    el("stepBackBtn").onclick = () => stepVideo(-1);
    el("stepFwdBtn").onclick = () => stepVideo(1);
    el("playPauseBtn").onclick = () => { const v = el("motVideo"); if (v.paused) v.play(); else v.pause(); };

    const vid = el("motVideo");
    vid.addEventListener("timeupdate", drawMotFrame);
    vid.addEventListener("seeked", drawMotFrame);
    vid.addEventListener("loadedmetadata", drawMotFrame);
    function tick() {
      if (!vid.paused && !vid.ended) drawMotFrame();
      window.requestAnimationFrame(tick);
    }
    window.requestAnimationFrame(tick);
  }

  init();
})();
</script>
</body>
</html>
"""


def _load_mot_txt(path: Path) -> dict[int, list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 6:
            continue
        try:
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            conf = float(parts[6]) if len(parts) > 6 and parts[6] else 1.0
            cls_id_0 = (
                max(0, int(float(parts[7])) - 1) if len(parts) > 7 and parts[7] else 0
            )
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (*bbox, conf)):
            continue
        frames[frame_id].append(
            {
                "track_id": track_id,
                "bbox": bbox,
                "conf": conf,
                "category_id": cls_id_0,
            }
        )
    return {k: frames[k] for k in sorted(frames)}


def _load_coco_preds(
    preds_path: Path | None, categories: dict[int, str]
) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if preds_path is None or not preds_path.is_file():
        return by_image
    raw = json.loads(preds_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return by_image
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            img_id = int(row["image_id"])
            cid = int(row["category_id"])
            bbox = [float(v) for v in row["bbox"]]
            score = float(row.get("score", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if len(bbox) != 4 or not all(math.isfinite(v) for v in (*bbox, score)):
            continue
        by_image[img_id].append(
            {
                "category_id": cid,
                "category_name": categories.get(cid, str(cid)),
                "bbox": bbox,
                "score": score,
            }
        )
    for img_id in by_image:
        by_image[img_id].sort(key=lambda item: item["score"], reverse=True)
    return by_image


def build_state(
    challenge_dir: Path,
    *,
    preds_path: Path | None = None,
    pred_dir: Path | None = None,
) -> dict[str, Any]:
    """Index Challenge 2 cross-modal RGB/Thermal splits, calibration, and MOT streams."""
    dataset_dir = challenge_dir / "dataset"
    cat_names: dict[int, str] = {c["id"]: c["name"] for c in DEFAULT_CATEGORIES}

    train_coco = dataset_dir / "train" / "_annotations.coco.json"
    if train_coco.is_file():
        cats = json.loads(train_coco.read_text(encoding="utf-8")).get("categories", [])
        if cats:
            cat_names = {int(c["id"]): str(c["name"]) for c in cats}

    preds_by_img = _load_coco_preds(preds_path, cat_names)
    train_modalities_by_name: dict[str, str] = {}
    splits: dict[str, Any] = {}

    for split in ("train", "valid", "test"):
        ann_path = dataset_dir / split / "_annotations.coco.json"
        if not ann_path.is_file():
            continue
        doc = json.loads(ann_path.read_text(encoding="utf-8"))
        for c in doc.get("categories", []):
            cat_names[int(c["id"])] = str(c["name"])
        anns = doc.get("annotations", [])
        anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
        class_boxes: dict[str, int] = {name: 0 for name in cat_names.values()}

        for ann in anns:
            cid = int(ann["category_id"])
            cname = cat_names.get(cid, str(cid))
            bbox = [float(v) for v in ann["bbox"]]
            area = float(ann.get("area", bbox[2] * bbox[3]))
            class_boxes[cname] = class_boxes.get(cname, 0) + 1
            anns_by_img[int(ann["image_id"])].append(
                {
                    "category_id": cid,
                    "category_name": cname,
                    "bbox": bbox,
                    "area": area,
                    "track_id": ann.get("track_id"),
                }
            )

        gt_available = True if split != "test" else bool(anns)
        modality_counts: dict[str, dict[str, int]] = {
            "rgb": {"images": 0, "boxes": 0},
            "thermal": {"images": 0, "boxes": 0},
        }
        images_out: list[dict[str, Any]] = []
        pred_count = 0
        for img in doc.get("images", []):
            img_id = int(img["id"])
            fname = str(img["file_name"])
            raw_mod = img.get("modality")
            mod = str(raw_mod).lower() if raw_mod else "unknown"
            if split == "train":
                train_modalities_by_name[fname] = mod
            boxes = anns_by_img.get(img_id, [])
            bucket = modality_counts.setdefault(mod, {"images": 0, "boxes": 0})
            bucket["images"] += 1
            bucket["boxes"] += len(boxes)
            preds = preds_by_img.get(img_id, [])
            pred_count += len(preds)
            images_out.append(
                {
                    "id": img_id,
                    "file_name": fname,
                    "media_src": f"dataset/{split}/{fname}",
                    "modality": mod,
                    "width": int(img.get("width", 0)),
                    "height": int(img.get("height", 0)),
                    "boxes": boxes,
                    "preds": preds,
                }
            )

        splits[split] = {
            "gt_available": gt_available,
            "counts": {
                "image_count": len(images_out),
                "box_count": len(anns),
                "class_boxes": class_boxes,
                "modality_counts": modality_counts,
                "pred_count": pred_count,
            },
            "images": images_out,
        }

    cal_dir = challenge_dir / "calibration"
    if cal_dir.is_dir():
        cal_imgs = sorted(
            p
            for p in cal_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        cal_mod_counts: dict[str, dict[str, int]] = {
            "rgb": {"images": 0, "boxes": 0},
            "thermal": {"images": 0, "boxes": 0},
        }
        cal_items: list[dict[str, Any]] = []
        for idx, p in enumerate(cal_imgs):
            mod = train_modalities_by_name.get(p.name, "unknown")
            if mod in cal_mod_counts:
                cal_mod_counts[mod]["images"] += 1
            cal_items.append(
                {
                    "id": 900000 + idx,
                    "file_name": p.name,
                    "media_src": f"calibration/{p.name}",
                    "modality": mod,
                    "width": 0,
                    "height": 0,
                    "boxes": [],
                    "preds": [],
                }
            )
        splits["calibration"] = {
            "gt_available": False,
            "counts": {
                "image_count": len(cal_items),
                "box_count": 0,
                "class_boxes": {},
                "modality_counts": cal_mod_counts,
                "pred_count": 0,
            },
            "images": cal_items,
        }

    sequences: list[dict[str, Any]] = []
    manifest_path = challenge_dir / "streams_eval" / "streams_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("streams", []):
            seq_name = str(entry["sequence"])
            gt_file = challenge_dir / "streams_eval" / seq_name / "gt" / "gt.txt"
            pred_file = pred_dir / f"{seq_name}.txt" if pred_dir is not None else None
            sequences.append(
                {
                    "sequence": seq_name,
                    "split": str(entry.get("split", "unknown")),
                    "modality": str(entry.get("modality", "rgb")).lower(),
                    "media_src": f"streams_eval/{seq_name}/video.mp4",
                    "width": int(entry.get("width", 0)),
                    "height": int(entry.get("height", 0)),
                    "fps": float(entry.get("fps", 30)),
                    "frame_count": int(entry.get("frame_count", 0)),
                    "classes": list(entry.get("classes", [])),
                    "gt_boxes": int(entry.get("gt_boxes", 0)),
                    "gt_available": gt_file.is_file(),
                    "pred_available": bool(pred_file and pred_file.is_file()),
                }
            )

    return {
        "challenge": "challenge-2-rgb-thermal-isr",
        "title": "Challenge 2: Day RGB + Thermal IR ISR Detection & Multi-Stream Tracking",
        "categories": [{"id": k, "name": v} for k, v in sorted(cat_names.items())],
        "splits": splits,
        "sequences": sequences,
    }


def _load_sequence_payload(
    challenge_dir: Path,
    seq_name: str,
    *,
    pred_dir: Path | None,
) -> dict[str, Any] | None:
    manifest_path = challenge_dir / "streams_eval" / "streams_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = next(
        (
            entry
            for entry in manifest.get("streams", [])
            if entry.get("sequence") == seq_name
        ),
        None,
    )
    if meta is None:
        return None
    gt_file = challenge_dir / "streams_eval" / seq_name / "gt" / "gt.txt"
    pred_file = pred_dir / f"{seq_name}.txt" if pred_dir is not None else None
    gt_raw = _load_mot_txt(gt_file) if gt_file.is_file() else {}
    pred_raw = (
        _load_mot_txt(pred_file)
        if (pred_file is not None and pred_file.is_file())
        else {}
    )
    return {
        "sequence": seq_name,
        "modality": str(meta.get("modality", "rgb")).lower(),
        "fps": float(meta.get("fps", 30)),
        "frame_count": int(meta.get("frame_count", 0)),
        "gt_available": gt_file.is_file(),
        "pred_available": bool(pred_file and pred_file.is_file()),
        "gt_frames": {str(k): v for k, v in gt_raw.items()},
        "pred_frames": {str(k): v for k, v in pred_raw.items()},
    }


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _EXTRA_MIME_TYPES:
        return _EXTRA_MIME_TYPES[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _resolve_safe_media_path(root_dir: Path, raw_rel: str) -> Path | None:
    rel = urllib.parse.unquote(raw_rel).lstrip("/")
    if not rel or "\x00" in rel:
        return None
    root_resolved = root_dir.resolve()
    candidate = (root_resolved / rel).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _serve_file_with_range(
    handler: BaseHTTPRequestHandler,
    path: Path,
    *,
    send_body: bool = True,
) -> None:
    size = path.stat().st_size
    mime = _guess_mime(path)
    range_hdr = handler.headers.get("Range")
    if range_hdr and range_hdr.startswith("bytes="):
        spec = range_hdr[len("bytes=") :].split(",", 1)[0].strip()
        start_str, _, end_str = spec.partition("-")
        try:
            if start_str == "":
                suffix_len = int(end_str)
                start = max(0, size - suffix_len)
                end = size - 1
            else:
                start = int(start_str)
                end = int(end_str) if end_str else size - 1
        except ValueError:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        if size == 0 or start < 0 or end < start or start >= size:
            handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.end_headers()
            return
        end = min(end, size - 1)
        length = end - start + 1
        handler.send_response(HTTPStatus.PARTIAL_CONTENT)
        handler.send_header("Content-Type", mime)
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.send_header("Content-Length", str(length))
        handler.end_headers()
        if not send_body:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
        return

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", mime)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(size))
    handler.end_headers()
    if not send_body:
        return
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            handler.wfile.write(chunk)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(
    challenge_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    preds_path: Path | None = None,
    pred_dir: Path | None = None,
) -> ThreadingHTTPServer:
    """Create the Challenge 2 HTTP viewer server bound to ``host:port``."""
    root_dir = challenge_dir.resolve()
    state_bytes = json.dumps(
        build_state(root_dir, preds_path=preds_path, pred_dir=pred_dir),
        allow_nan=False,
    ).encode("utf-8")
    html_bytes = _HTML_PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_request(self, *, send_body: bool) -> None:
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path
            try:
                if route in ("/", "/index.html"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.end_headers()
                    if send_body:
                        self.wfile.write(html_bytes)
                    return

                if route == "/api/state":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(state_bytes)))
                    self.end_headers()
                    if send_body:
                        self.wfile.write(state_bytes)
                    return

                if route.startswith("/api/sequence/"):
                    seq_name = urllib.parse.unquote(route[len("/api/sequence/") :])
                    payload = _load_sequence_payload(
                        root_dir, seq_name, pred_dir=pred_dir
                    )
                    if payload is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    body = json.dumps(payload, allow_nan=False).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if send_body:
                        self.wfile.write(body)
                    return

                if route.startswith("/media/"):
                    target = _resolve_safe_media_path(root_dir, route[len("/media/") :])
                    if target is None:
                        self.send_error(HTTPStatus.FORBIDDEN)
                        return
                    _serve_file_with_range(self, target, send_body=send_body)
                    return

                self.send_error(HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self) -> None:
            self._handle_request(send_body=True)

        def do_HEAD(self) -> None:
            self._handle_request(send_body=False)

    return _QuietThreadingHTTPServer((host, port), Handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenge-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Challenge 2 directory containing dataset/, calibration/, streams_eval/",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--preds",
        type=Path,
        default=None,
        help="Optional COCO predictions JSON to overlay alongside ground truth",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Optional MOTChallenge prediction directory (seq_XXXX.txt)",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = create_server(
        args.challenge_dir,
        host=args.host,
        port=args.port,
        preds_path=args.preds,
        pred_dir=args.pred_dir,
    )
    actual_url = f"http://{args.host}:{server.server_port}/"
    print(f"Challenge 2 viewer listening on {actual_url}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(actual_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
