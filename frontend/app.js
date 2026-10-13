/**
 * frontend/app.js
 * Client-side application logic for Dark Pattern Detection Research Web App
 */

const CATEGORY_META = {
  "Social Proof": {
    color: "cyan",
    bgClass: "bg-cyan-50",
    textClass: "text-cyan-800",
    borderClass: "border-cyan-200",
    badgeClass: "badge-social",
    icon: "ph-users",
    description: "Leverages crowd activity, peer endorsements, or high visitor numbers to trigger conformity bias.",
  },
  "Scarcity": {
    color: "amber",
    bgClass: "bg-amber-50",
    textClass: "text-amber-800",
    borderClass: "border-amber-200",
    badgeClass: "badge-scarcity",
    icon: "ph-hourglass-simple",
    description: "Signals limited product stock, scarce availability, or high purchase competition to induce FOMO.",
  },
  "Urgency": {
    color: "red",
    bgClass: "bg-red-50",
    textClass: "text-red-800",
    borderClass: "border-red-200",
    badgeClass: "badge-urgency",
    icon: "ph-alarm",
    description: "Imposes strict temporal constraints, expiring offers, or countdown intervals to force rushed purchases.",
  },
  "Misdirection": {
    color: "purple",
    bgClass: "bg-purple-50",
    textClass: "text-purple-800",
    borderClass: "border-purple-200",
    badgeClass: "badge-misdirection",
    icon: "ph-arrows-split",
    description: "Employs asymmetric phrasing, misleading price anchors, or visual misdirection to steer decisions.",
  },
};

let benchmarkChart = null;
let perClassChart = null;
let globalExperiments = {};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initClassifier();
  initPresets();
  loadHealth();
  loadBenchmarks();
  loadAuditStats();
  loadDatasetSamples();
  loadDataDictionary();
  initBatchPrediction();
});

/* =========================================================================
 * 1. TAB NAVIGATION
 * ========================================================================= */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.getAttribute("data-tab");
      
      // Update tab buttons
      tabs.forEach(t => {
        if (t.getAttribute("data-tab") === targetTab) {
          t.classList.add("active", "text-brand-700", "bg-brand-50");
          t.classList.remove("text-slate-600");
        } else {
          t.classList.remove("active", "text-brand-700", "bg-brand-50");
          t.classList.add("text-slate-600");
        }
      });

      // Show target section
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.add("hidden");
      });
      const activePane = document.getElementById(`tab-${targetTab}`);
      if (activePane) {
        activePane.classList.remove("hidden");
      }

      // Resize charts if switching to benchmark tab
      if (targetTab === "benchmark") {
        setTimeout(() => {
          if (benchmarkChart) benchmarkChart.resize();
          if (perClassChart) perClassChart.resize();
        }, 100);
      }
    });
  });
}

/* =========================================================================
 * 2. LIVE CLASSIFIER
 * ========================================================================= */
function initClassifier() {
  const input = document.getElementById("inputText");
  const charCount = document.getElementById("charCount");
  const btnPredict = document.getElementById("btnPredict");
  const btnClear = document.getElementById("btnClear");
  const modelSelect = document.getElementById("modelSelect");

  input.addEventListener("input", () => {
    charCount.textContent = `${input.value.length} characters`;
  });

  btnClear.addEventListener("click", () => {
    input.value = "";
    charCount.textContent = "0 characters";
    input.focus();
  });

  btnPredict.addEventListener("click", handlePrediction);

  modelSelect.addEventListener("change", () => {
    const isAll = modelSelect.value === "all";
    document.getElementById("multiModelCard").classList.toggle("hidden", !isAll);
  });
}

function initPresets() {
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      const input = document.getElementById("inputText");
      input.value = text;
      document.getElementById("charCount").textContent = `${text.length} characters`;
      handlePrediction();
    });
  });
}

async function handlePrediction() {
  const text = document.getElementById("inputText").value.trim();
  if (!text) {
    alert("Please enter interface text to analyze.");
    return;
  }

  const modelChoice = document.getElementById("modelSelect").value;
  const btnPredict = document.getElementById("btnPredict");
  btnPredict.disabled = true;
  btnPredict.innerHTML = `<i class="ph-bold ph-spinner animate-spin"></i><span>Analyzing...</span>`;

  try {
    if (modelChoice === "all") {
      // Call multi-model endpoint
      const response = await fetch("/predict/all-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Multi-model inference failed.");
      }
      const data = await response.json();
      renderMultiModelResults(data);
    } else {
      // Call single model endpoint
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_choice: modelChoice }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Prediction failed.");
      }
      const data = await response.json();
      renderSingleResult(data);
      document.getElementById("multiModelCard").classList.add("hidden");
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btnPredict.disabled = false;
    btnPredict.innerHTML = `<i class="ph-bold ph-lightning"></i><span>Analyze Text</span>`;
  }
}

function renderSingleResult(data) {
  const category = data.predicted_category;
  const meta = CATEGORY_META[category] || {
    color: "indigo",
    badgeClass: "badge-social",
    icon: "ph-shield-check",
    description: "Analyzed category cue.",
  };

  // Title and category
  document.getElementById("categoryTitle").textContent = category;
  document.getElementById("categoryDescription").textContent = meta.description;
  document.getElementById("latencyBadge").textContent = `${data.latency_ms} ms`;

  // Badge & Model
  const confPercent = (data.confidence * 100).toFixed(1);
  const badge = document.getElementById("confidenceBadge");
  badge.textContent = `${confPercent}% Confidence`;
  badge.className = `text-xs font-semibold px-2 py-0.5 rounded-full ${meta.badgeClass}`;

  const modelNames = {
    model_a: "Model A (TF-IDF)",
    model_b: "Model B (DistilBERT)",
    model_c: "Model C (Linguistic)",
    model_d: "Model D (Late Fusion)",
  };
  document.getElementById("modelUsedBadge").textContent = modelNames[data.model_used] || data.model_used;

  // Icon
  const iconContainer = document.getElementById("categoryIcon");
  iconContainer.className = `w-12 h-12 rounded-xl flex items-center justify-center text-2xl font-bold ${meta.bgClass} ${meta.textClass} border ${meta.borderClass}`;
  iconContainer.innerHTML = `<i class="ph-bold ${meta.icon}"></i>`;

  // Probabilities
  const probs = data.class_probabilities || {};
  updateProbBar("Social", probs["Social Proof"] || 0);
  updateProbBar("Scarcity", probs["Scarcity"] || 0);
  updateProbBar("Urgency", probs["Urgency"] || 0);
  updateProbBar("Misdirection", probs["Misdirection"] || 0);

  // Linguistic Cues
  renderCues(data.observed_cues || []);
}

function updateProbBar(key, prob) {
  const percent = Math.min(100, Math.max(0, prob * 100)).toFixed(1);
  document.getElementById(`prob${key}`).textContent = `${percent}%`;
  document.getElementById(`bar${key}`).style.width = `${percent}%`;
}

function renderCues(cues) {
  const container = document.getElementById("cuesContainer");
  if (!cues || cues.length === 0) {
    container.innerHTML = `<span class="text-xs text-slate-400 italic">No salient linguistic indicators detected.</span>`;
    return;
  }

  container.innerHTML = cues.map(cue => `
    <div class="p-2 rounded-lg bg-slate-50 border border-slate-200 text-xs">
      <div class="flex items-center justify-between font-semibold text-slate-800">
        <span>${cue.type}</span>
        <span class="text-[10px] px-1.5 py-0.5 rounded bg-brand-50 text-brand-700 font-mono">${cue.associated_category}</span>
      </div>
      <p class="text-[11px] text-slate-500 mt-0.5">${cue.detail}</p>
    </div>
  `).join("");
}

function renderMultiModelResults(data) {
  // Show multi model panel
  const card = document.getElementById("multiModelCard");
  const grid = document.getElementById("multiModelGrid");
  card.classList.remove("hidden");

  // Render cards for each model
  const models = data.models || {};
  grid.innerHTML = Object.keys(models).map(key => {
    const m = models[key];
    if (m.error) {
      return `
        <div class="p-3.5 rounded-xl border border-red-200 bg-red-50 text-xs">
          <span class="font-bold text-red-800 block">${m.name}</span>
          <span class="text-red-600 mt-1 block">${m.error}</span>
        </div>
      `;
    }

    const cat = m.predicted_category;
    const meta = CATEGORY_META[cat] || { badgeClass: "badge-social" };
    const conf = (m.confidence * 100).toFixed(1);

    return `
      <div class="p-4 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white transition space-y-2">
        <div class="flex justify-between items-start">
          <span class="text-xs font-bold text-slate-900">${m.name}</span>
          <span class="text-[10px] text-slate-400 font-mono">${m.latency_ms}ms</span>
        </div>
        <div class="flex items-center justify-between pt-1">
          <span class="text-xs font-bold px-2 py-0.5 rounded-full ${meta.badgeClass}">${cat}</span>
          <span class="text-xs font-bold text-slate-700">${conf}%</span>
        </div>
      </div>
    `;
  }).join("");

  // Default display single result to Model D
  if (models["model_d"] && !models["model_d"].error) {
    const d = models["model_d"];
    renderSingleResult({
      predicted_category: d.predicted_category,
      confidence: d.confidence,
      class_probabilities: d.class_probabilities,
      model_used: "model_d",
      latency_ms: d.latency_ms,
      observed_cues: data.observed_cues,
    });
  }
}

/* =========================================================================
 * 3. BENCHMARK & COMPARISONS
 * ========================================================================= */
async function loadBenchmarks() {
  try {
    const res = await fetch("/experiments");
    if (!res.ok) return;
    const data = await res.json();
    globalExperiments = data;

    renderBenchmarkTable(data);
    renderBenchmarkCharts(data);
    renderConfusionMatrix("Model D: Late Fusion (Proposed)");
    initConfusionMatrixButtons();
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

function renderBenchmarkTable(experiments) {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!tbody) return;

  const rows = Object.keys(experiments).map(modelName => {
    const m = experiments[modelName];
    const isProposed = modelName.includes("Proposed") || modelName.includes("Model D");
    const rowClass = isProposed ? "bg-brand-50/50 font-semibold" : "";
    const badge = isProposed ? `<span class="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-brand-600 text-white uppercase font-bold">Proposed</span>` : "";

    return `
      <tr class="${rowClass}">
        <td class="py-3 px-4 font-medium text-slate-900">${modelName} ${badge}</td>
        <td class="py-3 px-4 text-right font-mono">${(m.accuracy * 100).toFixed(2)}%</td>
        <td class="py-3 px-4 text-right font-mono font-bold text-brand-700">${m.macro_f1.toFixed(4)}</td>
        <td class="py-3 px-4 text-right font-mono">${m.weighted_f1.toFixed(4)}</td>
        <td class="py-3 px-4 text-right font-mono">${m.macro_precision.toFixed(4)}</td>
        <td class="py-3 px-4 text-right font-mono">${m.macro_recall.toFixed(4)}</td>
        <td class="py-3 px-4 text-right text-xs text-slate-500 font-mono">${m.training_time_sec ? m.training_time_sec + "s" : "--"}</td>
      </tr>
    `;
  }).join("");

  tbody.innerHTML = rows;
}

function renderBenchmarkCharts(experiments) {
  const models = Object.keys(experiments);
  const macroF1s = models.map(m => experiments[m].macro_f1);
  const accuracies = models.map(m => experiments[m].accuracy);

  // Bar chart
  const ctx = document.getElementById("benchmarkBarChart");
  if (ctx) {
    if (benchmarkChart) benchmarkChart.destroy();
    benchmarkChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: models.map(m => m.replace("Model ", "M-").split(":")[0]),
        datasets: [
          {
            label: "Macro F1",
            data: macroF1s,
            backgroundColor: "#6366f1",
            borderRadius: 6,
          },
          {
            label: "Accuracy",
            data: accuracies,
            backgroundColor: "#c7d2fe",
            borderRadius: 6,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
        },
        scales: {
          y: { min: 0.7, max: 1.0, ticks: { font: { size: 10 } } },
          x: { ticks: { font: { size: 10 } } }
        }
      }
    });
  }

  // Per-class radar / bar chart for Model D
  const modelD = experiments["Model D: Late Fusion (Proposed)"] || experiments["Model D: Semantic-Linguistic Fusion"];
  if (modelD && modelD.per_class) {
    const classCtx = document.getElementById("perClassChart");
    if (classCtx) {
      if (perClassChart) perClassChart.destroy();
      const labels = Object.keys(modelD.per_class);
      const f1Scores = labels.map(l => modelD.per_class[l].f1_score);

      perClassChart = new Chart(classCtx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Model D F1 Score",
            data: f1Scores,
            backgroundColor: ["#06b6d4", "#f59e0b", "#ef4444", "#8b5cf6"],
            borderRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
          },
          scales: {
            y: { min: 0.7, max: 1.0, ticks: { font: { size: 10 } } },
            x: { ticks: { font: { size: 10 } } }
          }
        }
      });
    }
  }
}

function initConfusionMatrixButtons() {
  document.querySelectorAll(".cm-switch-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".cm-switch-btn").forEach(b => {
        b.className = "cm-switch-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200";
      });
      btn.className = "cm-switch-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-brand-600 text-white";
      const modelKey = btn.getAttribute("data-model");
      renderConfusionMatrix(modelKey);
    });
  });
}

function renderConfusionMatrix(modelKey) {
  const container = document.getElementById("cmContainer");
  if (!container || !globalExperiments[modelKey]) return;

  const data = globalExperiments[modelKey];
  const cm = data.confusion_matrix;
  const classes = data.classes || ["Social Proof", "Scarcity", "Urgency", "Misdirection"];

  // Find max value in cm for shading
  let maxVal = 1;
  cm.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

  let html = `
    <div class="text-xs font-semibold text-slate-700 mb-2">${modelKey} — Confusion Matrix</div>
    <div class="grid grid-cols-5 gap-1.5 text-center text-xs">
      <div class="font-bold text-slate-400">True \\ Pred</div>
      ${classes.map(c => `<div class="font-bold text-slate-600 truncate" title="${c}">${c.split(" ")[0]}</div>`).join("")}
  `;

  classes.forEach((trueClass, i) => {
    html += `<div class="font-bold text-slate-600 flex items-center justify-end pr-2 truncate" title="${trueClass}">${trueClass.split(" ")[0]}</div>`;
    classes.forEach((_, j) => {
      const val = cm[i][j];
      const intensity = Math.min(1, val / maxVal);
      const isDiag = i === j;
      const bg = isDiag
        ? `rgba(99, 102, 241, ${Math.max(0.15, intensity)})`
        : val > 0 ? `rgba(239, 68, 68, ${Math.max(0.15, intensity * 0.5)})` : '#f8fafc';
      const textCol = intensity > 0.5 && isDiag ? '#ffffff' : '#1e293b';

      html += `<div class="cm-cell border border-slate-200" style="background-color: ${bg}; color: ${textCol}">${val}</div>`;
    });
  });

  html += `</div>`;
  container.innerHTML = html;
}

/* =========================================================================
 * 4. DATASET & AUDIT
 * ========================================================================= */
async function loadAuditStats() {
  try {
    const res = await fetch("/audit");
    if (!res.ok) return;
    const data = await res.json();

    const raw = data.initial_dataset_verification || {};
    const clean = data.data_cleaning_and_exclusion || {};
    const target = data.primary_4class_dataset || {};

    if (raw.total_raw_rows) document.getElementById("statRawRows").textContent = raw.total_raw_rows.toLocaleString();
    if (clean.missing_text_rows_excluded !== undefined) document.getElementById("statMissingText").textContent = clean.missing_text_rows_excluded.toLocaleString();
    if (clean.rows_with_usable_text) document.getElementById("statUsableText").textContent = clean.rows_with_usable_text.toLocaleString();
    if (clean.excluded_categories_count !== undefined) document.getElementById("statExcludedClasses").textContent = clean.excluded_categories_count.toLocaleString();
    if (target.eligible_sample_count) document.getElementById("statTargetRows").textContent = target.eligible_sample_count.toLocaleString();
  } catch (err) {
    console.error("Failed to load audit stats:", err);
  }
}

async function loadDatasetSamples() {
  const catFilter = document.getElementById("filterCategory").value;
  const searchFilter = document.getElementById("filterSearch").value;

  try {
    let url = `/dataset-sample?limit=15&offset=0`;
    if (catFilter) url += `&category=${encodeURIComponent(catFilter)}`;
    if (searchFilter) url += `&search=${encodeURIComponent(searchFilter)}`;

    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();

    const tbody = document.getElementById("datasetSampleTable");
    if (!tbody) return;

    tbody.innerHTML = (data.samples || []).map(item => {
      const meta = CATEGORY_META[item.category] || { badgeClass: "badge-social" };
      const shortUrl = item.url ? item.url.replace(/^https?:\/\//, "").split("/")[0] : "N/A";

      return `
        <tr>
          <td class="py-2.5 px-3 max-w-xs truncate font-medium text-slate-800" title="${item.text}">${item.text}</td>
          <td class="py-2.5 px-3 whitespace-nowrap"><span class="px-2 py-0.5 rounded-full font-semibold ${meta.badgeClass}">${item.category}</span></td>
          <td class="py-2.5 px-3 whitespace-nowrap text-slate-600">${item.pattern_type}</td>
          <td class="py-2.5 px-3 whitespace-nowrap text-slate-500">${item.where_in_website}</td>
          <td class="py-2.5 px-3 whitespace-nowrap"><a href="${item.url}" target="_blank" class="text-brand-600 hover:underline" title="${item.url}">${shortUrl}</a></td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load dataset samples:", err);
  }
}

document.getElementById("filterCategory")?.addEventListener("change", loadDatasetSamples);
document.getElementById("filterSearch")?.addEventListener("input", debounce(loadDatasetSamples, 300));

/* =========================================================================
 * 5. DATA DICTIONARY
 * ========================================================================= */
async function loadDataDictionary() {
  try {
    const res = await fetch("/data-dictionary");
    if (!res.ok) return;
    const data = await res.json();
    const cols = data.columns || {};

    const grid = document.getElementById("dataDictionaryGrid");
    if (!grid) return;

    grid.innerHTML = Object.keys(cols).map(colName => {
      const col = cols[colName];
      return `
        <div class="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
          <div class="flex justify-between items-start">
            <span class="text-sm font-bold text-slate-900 font-mono">${colName}</span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-brand-50 border border-brand-200 text-brand-700 font-semibold uppercase">${col.type}</span>
          </div>
          <p class="text-xs text-slate-700">${col.description}</p>
          <div class="text-[11px] text-slate-500 border-t border-slate-200 pt-1.5 mt-1 space-y-0.5">
            <div><strong>Role:</strong> ${col.role}</div>
            <div><strong>Handling:</strong> ${col.handling}</div>
          </div>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load data dictionary:", err);
  }
}

/* =========================================================================
 * 6. BATCH PREDICTION
 * ========================================================================= */
let lastBatchPredictions = [];

function initBatchPrediction() {
  const btn = document.getElementById("btnRunBatch");
  const downloadBtn = document.getElementById("btnDownloadBatchCsv");

  btn.addEventListener("click", async () => {
    const raw = document.getElementById("batchInput").value;
    const lines = raw.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) {
      alert("Please enter text lines for batch classification.");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Processing Batch...";

    try {
      const res = await fetch("/batch-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines, model_choice: "model_d" }),
      });
      if (!res.ok) throw new Error("Batch prediction failed.");
      const data = await res.json();

      lastBatchPredictions = data.predictions || [];
      document.getElementById("batchCount").textContent = lastBatchPredictions.length;
      document.getElementById("batchResultsContainer").classList.remove("hidden");

      const tbody = document.getElementById("batchTableBody");
      tbody.innerHTML = lastBatchPredictions.map(item => {
        const meta = CATEGORY_META[item.predicted_category] || { badgeClass: "badge-social" };
        const conf = (item.confidence * 100).toFixed(1);
        return `
          <tr>
            <td class="py-2.5 px-3 max-w-sm truncate font-medium text-slate-800" title="${item.text}">${item.text}</td>
            <td class="py-2.5 px-3 whitespace-nowrap"><span class="px-2 py-0.5 rounded-full font-semibold ${meta.badgeClass}">${item.predicted_category}</span></td>
            <td class="py-2.5 px-3 whitespace-nowrap text-right font-mono font-bold">${conf}%</td>
          </tr>
        `;
      }).join("");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Process Batch";
    }
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastBatchPredictions.length) return;
    let csv = "Text,Predicted Category,Confidence\n";
    lastBatchPredictions.forEach(row => {
      const escaped = `"${row.text.replace(/"/g, '""')}"`;
      csv += `${escaped},${row.predicted_category},${row.confidence}\n`;
    });
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dark_pattern_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  });
}

/* =========================================================================
 * 7. HEALTH CHECK
 * ========================================================================= */
async function loadHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    document.getElementById("statusText").textContent = `Online (${data.device.toUpperCase()})`;
  } catch {
    document.getElementById("statusText").textContent = "Service Offline";
    document.getElementById("statusIndicator").className = "flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-red-50 border border-red-200 text-red-700 text-xs font-semibold";
  }
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
