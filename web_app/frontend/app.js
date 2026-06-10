/**
 * Perovskite Defect Detection — Frontend Logic
 * ==============================================
 * Handles tab switching, drag-drop upload, API calls,
 * result rendering, and benchmark chart population.
 */

// ═══════════════════════════════════════════════════════════════════
// THEME TOGGLE
// ═══════════════════════════════════════════════════════════════════
const themeToggle = document.getElementById('themeToggle');
let isDarkTheme = true;

if(themeToggle) {
  themeToggle.addEventListener('click', () => {
    isDarkTheme = !isDarkTheme;
    document.documentElement.setAttribute('data-theme', isDarkTheme ? 'dark' : 'light');
    themeToggle.textContent = isDarkTheme ? '☀️' : '🌙';
    themeToggle.style.color = isDarkTheme ? 'var(--text-primary)' : '#111827';
  });
}

// ═══════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════
document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const target = document.getElementById('tab-' + btn.dataset.tab);
    if (target) target.classList.add('active');
    // Load benchmark data when switching to benchmark tab (idempotent)
    if (btn.dataset.tab === 'benchmark') loadBenchmark();
    if (btn.dataset.tab === 'annotation' && annoImage && annoImage.src) {
      syncCanvas();
    }
  });
});

// Auto-load benchmark data at startup so it's ready when the user switches to it
window.addEventListener('DOMContentLoaded', () => loadBenchmark());


// ═══════════════════════════════════════════════════════════════════
// UPLOAD & INFERENCE
// ═══════════════════════════════════════════════════════════════════
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const loader      = document.getElementById('loader');
const alertError  = document.getElementById('alertError');
const alertNoDetect = document.getElementById('alertNoDetect');
const resultsArea = document.getElementById('resultsArea');
const btnReset    = document.getElementById('btnReset');

// Click on dropZone is handled natively by the <label for="fileInput"> in HTML

['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
});
['dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
});

// File selected via input
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// File dropped
dropZone.addEventListener('drop', e => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

// Reset button
btnReset.addEventListener('click', resetUI);

// Confidence slider live update
const confSlider = document.getElementById('confSlider');
const confVal    = document.getElementById('confVal');
if (confSlider && confVal) {
  confSlider.addEventListener('input', () => {
    confVal.textContent = confSlider.value + '%';
  });
}

function resetUI() {
  resultsArea.classList.remove('visible');
  alertError.classList.remove('visible');
  alertNoDetect.classList.remove('visible');
  fileInput.value = '';
}

/**
 * Validate the image type and upload.
 */
function handleFile(file) {
  resetUI();

  if (!file.type.startsWith('image/')) {
    showError('Please upload a valid image file (JPEG, PNG, or TIFF).');
    return;
  }

  // Valid — send to backend
  uploadAndDetect(file);
}

async function uploadAndDetect(file) {
  loader.classList.add('active');

  const conf = confSlider ? (parseInt(confSlider.value) / 100) : 0.01;

  const form = new FormData();
  form.append('file', file);
  form.append('conf', conf.toFixed(2));

  try {
    const res = await fetch('/api/detect', { method: 'POST', body: form });
    const data = await res.json();

    loader.classList.remove('active');

    if (!res.ok) {
      showError(data.detail || 'Server error. Please try again.');
      return;
    }

    if (!data.success) {
      if (data.is_background && data.background_name) {
        showBackgroundResult(data);
      } else {
        alertNoDetect.classList.add('visible');
      }
      return;
    }

    renderResults(data);

  } catch (err) {
    loader.classList.remove('active');
    showError('Could not connect to the server. Is the backend running?');
  }
}

function showError(msg) {
  document.getElementById('alertErrorMsg').textContent = msg;
  alertError.classList.add('visible');
}

function showBackgroundResult(data) {
  // Build a rich background classification card inside resultsArea
  const badge = document.getElementById('resultBadge');
  badge.textContent = 'Clean Background — No Defects';
  badge.style.background = 'linear-gradient(135deg, #10B981, #059669)';

  // Show original image
  document.getElementById('imgOriginal').src  = 'data:image/png;base64,' + data.original_b64;
  document.getElementById('imgAnnotated').src = 'data:image/png;base64,' + data.original_b64;

  // Build background class info card in detection list
  const list = document.getElementById('detectionList');
  list.innerHTML = '';

  const bgClass   = data.background_name;    // e.g. "3D_background"
  const bgScores  = data.bg_scores || {};
  const score3    = bgScores['3D_background']     || 0;
  const score4    = bgScores['3D-2D_background']  || 0;

  // Colour-code: class 3 = teal, class 4 = indigo
  const bgColor  = data.background_class === 3 ? '#10B981' : '#818CF8';
  const bgIcon   = data.background_class === 3 ? '🟩' : '🟦';
  const bgDesc   = data.background_class === 3
    ? 'Pure 3D perovskite crystal region — no structural defects visible'
    : 'Mixed 3D-2D perovskite interface region — no defects visible';

  list.innerHTML = `
    <div class="detection-card" style="border-left: 3px solid ${bgColor}; background: color-mix(in srgb, ${bgColor} 8%, var(--glass-bg));">
      <div class="dot-lg" style="background: ${bgColor}; border-radius: 6px; font-size: 1.2rem; display:flex; align-items:center; justify-content:center;">${bgIcon}</div>
      <div class="info" style="flex:1">
        <div class="name" style="color:${bgColor}; font-size:1rem;">Background Classified: ${bgClass}</div>
        <div class="conf" style="color: var(--text-secondary); margin-top:4px;">${bgDesc}</div>
      </div>
    </div>
    <div style="margin-top:12px; padding:12px; background: var(--card-bg); border-radius:10px; border:1px solid var(--border-color);">
      <div style="font-size:0.78rem; color:var(--text-secondary); margin-bottom:8px; text-transform:uppercase; letter-spacing:0.06em;">Background Class Scores (YOLO raw)</div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
        <span style="font-size:0.82rem; color:#10B981; width:140px;">3D_background</span>
        <div style="flex:1; height:8px; background:var(--border-color); border-radius:4px; overflow:hidden;">
          <div style="height:100%; width:${Math.min(score3 * 2000, 100)}%; background:#10B981; border-radius:4px; transition:width 0.5s;"></div>
        </div>
        <span style="font-size:0.78rem; color:var(--text-secondary); width:55px; text-align:right;">${score3.toFixed(5)}</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:0.82rem; color:#818CF8; width:140px;">3D-2D_background</span>
        <div style="flex:1; height:8px; background:var(--border-color); border-radius:4px; overflow:hidden;">
          <div style="height:100%; width:${Math.min(score4 * 2000, 100)}%; background:#818CF8; border-radius:4px; transition:width 0.5s;"></div>
        </div>
        <span style="font-size:0.78rem; color:var(--text-secondary); width:55px; text-align:right;">${score4.toFixed(5)}</span>
      </div>
    </div>
  `;

  resultsArea.classList.add('visible');
  resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderResults(data) {
  // Badge
  const badge = document.getElementById('resultBadge');
  badge.textContent = `${data.num_detections} defect${data.num_detections > 1 ? 's' : ''} found`;

  // Images
  document.getElementById('imgOriginal').src  = 'data:image/png;base64,' + data.original_b64;
  document.getElementById('imgAnnotated').src = 'data:image/png;base64,' + data.annotated_b64;

  // Detection list
  const list = document.getElementById('detectionList');
  list.innerHTML = '';
  data.detections.forEach(d => {
    const card = document.createElement('div');
    card.className = 'detection-card';
    card.innerHTML = `
      <div class="dot-lg" style="background: ${d.color};"></div>
      <div class="info">
        <div class="name">${d.class_name}</div>
        <div class="conf">${(d.confidence * 100).toFixed(1)}% confidence</div>
      </div>
    `;
    list.appendChild(card);
  });

  resultsArea.classList.add('visible');
  resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK TAB
// ═══════════════════════════════════════════════════════════════════
let benchmarkLoaded = false;

// Fallback metrics (embedded — used if API is unreachable)
const FALLBACK_METRICS = [
  { model: "yolo11m", mAP50: 0.4219, mAP50_95: 0.1905, precision: 0.4488, recall: 0.482, f1: 0.4648, epochs: 50, imgsz: 224, train_time_s: 4107.7 },
  { model: "yolo11s", mAP50: 0.3957, mAP50_95: 0.1734, precision: 0.4326, recall: 0.4582, f1: 0.445,  epochs: 50, imgsz: 224, train_time_s: 2237.0 },
  { model: "yolov8s", mAP50: 0.3779, mAP50_95: 0.1629, precision: 0.4553, recall: 0.4249, f1: 0.4396, epochs: 50, imgsz: 224, train_time_s: 1663.2 },
  { model: "yolo11l", mAP50: 0.3526, mAP50_95: 0.1602, precision: 0.4054, recall: 0.4315, f1: 0.418,  epochs: 50, imgsz: 224, train_time_s: 10607.9 },
  { model: "yolov8m", mAP50: 0.3528, mAP50_95: 0.1535, precision: 0.3976, recall: 0.43,   f1: 0.4132, epochs: 50, imgsz: 224, train_time_s: 2342.8 },
  { model: "yolov8l", mAP50: 0.23741,mAP50_95: 0.08912,precision: 0.30402,recall: 0.30826,f1: 0.30614,epochs: 50, imgsz: 224, train_time_s: 4219.24 },
];

async function loadBenchmark() {
  let metrics;
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    metrics = data.metrics && data.metrics.length ? data.metrics : FALLBACK_METRICS;
  } catch {
    metrics = FALLBACK_METRICS;
  }

  // Deduplicate: keep latest per model
  const latest = {};
  metrics.forEach(m => {
    const lbl = m.model;
    if (!latest[lbl] || (m.timestamp && m.timestamp > latest[lbl].timestamp)) {
      latest[lbl] = m;
    }
  });
  const rows = Object.values(latest).sort((a, b) => (b.mAP50_95 || 0) - (a.mAP50_95 || 0));

  renderStats(rows);
  renderTable(rows);
  renderBarChart(rows);
  renderModelExplorer(rows);
  benchmarkLoaded = true;
}

function fmtTime(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h ? `${h}h ${m}m` : `${m}m ${sec}s`;
}

function pct(v) { return (v * 100).toFixed(1) + '%'; }

function renderStats(rows) {
  const best = rows[0];
  const statsRow = document.getElementById('statsRow');
  const stats = [
    { value: rows.length, label: 'Models Trained' },
    { value: best.model.toUpperCase(), label: 'Best Model' },
    { value: pct(best.mAP50), label: 'Best mAP50' },
    { value: pct(best.mAP50_95), label: 'Best mAP50-95' },
    { value: pct(best.f1), label: 'Best F1 Score' },
  ];
  statsRow.innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-value">${s.value}</div>
      <div class="stat-label">${s.label}</div>
    </div>
  `).join('');
}

function renderTable(rows) {
  const metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1'];
  // Find best per metric
  const best = {};
  metrics.forEach(m => {
    best[m] = Math.max(...rows.map(r => r[m] || 0));
  });

  const tbody = document.getElementById('cmpTableBody');
  tbody.innerHTML = rows.map((r, i) => {
    const isWinner = i === 0;
    const cells = metrics.map(m => {
      const v = r[m] || 0;
      const isBest = Math.abs(v - best[m]) < 1e-6;
      return `<td class="${isBest ? 'best-cell' : ''}">${pct(v)}</td>`;
    });
    return `
      <tr class="${isWinner ? 'winner-row' : ''}">
        <td class="model-name">${isWinner ? '🏆 ' : ''}${r.model}</td>
        ${cells.join('')}
        <td>${r.epochs || '?'}</td>
        <td>${r.imgsz || '?'}</td>
        <td>${fmtTime(r.train_time_s || 0)}</td>
      </tr>
    `;
  }).join('');
}

function renderBarChart(rows) {
  const maxVal = Math.max(...rows.map(r => r.mAP50_95 || 0));
  const container = document.getElementById('barChart');
  container.innerHTML = rows.map((r, i) => {
    const v = r.mAP50_95 || 0;
    const w = maxVal > 0 ? (v / maxVal * 100) : 0;
    const isWinner = i === 0;
    return `
      <div class="bar-row">
        <div class="bar-label">${r.model}</div>
        <div class="bar-track">
          <div class="bar-fill ${isWinner ? 'winner' : ''}" style="width: 0%;" data-width="${w}"></div>
        </div>
        <div class="bar-val">${pct(v)}</div>
      </div>
    `;
  }).join('');

  // Animate bars after a tick
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 100);
  });
}

// ═══════════════════════════════════════════════════════════════════
// MODEL EXPLORER
// ═══════════════════════════════════════════════════════════════════
// Known run folder mapping for each model name (fallback when API doesn't provide run_folder)
const KNOWN_RUN_FOLDERS = {
  'yolo11m': 'detect/yolo11m_20260420_205814',
  'yolo11s': 'detect/yolo11s_20260420_202032',
  'yolo11l': 'detect/yolo11l_20260421_002408',
  'yolov8m': 'detect/yolov8m_20260419_213143',
  'yolov8s': 'detect/yolov8s_20260419_210343',
  'yolov8l': 'detect/yolov8l_20260419_221103',
};

function renderModelExplorer(rows) {
  const select = document.getElementById('modelSelect');
  if (!select) return;

  select.innerHTML = '';
  rows.forEach(r => {
    const opt = document.createElement('option');
    // Use run_folder from API, or look up in known map, or skip
    opt.value = r.run_folder || KNOWN_RUN_FOLDERS[r.model] || '';
    opt.textContent = r.model.toUpperCase();
    select.appendChild(opt);
  });

  const imgResults  = document.getElementById('imgResults');
  const imgConfusion = document.getElementById('imgConfusion');

  const PLACEHOLDER = 'data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22200%22><rect width=%22400%22 height=%22200%22 fill=%22%23111827%22/><text x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%236B7280%22 font-size=%2216%22>Image not available</text></svg>';

  const updateImages = () => {
    const runFolder = select.value;
    if (runFolder) {
      imgResults.src   = `/runs/${runFolder}/results.png`;
      imgConfusion.src = `/runs/${runFolder}/confusion_matrix_normalized.png`;
      imgResults.style.display   = 'block';
      imgConfusion.style.display = 'block';
      imgResults.onerror   = () => { imgResults.src = PLACEHOLDER; };
      imgConfusion.onerror = () => { imgConfusion.src = PLACEHOLDER; };
    } else {
      imgResults.style.display   = 'none';
      imgConfusion.style.display = 'none';
    }
  };

  select.addEventListener('change', updateImages);
  updateImages();
}


// ═══════════════════════════════════════════════════════════════════
// ACCORDION
// ═══════════════════════════════════════════════════════════════════
function toggleAccordion(header) {
  const item = header.parentElement;
  item.classList.toggle('open');
}

// ═══════════════════════════════════════════════════════════════════
// HEALTH CHECK — poll every 3s until model is loaded
// ═══════════════════════════════════════════════════════════════════
(function pollHealth() {
  const dot = document.querySelector('.nav-status .dot');
  const txt = document.querySelector('.nav-status span:last-child');

  async function check() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      if (data.model_loaded) {
        dot.style.background = '#4ADE80';
        txt.textContent = 'YOLO11m Online';
        return; // stop polling
      } else if (data.model_loading) {
        dot.style.background = '#F59E0B';
        txt.textContent = 'Model Loading...';
      } else {
        dot.style.background = '#EF4444';
        txt.textContent = 'Model Error';
        return; // stop polling on error
      }
    } catch {
      dot.style.background = '#EF4444';
      txt.textContent = 'Server Offline';
    }
    // Poll again in 3 seconds
    setTimeout(check, 3000);
  }

  check();
})();

// ═══════════════════════════════════════════════════════════════════
// MANUAL ANNOTATION & OPENCV LOGIC
// ═══════════════════════════════════════════════════════════════════
const annoRootDirInput = document.getElementById('annoRootDir');
const annoLabelsRoot   = document.getElementById('annoLabelsRoot');
const annoLoadRootBtn  = document.getElementById('annoLoadRoot');
const annoClassFolder  = document.getElementById('annoClassFolder');
const annoPrevBtn      = document.getElementById('annoPrev');
const annoNextBtn      = document.getElementById('annoNext');
const annoProgress     = document.getElementById('annoProgress');
const annoImage        = document.getElementById('annoImage');
const annoCanvas       = document.getElementById('annoCanvas');
const btnSaveLabels    = document.getElementById('btnSaveLabels');
const drawClassId      = document.getElementById('drawClassId');
const btnRunOpenCV     = document.getElementById('btnRunOpenCV');

let annoCtx = annoCanvas ? annoCanvas.getContext('2d') : null;
let annoState = {
  images: [],
  currentIndex: -1,
  currentFolder: '',
  boxes: [],
  boxColors: { 0: "#FF6B35", 1: "#00D4FF", 2: "#A855F7", 3: "#4ADE80", 4: "#FACC15" },
  isDrawing: false,
  startX: 0,
  startY: 0,
  currentRect: null,
  selectedBoxIndex: -1
};

// Slider live value display
['cvBrightness','cvContrast','cvThresh1','cvThresh2'].forEach(id => {
  const el = document.getElementById(id);
  const lbl = document.getElementById(id + 'Val');
  if (el && lbl) {
    el.addEventListener('input', () => { lbl.textContent = el.value; });
  }
});

if(annoLoadRootBtn) {
  annoLoadRootBtn.addEventListener('click', async () => {
    const root = annoRootDirInput.value.trim();
    if (!root) return;
    try {
      annoLoadRootBtn.textContent = 'Loading…';
      annoLoadRootBtn.disabled = true;
      const res = await fetch(`/api/dataset/folders?root_dir=${encodeURIComponent(root)}`);
      const data = await res.json();
      annoLoadRootBtn.textContent = 'Load';
      annoLoadRootBtn.disabled = false;

      if (data.error) {
         alert("Error: " + data.error + "\n\nCheck that the path exists on the server.");
         return;
      }
      if (!data.folders || data.folders.length === 0) {
         alert("No sub-folders found in:\n" + root);
         return;
      }

      annoClassFolder.innerHTML = '<option value="">-- Select a class folder --</option>';
      data.folders.forEach(f => {
         const opt = document.createElement('option');
         opt.value = f;
         opt.textContent = f;
         annoClassFolder.appendChild(opt);
      });
      annoClassFolder.disabled = false;
      const statsEl = document.getElementById('annoStats');
      if (statsEl) statsEl.textContent = `✓ Found ${data.folders.length} folders`;
    } catch (err) {
      annoLoadRootBtn.textContent = 'Load';
      annoLoadRootBtn.disabled = false;
      alert("Network error: " + err.message);
    }
  });


  annoClassFolder.addEventListener('change', async () => {
    const folder = annoClassFolder.value;
    if (!folder) return;
    
    const root = annoRootDirInput.value.trim();
    const folderPath = root + '\\' + folder;
    
    annoState.currentFolder = folder;
    
    try {
      annoClassFolder.disabled = true;
      const res = await fetch(`/api/dataset/images?folder_path=${encodeURIComponent(folderPath)}`);
      const data = await res.json();
      annoClassFolder.disabled = false;
      
      if (data.error) {
         alert("Error: " + data.error);
         return;
      }
      
      annoState.images = data.images;
      if (annoState.images.length > 0) {
         annoState.currentIndex = 0;
         btnSaveLabels.disabled = false;
         btnRunOpenCV.disabled = false;
         loadCurrentImage();
      } else {
         annoState.currentIndex = -1;
         annoProgress.textContent = "0 / 0";
         annoImage.src = "";
         clearCanvas();
      }
    } catch (err) {
       annoClassFolder.disabled = false;
    }
  });

  annoPrevBtn.addEventListener('click', () => {
     if (annoState.currentIndex > 0) {
        annoState.currentIndex--;
        loadCurrentImage();
     }
  });
  annoNextBtn.addEventListener('click', () => {
     if (annoState.currentIndex < annoState.images.length - 1) {
        annoState.currentIndex++;
        loadCurrentImage();
     }
  });
}

async function loadCurrentImage() {
   const imgPath = annoState.images[annoState.currentIndex];
   annoProgress.textContent = `${annoState.currentIndex + 1} / ${annoState.images.length}`;

   annoPrevBtn.disabled = annoState.currentIndex === 0;
   annoNextBtn.disabled = annoState.currentIndex === annoState.images.length - 1;

   // Load labels first (async), then load image
   const labelsRoot = annoLabelsRoot ? annoLabelsRoot.value.trim() : '';
   try {
      const url = `/api/dataset/labels?image_path=${encodeURIComponent(imgPath)}&class_folder=${encodeURIComponent(annoState.currentFolder)}&labels_root=${encodeURIComponent(labelsRoot)}`;
      const res = await fetch(url);
      const data = await res.json();
      annoState.boxes = data.boxes || [];
      annoState.selectedBoxIndex = -1;

      // Show which label file was looked up
      const statsEl = document.getElementById('annoStats');
      if (statsEl && data.label_file) {
        const found = data.found ? `✅ ${data.boxes.length} box(es)` : '⬜ No labels yet';
        statsEl.textContent = found + ' — ' + data.label_file.split('\\').pop();
      }
   } catch(err) {
      annoState.boxes = [];
   }

   annoImage.onload = () => {
      syncCanvas();
   };
   annoImage.src = `/api/dataset/image?image_path=${encodeURIComponent(imgPath)}&t=${Date.now()}`;
}

// Return the pixel rect of the actual image content inside the img element
// (accounts for object-fit: contain letterbox/pillarbox padding)
function getImageContentRect() {
   const iw = annoImage.naturalWidth;
   const ih = annoImage.naturalHeight;
   const ew = annoImage.clientWidth;   // element width
   const eh = annoImage.clientHeight;  // element height
   if (!iw || !ih || !ew || !eh) return { x:0, y:0, w:ew, h:eh };

   const scaleW = ew / iw;
   const scaleH = eh / ih;
   const scale  = Math.min(scaleW, scaleH);  // object-fit: contain uses min

   const rw = iw * scale;
   const rh = ih * scale;
   const rx = (ew - rw) / 2;  // horizontal offset (pillarbox)
   const ry = (eh - rh) / 2;  // vertical offset   (letterbox)
   return { x: rx, y: ry, w: rw, h: rh };
}

// Sync canvas to sit exactly over the rendered image content area
function syncCanvas() {
   if (!annoImage || !annoImage.naturalWidth) return;

   const { x, y, w, h } = getImageContentRect();

   // Position canvas pixel-perfectly over image content (inside img element)
   annoCanvas.style.left   = x + 'px';
   annoCanvas.style.top    = y + 'px';
   annoCanvas.style.width  = w + 'px';
   annoCanvas.style.height = h + 'px';
   annoCanvas.width  = Math.round(w);
   annoCanvas.height = Math.round(h);

   drawAllBoxes();
}

// Ensure responsiveness
window.addEventListener('resize', () => {
   if (annoImage && annoImage.naturalWidth) syncCanvas();
});


// ── 2. CANVAS DRAWING LOGIC ───────────────────────────────────────

function clearCanvas() {
   if(annoCtx) annoCtx.clearRect(0, 0, annoCanvas.width, annoCanvas.height);
}

function drawAllBoxes() {
   clearCanvas();
   if(!annoCtx) return;
   const cw = annoCanvas.width;
   const ch = annoCanvas.height;

   annoState.boxes.forEach((box, i) => {
      const bw = box.width   * cw;
      const bh = box.height  * ch;
      const bx = (box.x_center * cw) - (bw / 2);
      const by = (box.y_center * ch) - (bh / 2);

      const color = annoState.boxColors[box.class_id] || '#FFFFFF';
      const selected = i === annoState.selectedBoxIndex;

      // Filled highlight for selected box
      if (selected) {
         annoCtx.fillStyle = 'rgba(255,80,80,0.18)';
         annoCtx.fillRect(bx, by, bw, bh);
      }

      // Border
      annoCtx.strokeStyle = selected ? '#FF4444' : color;
      annoCtx.lineWidth   = selected ? 3 : 2;
      annoCtx.strokeRect(bx, by, bw, bh);

      // Class label chip
      const label = `C${box.class_id}`;
      annoCtx.font = 'bold 11px Inter, sans-serif';
      const tw = annoCtx.measureText(label).width;
      annoCtx.fillStyle = selected ? '#FF4444' : color;
      annoCtx.fillRect(bx, Math.max(0, by - 16), tw + 8, 16);
      annoCtx.fillStyle = '#000';
      annoCtx.fillText(label, bx + 4, Math.max(10, by - 4));
   });

   // Draw in-progress drag rectangle
   if (annoState.isDrawing && annoState.currentRect) {
      annoCtx.strokeStyle = 'rgba(255,255,255,0.9)';
      annoCtx.lineWidth   = 1.5;
      annoCtx.setLineDash([6, 4]);
      annoCtx.strokeRect(
         annoState.currentRect.x, annoState.currentRect.y,
         annoState.currentRect.w, annoState.currentRect.h
      );
      annoCtx.setLineDash([]);
   }
}

// Mouse position relative to the CANVAS (which is offset to match image content)
function getMousePos(evt) {
   const rect = annoCanvas.getBoundingClientRect();
   return {
      x: (evt.clientX - rect.left) * (annoCanvas.width  / rect.width),
      y: (evt.clientY - rect.top)  * (annoCanvas.height / rect.height)
   };
}

if(annoCanvas) {
  annoCanvas.addEventListener('mousedown', (e) => {
     const pos = getMousePos(e);
     
     // Check if clicking an existing box to select it
     let clickedBox = -1;
     const cw = annoCanvas.width;
     const ch = annoCanvas.height;
     
     // Reverse iterate so top-most box is selected
     for (let i = annoState.boxes.length - 1; i >= 0; i--) {
        const box = annoState.boxes[i];
        const w = box.width * cw;
        const h = box.height * ch;
        const bx = (box.x_center * cw) - (w / 2);
        const by = (box.y_center * ch) - (h / 2);
        if (pos.x >= bx && pos.x <= bx + w && pos.y >= by && pos.y <= by + h) {
           clickedBox = i;
           break;
        }
     }
     
     if (clickedBox !== -1) {
        annoState.selectedBoxIndex = clickedBox;
        drawAllBoxes();
        return; 
     }
     
     // Otherwise start drawing new box
     annoState.selectedBoxIndex = -1;
     annoState.isDrawing = true;
     annoState.startX = pos.x;
     annoState.startY = pos.y;
     annoState.currentRect = null;
  });

  annoCanvas.addEventListener('mousemove', (e) => {
     if (!annoState.isDrawing) return;
     
     const pos = getMousePos(e);
     const w = pos.x - annoState.startX;
     const h = pos.y - annoState.startY;
     
     annoState.currentRect = { 
         x: w < 0 ? pos.x : annoState.startX, 
         y: h < 0 ? pos.y : annoState.startY, 
         w: Math.abs(w), 
         h: Math.abs(h) 
     };
     
     drawAllBoxes();
  });

  window.addEventListener('mouseup', () => {
     if (!annoState.isDrawing) return;
     annoState.isDrawing = false;
     
     if (annoState.currentRect && annoState.currentRect.w > 5 && annoState.currentRect.h > 5) {
        // Normalize to YOLO coordinates (0.0 to 1.0)
        const cw = annoCanvas.width;
        const ch = annoCanvas.height;
        const obj = annoState.currentRect;
        
        const x_center = (obj.x + (obj.w / 2)) / cw;
        const y_center = (obj.y + (obj.h / 2)) / ch;
        const width = obj.w / cw;
        const height = obj.h / ch;
        
        annoState.boxes.push({
           class_id: parseInt(drawClassId.value),
           x_center, y_center, width, height
        });
        annoState.selectedBoxIndex = annoState.boxes.length - 1; // auto-select new box
     }
     annoState.currentRect = null;
     drawAllBoxes();
  });
}

// Delete selected box
window.addEventListener('keydown', (e) => {
   const tab = document.getElementById('tab-annotation');
   if(!tab) return;
   
   // Only trigger if annotation tab is active and not typing in an input
   if (!tab.classList.contains('active')) return;
   if (e.target.tagName === 'INPUT') return;
   
   if ((e.key === "Delete" || e.key === "Backspace") && annoState.selectedBoxIndex !== -1) {
      annoState.boxes.splice(annoState.selectedBoxIndex, 1);
      annoState.selectedBoxIndex = -1;
      drawAllBoxes();
   }
});

// ── 3. SAVING LABELS ──────────────────────────────────────────────

if(btnSaveLabels) {
  btnSaveLabels.addEventListener('click', async () => {
     if (annoState.currentIndex === -1) return;
     const imgPath = annoState.images[annoState.currentIndex];
     const labelsRoot = annoLabelsRoot ? annoLabelsRoot.value.trim() : '';

     const reqBody = {
        image_path: imgPath,
        class_folder: annoState.currentFolder,
        labels_root: labelsRoot,
        boxes: annoState.boxes
     };

     try {
        btnSaveLabels.textContent = "Saving...";
        const res = await fetch('/api/dataset/labels', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify(reqBody)
        });
        const data = await res.json();
        if (!data.success) {
           alert("Failed to save: " + data.error);
        } else {
           btnSaveLabels.textContent = "✅ Saved!";
        }
        setTimeout(() => btnSaveLabels.textContent = "💾 Save", 1200);
     } catch(e) {
        alert("Error saving: " + e.message);
        btnSaveLabels.textContent = "💾 Save";
     }
  });
}

// ── 4. OPENCV AUTO-DETECT ─────────────────────────────────────────

if(btnRunOpenCV) {
  btnRunOpenCV.addEventListener('click', async () => {
     if (annoState.currentIndex === -1) return;

     const imgPath = annoState.images[annoState.currentIndex];
     const labelsRoot = annoLabelsRoot ? annoLabelsRoot.value.trim() : '';
     btnRunOpenCV.textContent = "Processing...";
     btnRunOpenCV.disabled = true;

     const reqBody = {
        image_path: imgPath,
        class_folder: annoState.currentFolder,
        labels_root: labelsRoot,
        class_id: parseInt(drawClassId.value),
        method: document.getElementById('cvMethod').value,
        threshold1: parseInt(document.getElementById('cvThresh1').value),
        threshold2: parseInt(document.getElementById('cvThresh2').value),
        brightness: parseInt(document.getElementById('cvBrightness').value),
        contrast: parseFloat(document.getElementById('cvContrast').value),
        min_area: parseInt(document.getElementById('cvMinArea').value),
        max_area: parseInt(document.getElementById('cvMaxArea').value),
        use_clahe: document.getElementById('cvClahe').checked,
        overwrite: document.getElementById('cvOverwrite').checked,
        clahe_clip: 2.0,
        clahe_grid: 8
     };

     try {
        const res = await fetch('/api/opencv_detect', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify(reqBody)
        });
        const data = await res.json();
        btnRunOpenCV.textContent = "🚀 Run OpenCV Detect";
        btnRunOpenCV.disabled = false;

        if (data.error) {
           alert("OpenCV Error: " + data.error);
        } else {
           annoState.boxes = data.boxes || [];
           annoState.selectedBoxIndex = -1;
           drawAllBoxes();
        }
     } catch(err) {
        alert("API Error: " + err.message);
        btnRunOpenCV.textContent = "🚀 Run OpenCV Detect";
        btnRunOpenCV.disabled = false;
     }
  });
}
