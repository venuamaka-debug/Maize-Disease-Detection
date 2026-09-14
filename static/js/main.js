const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const MAX_SIZE = 8 * 1024 * 1024;

const uploadEmpty = document.getElementById("uploadEmpty");
const uploadFilled = document.getElementById("uploadFilled");
const fileInput = document.getElementById("fileInput");
const filledThumb = document.getElementById("filledThumb");
const filledName = document.getElementById("filledName");
const filledMeta = document.getElementById("filledMeta");
const removeFileBtn = document.getElementById("removeFileBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const resultSection = document.getElementById("resultSection");
const resultImg = document.getElementById("resultImg");
const resultStatus = document.getElementById("resultStatus");
const resultClass = document.getElementById("resultClass");
const gaugeRow = document.getElementById("gaugeRow");
const gauge = document.getElementById("gauge");
const gaugePct = document.getElementById("gaugePct");
const gaugeCaption = document.getElementById("gaugeCaption");
const observedPattern = document.getElementById("observedPattern");
const fieldGuidance = document.getElementById("fieldGuidance");
const inconclusiveNote = document.getElementById("inconclusiveNote");
const noMatchNote = document.getElementById("noMatchNote");
const reanalyzeBtn = document.getElementById("reanalyzeBtnTop");

let selectedFile = null;

const CONTENT = {
  Northern_Corn_Leaf_Blight: {
    label: "Northern Corn Leaf Blight",
    pattern: "Long, cigar-shaped grey-green lesions running parallel to the leaf's length.",
    guidance: "Consider consulting a local agronomist about fungicide timing and resistant hybrids for future planting."
  },
  Common_Rust: {
    label: "Common Rust",
    pattern: "Small, reddish-brown raised pustules scattered across both leaf surfaces.",
    guidance: "Monitor spread across the field. Severe early-season infections may warrant a fungicide application."
  },
  Gray_Leaf_Spot: {
    label: "Gray Leaf Spot",
    pattern: "Rectangular tan-to-grey lesions bounded by leaf veins, often starting on lower leaves.",
    guidance: "Check lower canopy leaves across the field, as this condition typically progresses upward."
  },
  Healthy: {
    label: "Healthy",
    pattern: "No lesions, discoloration, or pustules consistent with the conditions this system checks for.",
    guidance: "Continue routine monitoring, particularly after periods of high humidity or rainfall."
  }
};

function showError(msg) { errorMsg.textContent = msg; errorMsg.classList.add("active"); }
function clearError() { errorMsg.textContent = ""; errorMsg.classList.remove("active"); }

function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return "This file type isn't supported. Please upload a JPG, JPEG or PNG image.";
  if (file.size > MAX_SIZE) return "That image is too large. Please choose a file under 8 MB.";
  return null;
}

function formatSize(bytes) {
  return bytes > 1024 * 1024 ? (bytes / (1024 * 1024)).toFixed(1) + " MB" : Math.round(bytes / 1024) + " KB";
}

function handleFile(file) {
  clearError();
  const err = validateFile(file);
  if (err) { showError(err); return; }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  filledThumb.src = url;

  const tempImg = new Image();
  tempImg.onload = () => {
    filledMeta.textContent = `${tempImg.naturalWidth}×${tempImg.naturalHeight} · ${formatSize(file.size)}`;
  };
  tempImg.src = url;

  filledName.textContent = file.name;
  uploadEmpty.style.display = "none";
  uploadFilled.classList.add("active");
  analyzeBtn.disabled = false;
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  uploadFilled.classList.remove("active");
  uploadEmpty.style.display = "block";
  analyzeBtn.disabled = true;
  clearError();
}

uploadEmpty.addEventListener("click", (e) => { if (e.target.id !== "cameraBtn") fileInput.click(); });
const cameraBtn = document.getElementById("cameraBtn");
const cameraInput = document.getElementById("cameraInput");
if (cameraBtn) {
  cameraBtn.addEventListener("click", (e) => { e.stopPropagation(); cameraInput.click(); });
  cameraInput.addEventListener("change", (e) => { e.stopPropagation(); if (cameraInput.files.length) handleFile(cameraInput.files[0]); });
}
uploadEmpty.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
uploadEmpty.addEventListener("dragover", (e) => { e.preventDefault(); uploadEmpty.classList.add("dragover"); });
uploadEmpty.addEventListener("dragleave", () => uploadEmpty.classList.remove("dragover"));
uploadEmpty.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadEmpty.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
removeFileBtn.addEventListener("click", resetUpload);

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) { showError("Please select an image before continuing."); return; }
  clearError();
  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("loading");
  analyzeBtn.textContent = "Analyzing leaf image…";

  const formData = new FormData();
  formData.append("image", selectedFile);
  const imgSrc = filledThumb.src;

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "The analysis could not be completed. Please try again.");
    } else {
      displayResult(data, imgSrc);
    }
  } catch (e) {
    showError("The analysis could not be completed. Please check your connection and try again.");
  }

  analyzeBtn.classList.remove("loading");
  analyzeBtn.textContent = "Analyze leaf";
  analyzeBtn.disabled = false;
});

function resetResultPanels() {
  gaugeRow.style.display = "none";
  inconclusiveNote.classList.remove("active");
  noMatchNote.classList.remove("active");
  observedPattern.closest(".result-section").style.display = "none";
  fieldGuidance.closest(".result-section").style.display = "none";
}

function displayResult(data, imgSrc) {
  resultImg.src = imgSrc;
  resetResultPanels();

  if (data.status === "no_match") {
    resultClass.className = "no-match";
    resultClass.textContent = "No match found";
    resultStatus.textContent = "This doesn't appear to be a maize leaf";
    noMatchNote.classList.add("active");
    noMatchNote.textContent = "We couldn't confidently identify this as a maize leaf. Try uploading a clearer, well-lit photo of a single leaf.";
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }

  const isHealthy = data.predicted_class === "Healthy";
  const info = CONTENT[data.predicted_class] || { label: data.predicted_class, pattern: "", guidance: "" };
  const pct = Math.min(99, Math.round(data.confidence * 100));

  gaugeRow.style.display = "flex";

  if (data.status === "unrecognised") {
    resultClass.className = "unrecognised";
    resultClass.textContent = "Unrecognised";
    resultStatus.textContent = "Leaf detected, but condition unclear";
    gauge.style.setProperty("--gauge-color", "var(--amber)");
    gauge.style.setProperty("--pct", 0);
    requestAnimationFrame(() => { gauge.style.setProperty("--pct", pct); });
    gaugePct.textContent = pct + "%";
    gaugeCaption.innerHTML = `<strong>${pct}%</strong> best-guess confidence`;
    inconclusiveNote.classList.add("active");
    inconclusiveNote.textContent = "This looks like a leaf, but the model isn't confident enough to name a condition. Try retaking the photo with better lighting, closer framing, or a plainer background.";
  } else {
    resultClass.className = isHealthy ? "healthy" : "disease";
    resultClass.textContent = info.label;
    resultStatus.textContent = isHealthy ? "No condition detected" : "Condition detected";
    const gaugeColor = isHealthy ? "var(--leaf)" : "var(--clay)";
    gauge.style.setProperty("--gauge-color", gaugeColor);
    gauge.style.setProperty("--pct", 0);
    requestAnimationFrame(() => { gauge.style.setProperty("--pct", pct); });
    gaugePct.textContent = pct + "%";
    gaugeCaption.innerHTML = `<strong>${pct}%</strong> model confidence`;
    observedPattern.closest(".result-section").style.display = "block";
    fieldGuidance.closest(".result-section").style.display = "block";
    observedPattern.textContent = info.pattern;
    fieldGuidance.textContent = info.guidance;
  }

  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

reanalyzeBtn.addEventListener("click", () => {
  resetUpload();
  resultSection.hidden = true;
});
