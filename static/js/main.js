const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const MAX_SIZE = 8 * 1024 * 1024;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultCard = document.getElementById("resultCard");
const resultUploadedImg = document.getElementById("resultUploadedImg");
const diagnosisBadge = document.getElementById("diagnosisBadge");
const specimenTimestamp = document.getElementById("specimenTimestamp");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceReadout = document.getElementById("confidenceReadout");
const interpretation = document.getElementById("interpretation");
const inconclusiveNote = document.getElementById("inconclusiveNote");
const reanalyzeBtn = document.getElementById("reanalyzeBtn");

let selectedFile = null;

const INTERPRETATIONS = {
  Northern_Corn_Leaf_Blight: "Long, cigar-shaped grey-green lesions are consistent with Northern Corn Leaf Blight. Consider consulting a local agronomist about fungicide timing.",
  Common_Rust: "Small reddish-brown pustules on both leaf surfaces are consistent with Common Rust.",
  Gray_Leaf_Spot: "Rectangular tan-to-grey lesions running parallel to leaf veins are consistent with Gray Leaf Spot.",
  Healthy: "No visible signs of the diseases this system checks for were detected in this leaf."
};

function showError(msg) { errorMsg.textContent = msg; errorMsg.classList.add("active"); }
function clearError() { errorMsg.textContent = ""; errorMsg.classList.remove("active"); }

function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return "This file type is not supported. Please upload a JPG, JPEG or PNG image.";
  if (file.size > MAX_SIZE) return "The selected image is too large. Please choose a smaller image.";
  return null;
}

function handleFile(file) {
  clearError();
  const err = validateFile(file);
  if (err) { showError(err); return; }
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewWrap.classList.add("active");
  analyzeBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) { showError("Please select an image before continuing."); return; }
  clearError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  const formData = new FormData();
  formData.append("image", selectedFile);
  const imgSrcForResult = previewImg.src;

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "The analysis could not be completed. Please try again.");
    } else {
      displayResult(data, imgSrcForResult);
      // Clear the left-column preview once the result plate on the right takes over —
      // one image doing one job, not the same leaf shown twice.
      previewWrap.classList.remove("active");
      fileInput.value = "";
      selectedFile = null;
    }
  } catch (e) {
    showError("The analysis could not be completed. Please try again.");
  }
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyze";
});

function displayResult(data, imgSrc) {
  resultPlaceholder.style.display = "none";
  resultCard.classList.add("active");
  resultUploadedImg.src = imgSrc;
  specimenTimestamp.textContent = new Date().toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

  const isHealthy = data.predicted_class === "Healthy";
  const label = data.inconclusive ? "Inconclusive" : data.predicted_class.replace(/_/g, " ");
  diagnosisBadge.textContent = label;
  diagnosisBadge.className = "diagnosis-badge " + (data.inconclusive ? "inconclusive" : (isHealthy ? "healthy" : "disease"));

  const pct = Math.round(data.confidence * 100);
  confidenceFill.className = "confidence-fill " + (isHealthy ? "healthy" : "");
  requestAnimationFrame(() => { confidenceFill.style.width = pct + "%"; });
  confidenceReadout.textContent = `${pct}% confidence`;

  if (data.inconclusive) {
    inconclusiveNote.classList.add("active");
    inconclusiveNote.textContent = "This result falls below the confidence threshold. Consider retaking the photo in better lighting or from a closer angle.";
    interpretation.textContent = "";
  } else {
    inconclusiveNote.classList.remove("active");
    interpretation.textContent = INTERPRETATIONS[data.predicted_class] || "";
  }
}

reanalyzeBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewWrap.classList.remove("active");
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyze";
  resultCard.classList.remove("active");
  resultPlaceholder.style.display = "block";
  confidenceFill.style.width = "0%";
  clearError();
});
