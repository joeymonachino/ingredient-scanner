document.addEventListener("DOMContentLoaded", () => {
    const queryInput = document.getElementById("query-text");
    const sampleQueryButtons = document.querySelectorAll("[data-sample-query]");
    const photoInput = document.getElementById("label-photo");
    const photoStatus = document.getElementById("photo-upload-status");
    const cropTool = document.getElementById("crop-tool");
    const cropPreviewImage = document.getElementById("crop-preview-image");
    const cropOverlay = document.getElementById("crop-overlay");
    const cropReset = document.getElementById("crop-reset");
    const uploadedLabelPreview = document.getElementById("uploaded-label-preview");
    const photoUploadStatusPill = document.getElementById("photo-upload-status-pill");
    const photoUploadOcrStatusPill = document.getElementById("photo-upload-ocr-status-pill");
    const photoUploadReconstructionPill = document.getElementById("photo-upload-reconstruction-pill");
    const photoUploadNextStep = document.getElementById("photo-upload-next-step");
    const photoUploadHelperText = document.getElementById("photo-upload-helper-text");
    const photoUploadReconstructionNote = document.getElementById("photo-upload-reconstruction-note");
    const photoUploadOcrLines = document.getElementById("photo-upload-ocr-lines");
    const photoUploadOcrConfidence = document.getElementById("photo-upload-ocr-confidence");
    const photoUploadOcrProvider = document.getElementById("photo-upload-ocr-provider");
    const photoUploadRepairRatio = document.getElementById("photo-upload-repair-ratio");
    const ocrQueryText = document.getElementById("ocr-query-text");
    const ocrSuggestedText = document.getElementById("ocr-suggested-text");
    const useOcrSuggestion = document.getElementById("use-ocr-suggestion");
    const ocrMergedSuggestedText = document.getElementById("ocr-merged-suggested-text");
    const useOcrMergedSuggestion = document.getElementById("use-ocr-merged-suggestion");
    const ocrSuggestionCard = document.getElementById("ocr-suggestion-card");
    const ocrChangeList = document.getElementById("ocr-change-list");
    const ocrMergedDetails = document.getElementById("ocr-merged-details");
    const ocrRawDetails = document.getElementById("ocr-raw-details");
    const ocrRawText = document.getElementById("ocr-raw-text");
    const cropInputs = {
        top: document.getElementById("crop-top"),
        bottom: document.getElementById("crop-bottom"),
        left: document.getElementById("crop-left"),
        right: document.getElementById("crop-right"),
    };

    sampleQueryButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (!queryInput) return;
            queryInput.value = button.dataset.sampleQuery || "";
            queryInput.focus();
        });
    });

    const updateCropOverlay = () => {
        if (!cropOverlay) return;
        let top = Number(cropInputs.top?.value || 0);
        let bottom = Number(cropInputs.bottom?.value || 100);
        let left = Number(cropInputs.left?.value || 0);
        let right = Number(cropInputs.right?.value || 100);

        if (bottom <= top) {
            bottom = Math.min(100, top + 1);
            if (cropInputs.bottom) cropInputs.bottom.value = String(bottom);
        }
        if (right <= left) {
            right = Math.min(100, left + 1);
            if (cropInputs.right) cropInputs.right.value = String(right);
        }

        cropOverlay.style.setProperty("--crop-top", `${top}%`);
        cropOverlay.style.setProperty("--crop-left", `${left}%`);
        cropOverlay.style.setProperty("--crop-width", `${right - left}%`);
        cropOverlay.style.setProperty("--crop-height", `${bottom - top}%`);
    };

    const resetCrop = () => {
        if (cropInputs.top) cropInputs.top.value = "0";
        if (cropInputs.left) cropInputs.left.value = "0";
        if (cropInputs.right) cropInputs.right.value = "100";
        if (cropInputs.bottom) cropInputs.bottom.value = "100";
        updateCropOverlay();
    };

    Object.values(cropInputs).forEach((input) => {
        if (!input) return;
        input.addEventListener("input", updateCropOverlay);
    });

    if (cropReset) {
        cropReset.addEventListener("click", resetCrop);
    }

    if (useOcrSuggestion && ocrQueryText && ocrSuggestedText) {
        useOcrSuggestion.addEventListener("click", () => {
            ocrQueryText.value = ocrSuggestedText.textContent?.trim() || "";
            ocrQueryText.focus();
        });
    }

    if (useOcrMergedSuggestion && ocrQueryText && ocrMergedSuggestedText) {
        useOcrMergedSuggestion.addEventListener("click", () => {
            ocrQueryText.value = ocrMergedSuggestedText.textContent?.trim() || "";
            ocrQueryText.focus();
        });
    }

    if (photoInput && photoStatus && cropTool && cropPreviewImage) {
        photoInput.addEventListener("change", () => {
            const file = photoInput.files && photoInput.files[0];
            photoStatus.textContent = file
                ? `Selected: ${file.name}. Adjust the crop box if the ingredient panel is only part of the photo.`
                : "Use a clear photo of the ingredient panel. OCR now runs in your browser after upload.";
            if (!file) {
                cropTool.classList.remove("is-ready");
                cropPreviewImage.removeAttribute("src");
                resetCrop();
                return;
            }
            const reader = new FileReader();
            reader.onload = () => {
                cropPreviewImage.src = String(reader.result || "");
                cropTool.classList.add("is-ready");
                resetCrop();
            };
            reader.readAsDataURL(file);
        });
    }

    const renderOcrSuggestions = (ocr) => {
        if (!ocrSuggestionCard || !ocrSuggestedText || !ocrChangeList) return;
        ocrSuggestedText.textContent = ocr.suggested_text || "";
        if (ocrMergedSuggestedText) {
            ocrMergedSuggestedText.textContent = ocr.suggested_merged_text || "";
        }
        ocrChangeList.innerHTML = "";
        (ocr.suggested_changes || []).forEach((change) => {
            const line = document.createElement("p");
            line.className = "soft-copy";
            line.innerHTML = `<strong>${change.from}</strong> -> <strong>${change.to}</strong>`;
            ocrChangeList.appendChild(line);
        });
        ocrSuggestionCard.hidden = !(ocr.has_suggested_changes || ocr.suggested_merged_text);
        if (ocrMergedDetails) {
            ocrMergedDetails.hidden = !ocr.suggested_merged_text;
        }
        if (useOcrMergedSuggestion) {
            useOcrMergedSuggestion.hidden = !ocr.suggested_merged_text;
        }
        if (ocrRawText) {
            ocrRawText.textContent = ocr.raw_text || "";
        }
        if (ocrRawDetails) {
            ocrRawDetails.hidden = !ocr.raw_text;
        }
    };

    const updateBrowserOcrUi = (ocr, tesseractConfidence) => {
        if (photoUploadStatusPill) photoUploadStatusPill.textContent = "uploaded";
        if (photoUploadOcrStatusPill) photoUploadOcrStatusPill.textContent = (ocr.status || "ready").replace(/-/g, " ");
        if (photoUploadReconstructionPill) {
            photoUploadReconstructionPill.textContent = `reconstruction ${ocr.reconstruction_confidence || "high"}`;
            photoUploadReconstructionPill.className = `confidence-pill ${ocr.reconstruction_confidence || "high"}`;
        }
        if (photoUploadNextStep) photoUploadNextStep.textContent = ocr.message || "Browser OCR finished. Review the extracted text before analysis.";
        if (photoUploadHelperText) photoUploadHelperText.textContent = "The OCR step ran in your browser with free Tesseract.js, then the app applied label-specific cleanup and repair suggestions.";
        if (photoUploadReconstructionNote) photoUploadReconstructionNote.textContent = ocr.reconstruction_note || "";
        if (photoUploadOcrLines) photoUploadOcrLines.textContent = String(ocr.line_count || 0);
        if (photoUploadOcrConfidence) {
            photoUploadOcrConfidence.textContent = Number.isFinite(tesseractConfidence)
                ? `${Math.round(tesseractConfidence)}%`
                : "n/a";
        }
        if (photoUploadOcrProvider) photoUploadOcrProvider.textContent = ocr.provider || "browser tesseract";
        if (photoUploadRepairRatio) photoUploadRepairRatio.textContent = String(ocr.repair_ratio ?? 0);
        if (ocrQueryText) {
            ocrQueryText.value = ocr.suggested_merged_text || ocr.suggested_text || ocr.candidate_text || ocr.raw_text || "";
        }
        renderOcrSuggestions(ocr);
    };

    const runBrowserOcr = async () => {
        if (!uploadedLabelPreview || !ocrQueryText) {
            return;
        }
        if (typeof window.Tesseract === "undefined") {
            if (photoUploadOcrStatusPill) photoUploadOcrStatusPill.textContent = "unavailable";
            if (photoUploadNextStep) photoUploadNextStep.textContent = "Browser OCR could not start because Tesseract.js did not load.";
            return;
        }
        if (uploadedLabelPreview.dataset.ocrStarted === "true") {
            return;
        }
        uploadedLabelPreview.dataset.ocrStarted = "true";
        if (photoUploadOcrStatusPill) photoUploadOcrStatusPill.textContent = "running";
        if (photoUploadNextStep) photoUploadNextStep.textContent = "Running OCR in your browser now. This can take a few seconds on mobile.";
        try {
            const result = await window.Tesseract.recognize(uploadedLabelPreview.src, "eng", {
                logger: (message) => {
                    if (!photoUploadHelperText || message.status !== "recognizing text") return;
                    const percent = typeof message.progress === "number" ? `${Math.round(message.progress * 100)}%` : "";
                    photoUploadHelperText.textContent = `Browser OCR is scanning the label${percent ? ` (${percent})` : ""}.`;
                },
            });
            const response = await fetch("/api/ocr-cleanup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: result?.data?.text || "",
                    confidence: result?.data?.confidence ?? null,
                }),
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
                throw new Error(payload.message || "OCR cleanup failed.");
            }
            updateBrowserOcrUi(payload.ocr, result?.data?.confidence ?? null);
        } catch (error) {
            if (photoUploadOcrStatusPill) photoUploadOcrStatusPill.textContent = "unavailable";
            if (photoUploadNextStep) photoUploadNextStep.textContent = "Browser OCR did not finish cleanly on this photo.";
            if (photoUploadHelperText) photoUploadHelperText.textContent = error instanceof Error ? error.message : "Something went wrong while running browser OCR.";
        }
    };

    if (uploadedLabelPreview) {
        if (uploadedLabelPreview.complete) {
            runBrowserOcr();
        } else {
            uploadedLabelPreview.addEventListener("load", runBrowserOcr, { once: true });
        }
    }

    updateCropOverlay();
});
