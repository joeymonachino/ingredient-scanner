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

    const updateBrowserOcrUi = (ocr, tesseractConfidence, variantName) => {
        if (photoUploadStatusPill) photoUploadStatusPill.textContent = "uploaded";
        if (photoUploadOcrStatusPill) photoUploadOcrStatusPill.textContent = (ocr.status || "ready").replace(/-/g, " ");
        if (photoUploadReconstructionPill) {
            photoUploadReconstructionPill.textContent = `reconstruction ${ocr.reconstruction_confidence || "high"}`;
            photoUploadReconstructionPill.className = `confidence-pill ${ocr.reconstruction_confidence || "high"}`;
        }
        if (photoUploadNextStep) photoUploadNextStep.textContent = ocr.message || "Browser OCR finished. Review the extracted text before analysis.";
        if (photoUploadHelperText) {
            const variantText = variantName ? ` Best scan pass: ${variantName}.` : "";
            photoUploadHelperText.textContent = `The OCR step ran in your browser with free Tesseract.js, then the app applied label-specific cleanup and repair suggestions.${variantText}`;
        }
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

    const cloneCanvas = (sourceCanvas) => {
        const canvas = document.createElement("canvas");
        canvas.width = sourceCanvas.width;
        canvas.height = sourceCanvas.height;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(sourceCanvas, 0, 0);
        return canvas;
    };

    const buildBaseCanvasFromImage = (image, scale = 1) => {
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round((image.naturalWidth || image.width) * scale));
        canvas.height = Math.max(1, Math.round((image.naturalHeight || image.height) * scale));
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        return canvas;
    };

    const cropCanvas = (sourceCanvas, bounds) => {
        const canvas = document.createElement("canvas");
        canvas.width = bounds.width;
        canvas.height = bounds.height;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(
            sourceCanvas,
            bounds.left,
            bounds.top,
            bounds.width,
            bounds.height,
            0,
            0,
            bounds.width,
            bounds.height,
        );
        return canvas;
    };

    const detectTextBounds = (sourceCanvas) => {
        const ctx = sourceCanvas.getContext("2d", { willReadFrequently: true });
        const { width, height } = sourceCanvas;
        const imageData = ctx.getImageData(0, 0, width, height).data;
        let minX = width;
        let minY = height;
        let maxX = 0;
        let maxY = 0;
        let darkPixelCount = 0;

        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const index = (y * width + x) * 4;
                const gray = 0.299 * imageData[index] + 0.587 * imageData[index + 1] + 0.114 * imageData[index + 2];
                if (gray < 165) {
                    darkPixelCount += 1;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (darkPixelCount < Math.max(250, Math.round(width * height * 0.004))) {
            return null;
        }

        const horizontalPadding = Math.max(14, Math.round(width * 0.035));
        const verticalPadding = Math.max(14, Math.round(height * 0.035));
        const left = Math.max(0, minX - horizontalPadding);
        const top = Math.max(0, minY - verticalPadding);
        const right = Math.min(width, maxX + horizontalPadding);
        const bottom = Math.min(height, maxY + verticalPadding);

        if (right - left < 40 || bottom - top < 40) {
            return null;
        }

        return {
            left,
            top,
            width: right - left,
            height: bottom - top,
        };
    };

    const applySharpenKernel = (sourceCanvas) => {
        const canvas = cloneCanvas(sourceCanvas);
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const { width, height } = canvas;
        const source = ctx.getImageData(0, 0, width, height);
        const output = ctx.createImageData(width, height);
        const kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];

        for (let y = 1; y < height - 1; y += 1) {
            for (let x = 1; x < width - 1; x += 1) {
                let red = 0;
                let green = 0;
                let blue = 0;
                let alpha = 0;
                let kernelIndex = 0;
                for (let ky = -1; ky <= 1; ky += 1) {
                    for (let kx = -1; kx <= 1; kx += 1) {
                        const offset = ((y + ky) * width + (x + kx)) * 4;
                        const weight = kernel[kernelIndex];
                        red += source.data[offset] * weight;
                        green += source.data[offset + 1] * weight;
                        blue += source.data[offset + 2] * weight;
                        alpha += source.data[offset + 3] * weight;
                        kernelIndex += 1;
                    }
                }
                const target = (y * width + x) * 4;
                output.data[target] = Math.max(0, Math.min(255, red));
                output.data[target + 1] = Math.max(0, Math.min(255, green));
                output.data[target + 2] = Math.max(0, Math.min(255, blue));
                output.data[target + 3] = Math.max(0, Math.min(255, alpha || 255));
            }
        }

        ctx.putImageData(output, 0, 0);
        return canvas;
    };

    const normalizeCanvasForOcr = (sourceCanvas, variantKind) => {
        let canvas = cloneCanvas(sourceCanvas);
        if (variantKind === "sharpened") {
            canvas = applySharpenKernel(canvas);
        }
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        for (let index = 0; index < data.length; index += 4) {
            const gray = 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
            let value = gray;
            if (variantKind === "high-contrast") {
                value = gray < 185 ? Math.max(0, gray * 0.42) : 255;
            } else if (variantKind === "binary") {
                value = gray > 168 ? 255 : 0;
            } else if (variantKind === "binary-soft") {
                value = gray > 182 ? 255 : gray < 138 ? 0 : 110;
            } else if (variantKind === "top-focused") {
                value = gray < 180 ? Math.max(0, gray * 0.55) : 255;
            } else if (variantKind === "sharpened") {
                value = gray < 188 ? Math.max(0, gray * 0.48) : 255;
            }
            data[index] = value;
            data[index + 1] = value;
            data[index + 2] = value;
            data[index + 3] = 255;
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas;
    };

    const createCanvasVariant = (image, kind) => {
        const scaleMap = {
            original: 2.4,
            "high-contrast": 3.2,
            binary: 3.4,
            "binary-soft": 3.2,
            "top-focused": 3.2,
            sharpened: 3.0,
        };
        let canvas = buildBaseCanvasFromImage(image, scaleMap[kind] || 2.8);

        if (kind === "top-focused") {
            canvas = cropCanvas(canvas, {
                left: 0,
                top: 0,
                width: canvas.width,
                height: Math.max(1, Math.round(canvas.height * 0.72)),
            });
        }

        canvas = normalizeCanvasForOcr(canvas, kind);
        const textBounds = detectTextBounds(canvas);
        if (textBounds) {
            canvas = cropCanvas(canvas, textBounds);
        }
        return canvas.toDataURL("image/png");
    };

    const scoreCleanupResult = (ocr) => {
        const candidate = (ocr.suggested_merged_text || ocr.suggested_text || ocr.candidate_text || ocr.raw_text || "").trim();
        const weirdPenalty = ((candidate.match(/[^A-Za-z0-9,.:;%()\-\s]/g) || []).length) * 12;
        const commaBonus = (candidate.match(/,/g) || []).length * 7;
        const ingredientBonus = /ingredients?/i.test(candidate) ? 40 : 0;
        const pantryBonus = ((candidate.match(/\b(water|salt|sugar|vinegar|oil|flavor|mustard|citric|spices|onion|garlic|gum|wheat|corn|oats)\b/gi) || []).length) * 6;
        const statusBonus = ocr.status === "ready" ? 45 : 0;
        const repairPenalty = Math.round(Number(ocr.repair_ratio || 0) * 55);
        const shortTokenPenalty = ((candidate.match(/\b[A-Z]{1,2}\b/g) || []).length) * 10;
        const lengthScore = Math.min(candidate.length, 280);
        return statusBonus + ingredientBonus + pantryBonus + commaBonus + lengthScore - weirdPenalty - repairPenalty - shortTokenPenalty;
    };

    const cleanupOcrText = async (text, confidence, variantName) => {
        const response = await fetch("/api/ocr-cleanup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, confidence }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.message || "OCR cleanup failed.");
        }
        return { ...payload.ocr, variantName, variantScore: scoreCleanupResult(payload.ocr), tesseractConfidence: confidence };
    };

    const runVariantOcr = async (variant) => {
        const result = await window.Tesseract.recognize(variant.image, "eng", {
            logger: (message) => {
                if (!photoUploadHelperText || message.status !== "recognizing text") return;
                const percent = typeof message.progress === "number" ? `${Math.round(message.progress * 100)}%` : "";
                photoUploadHelperText.textContent = `Browser OCR is scanning ${variant.name}${percent ? ` (${percent})` : ""}.`;
            },
            tessedit_pageseg_mode: variant.pageSegMode,
            preserve_interword_spaces: "1",
            tessedit_char_whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,:;%().-/&' ",
        });
        return cleanupOcrText(result?.data?.text || "", result?.data?.confidence ?? null, variant.name);
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
            const baseImages = [uploadedLabelPreview];
            if (uploadedLabelPreview.dataset.originalSrc && uploadedLabelPreview.dataset.originalSrc !== uploadedLabelPreview.src) {
                const originalImage = new Image();
                originalImage.src = uploadedLabelPreview.dataset.originalSrc;
                await originalImage.decode();
                baseImages.push(originalImage);
            }

            const variants = [];
            baseImages.forEach((imageSource, index) => {
                const sourceLabel = index === 0 ? "ocr crop" : "raw upload";
                variants.push(
                    { name: `${sourceLabel} original`, image: createCanvasVariant(imageSource, "original"), pageSegMode: window.Tesseract?.PSM?.SINGLE_BLOCK },
                    { name: `${sourceLabel} high contrast`, image: createCanvasVariant(imageSource, "high-contrast"), pageSegMode: window.Tesseract?.PSM?.SINGLE_BLOCK },
                    { name: `${sourceLabel} sharpened`, image: createCanvasVariant(imageSource, "sharpened"), pageSegMode: window.Tesseract?.PSM?.SINGLE_BLOCK },
                    { name: `${sourceLabel} binary`, image: createCanvasVariant(imageSource, "binary"), pageSegMode: window.Tesseract?.PSM?.SINGLE_BLOCK },
                    { name: `${sourceLabel} soft binary`, image: createCanvasVariant(imageSource, "binary-soft"), pageSegMode: window.Tesseract?.PSM?.SINGLE_BLOCK },
                    { name: `${sourceLabel} top block`, image: createCanvasVariant(imageSource, "top-focused"), pageSegMode: window.Tesseract?.PSM?.SINGLE_COLUMN },
                    { name: `${sourceLabel} sparse text`, image: createCanvasVariant(imageSource, "high-contrast"), pageSegMode: window.Tesseract?.PSM?.SPARSE_TEXT },
                );
            });

            let bestResult = null;
            for (const variant of variants) {
                const current = await runVariantOcr(variant);
                if (!bestResult || current.variantScore > bestResult.variantScore) {
                    bestResult = current;
                }
            }

            updateBrowserOcrUi(bestResult, bestResult.tesseractConfidence ?? null, bestResult.variantName || "");
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
