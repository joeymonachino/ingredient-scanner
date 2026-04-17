document.addEventListener("DOMContentLoaded", () => {
    const queryInput = document.getElementById("query-text");
    const sampleQueryButtons = document.querySelectorAll("[data-sample-query]");
    const photoInput = document.getElementById("label-photo");
    const photoStatus = document.getElementById("photo-upload-status");
    const cropTool = document.getElementById("crop-tool");
    const cropPreviewImage = document.getElementById("crop-preview-image");
    const cropOverlay = document.getElementById("crop-overlay");
    const cropReset = document.getElementById("crop-reset");
    const ocrQueryText = document.getElementById("ocr-query-text");
    const ocrSuggestedText = document.getElementById("ocr-suggested-text");
    const useOcrSuggestion = document.getElementById("use-ocr-suggestion");
    const ocrMergedSuggestedText = document.getElementById("ocr-merged-suggested-text");
    const useOcrMergedSuggestion = document.getElementById("use-ocr-merged-suggestion");
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
                : "Use a clear photo of the ingredient panel. OCR comes next.";
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

    updateCropOverlay();
});
