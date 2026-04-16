document.addEventListener("DOMContentLoaded", () => {
    const queryInput = document.getElementById("query-text");
    const sampleQueryButtons = document.querySelectorAll("[data-sample-query]");

    sampleQueryButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (!queryInput) return;
            queryInput.value = button.dataset.sampleQuery || "";
            queryInput.focus();
        });
    });
});
