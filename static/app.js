document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("ingredient");
    const sampleButtons = document.querySelectorAll("[data-sample]");

    sampleButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (!input) return;
            input.value = button.dataset.sample || "";
            input.focus();
        });
    });
});
