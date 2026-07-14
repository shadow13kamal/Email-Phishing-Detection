/* Phishing Email Detection System — Custom JavaScript */

document.addEventListener("DOMContentLoaded", function () {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll(".alert");
    alerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 5000);
    });

    // Animate metric bars on the metrics page
    const metricBars = document.querySelectorAll(".metric-bar-fill");
    if (metricBars.length > 0) {
        metricBars.forEach(function (bar) {
            const targetWidth = bar.style.width;
            bar.style.width = "0%";
            setTimeout(function () {
                bar.style.width = targetWidth;
            }, 100);
        });
    }
});
