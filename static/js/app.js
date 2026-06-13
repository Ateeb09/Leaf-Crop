(function () {
  "use strict";

  function byId(id) {
    return document.getElementById(id);
  }

  function showTab(tabId) {
    document.querySelectorAll(".panel").forEach(function (p) {
      p.classList.toggle("active", p.id === "tab-" + tabId);
    });
    document.querySelectorAll(".nav-btn").forEach(function (b) {
      b.classList.toggle("active", b.dataset.tab === tabId);
    });
  }

  document.querySelectorAll(".nav-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      showTab(btn.dataset.tab);
    });
  });

  function escapeHtml(s) {
    if (s == null) return "";
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // Load crops
  fetch("/api/crops")
    .then(function (r) { return r.json(); })
    .then(function (crops) {
      var sel = byId("crop");
      sel.innerHTML = "";
      (crops || []).forEach(function (c) {
        var opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      });
    })
    .catch(function () {
      byId("crop").innerHTML = "<option value=\"Tomato\">Tomato</option>";
    });

  // Advisory form
  byId("form-advisory").addEventListener("submit", function (e) {
    e.preventDefault();
    var btn = byId("btn-advisory");
    var city = (byId("city").value || "").trim() || "Bengaluru";
    var crop = byId("crop").value || "Tomato";
    var demo = byId("demo").checked;
    var fileInput = byId("leaf-file");
    var formData = new FormData();
    formData.set("city", city);
    formData.set("crop", crop);
    formData.set("demo", demo ? "true" : "false");
    if (fileInput.files && fileInput.files[0]) {
      formData.set("image", fileInput.files[0]);
    }
    btn.disabled = true;
    btn.innerHTML = "<span class=\"spinner\"></span> Running...";
    fetch("/api/advisory", { method: "POST", body: formData })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.statusText); });
        return r.json();
      })
      .then(function (data) {
        renderAdvisory(data);
        byId("advisory-results").classList.remove("hidden");
      })
      .catch(function (err) {
        byId("advisory-content").innerHTML = "<p class=\"error-msg\">" + escapeHtml(err.message) + "</p>";
        byId("advisory-results").classList.remove("hidden");
      })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "Get full advisory";
      });
  });

  byId("btn-clear").addEventListener("click", function () {
    byId("advisory-results").classList.add("hidden");
    byId("advisory-content").innerHTML = "";
  });

  function renderAdvisory(data) {
    var el = byId("advisory-content");
    if (data.error) {
      el.innerHTML = "<p class=\"error-msg\">" + escapeHtml(data.error) + "</p>";
      return;
    }
    var html = "";
    html += "<div class=\"block\"><h4>Weather</h4><p>" + escapeHtml(data.city) + ": ";
    if (data.weather) {
      html += "Temp " + data.weather.temp + " C, Humidity " + data.weather.humidity + "%, Rainfall " + (data.weather.rainfall || 0) + " mm</p></div>";
    } else {
      html += "-</p></div>";
    }
    html += "<div class=\"block\"><h4>Disease risk</h4><p>" + (data.risk != null ? Math.round(data.risk * 100) + "%" : "-") + "</p></div>";
    if (data.field && data.field.soil_type != null) {
      html += "<div class=\"block\"><h4>Field</h4><p>Soil: " + escapeHtml(String(data.field.soil_type)) + ", Moisture: " + (data.field.soil_moisture != null ? Number(data.field.soil_moisture).toFixed(2) : "-") + "</p></div>";
    }
    if (data.leaf) {
      if (data.leaf.error) {
        html += "<div class=\"block\"><h4>Leaf</h4><p class=\"error-msg\">" + escapeHtml(data.leaf.error) + "</p></div>";
      } else if (data.leaf.non_leaf) {
        html += "<div class=\"block\"><h4>Leaf</h4><p>Image was not identified as a leaf.</p></div>";
      } else {
        html += "<div class=\"block\"><h4>Leaf</h4><p>Detected: " + escapeHtml(data.leaf.label || "") + " (" + (data.leaf.confidence != null ? Math.round(data.leaf.confidence * 100) + "%" : "") + ")</p></div>";
      }
    }
    if (data.verdict) {
      html += "<div class=\"block\"><h4>Verdict</h4><p class=\"verdict " + escapeHtml(data.verdict.level || "") + "\">" + escapeHtml(data.verdict.text || "") + "</p></div>";
    }
    if (data.fertilizer) {
      html += "<div class=\"block\"><h4>Fertilizer</h4><p>" + escapeHtml(data.fertilizer) + "</p></div>";
    }
    if (data.actions && data.actions.length) {
      html += "<div class=\"block\"><h4>Recommended actions</h4><ul class=\"actions-list\">";
      data.actions.forEach(function (a) {
        html += "<li>" + escapeHtml(a) + "</li>";
      });
      html += "</ul></div>";
    }
    el.innerHTML = html || "<p>No results.</p>";
  }

  // Outlook form
  byId("form-outlook").addEventListener("submit", function (e) {
    e.preventDefault();
    var city = (byId("outlook-city").value || "").trim() || "Bengaluru";
    fetch("/api/outlook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city: city })
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || r.statusText); });
        return r.json();
      })
      .then(function (data) {
        var o = data.outlook || {};
        var html = "<p><strong>Weather summary</strong> (" + escapeHtml(o.city || "") + "): Avg temp " + (o.avg_temp != null ? o.avg_temp.toFixed(1) + " C" : "-") + ", Total rain " + (o.total_rain != null ? o.total_rain.toFixed(1) + " mm" : "-") + ", Avg risk " + (o.avg_risk != null ? Math.round(o.avg_risk * 100) + "%" : "-") + ".</p>";
        if (data.can_plant && data.can_plant.length) {
          html += "<h4>Crops you can consider planting</h4><div class=\"tags\">";
          data.can_plant.forEach(function (c) {
            html += "<span class=\"tag\">" + escapeHtml(c) + "</span>";
          });
          html += "</div>";
        }
        if (data.avoid && data.avoid.length) {
          html += "<h4 style=\"margin-top:1rem\">Crops to avoid</h4>";
          data.avoid.forEach(function (x) {
            html += "<div class=\"avoid-item\"><strong>" + escapeHtml(x.crop) + "</strong>: " + (x.reasons && x.reasons.length ? x.reasons.join("; ") : "") + "</div>";
          });
        }
        if (data.actions && data.actions.length) {
          html += "<h4 style=\"margin-top:1rem\">What you should do</h4><ul class=\"actions-list\">";
          data.actions.forEach(function (a) {
            html += "<li>" + escapeHtml(a) + "</li>";
          });
          html += "</ul>";
        }
        byId("outlook-content").innerHTML = html;
        byId("outlook-results").classList.remove("hidden");
      })
      .catch(function (err) {
        byId("outlook-content").innerHTML = "<p class=\"error-msg\">" + escapeHtml(err.message) + "</p>";
        byId("outlook-results").classList.remove("hidden");
      });
  });

  // Historical chart (simple canvas)
  var chartCtx = null;

  function drawChart(dates, temp, rainfall, risk) {
    var canvas = byId("chart");
    if (!canvas || !dates || !dates.length) return;
    chartCtx = chartCtx || canvas.getContext("2d");
    var w = canvas.width = canvas.offsetWidth;
    var h = canvas.height = canvas.offsetHeight;
    var padding = { top: 20, right: 20, bottom: 30, left: 50 };
    var graphW = w - padding.left - padding.right;
    var graphH = h - padding.top - padding.bottom;
    chartCtx.clearRect(0, 0, w, h);

    var n = dates.length;
    var maxT = Math.max.apply(null, temp.filter(Boolean)) || 1;
    var maxR = Math.max.apply(null, rainfall.filter(Boolean)) || 1;
    var maxRisk = Math.max.apply(null, (risk || []).filter(Boolean).map(function (v) { return v * 100; })) || 1;

    function x(i) { return padding.left + (i / (n - 1 || 1)) * graphW; }
    function yT(v) { return padding.top + graphH - (Number(v) / maxT) * graphH; }
    function yR(v) { return padding.top + graphH - (Number(v) / maxR) * graphH; }
    function yRisk(v) { return padding.top + graphH - ((v * 100) / maxRisk) * graphH; }

    chartCtx.strokeStyle = "#16a34a";
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    for (var i = 0; i < n; i++) {
      if (temp[i] == null) continue;
      if (i === 0) chartCtx.moveTo(x(i), yT(temp[i]));
      else chartCtx.lineTo(x(i), yT(temp[i]));
    }
    chartCtx.stroke();

    chartCtx.strokeStyle = "#2563eb";
    chartCtx.beginPath();
    for (var j = 0; j < n; j++) {
      if (rainfall[j] == null) continue;
      if (j === 0) chartCtx.moveTo(x(j), yR(rainfall[j]));
      else chartCtx.lineTo(x(j), yR(rainfall[j]));
    }
    chartCtx.stroke();

    if (risk && risk.length) {
      chartCtx.strokeStyle = "#dc2626";
      chartCtx.beginPath();
      for (var k = 0; k < n; k++) {
        var rv = risk[k] != null ? risk[k] * 100 : null;
        if (rv == null) continue;
        if (k === 0) chartCtx.moveTo(x(k), yRisk(risk[k]));
        else chartCtx.lineTo(x(k), yRisk(risk[k]));
      }
      chartCtx.stroke();
    }

    chartCtx.fillStyle = "#6b7280";
    chartCtx.font = "11px system-ui";
    chartCtx.fillText("Temp (green) | Rainfall (blue) | Risk % (red)", padding.left, h - 8);
  }

  function loadHistorical() {
    var days = parseInt(byId("historical-days").value, 10) || 90;
    fetch("/api/historical?days=" + days)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var dates = data.dates || [];
        drawChart(dates, data.temperature || [], data.rainfall || [], data.risk || []);
      })
      .catch(function () {
        var canvas = byId("chart");
        if (canvas && canvas.getContext) {
          var ctx = canvas.getContext("2d");
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#6b7280";
          ctx.font = "14px system-ui";
          ctx.fillText("Could not load historical data.", 20, 40);
        }
      });
  }

  byId("btn-historical").addEventListener("click", loadHistorical);
  byId("historical-days").addEventListener("change", function () {
    if (document.getElementById("tab-historical").classList.contains("active")) {
      loadHistorical();
    }
  });
})();
