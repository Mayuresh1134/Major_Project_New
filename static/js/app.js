/* Shared behaviour: theme, menu, file pickers, sliders, tabs, and running jobs with progress. */
(function () {
  "use strict";

  // Theme ---------------------------------------------------------------
  const root = document.documentElement;
  const themeBtn = document.getElementById("theme-toggle");
  function paintThemeButton() {
    if (!themeBtn) return;
    const dark = root.dataset.theme === "dark";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    themeBtn.innerHTML = dark
      ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></svg>'
      : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>';
  }
  paintThemeButton();
  if (themeBtn) themeBtn.addEventListener("click", () => {
    const dark = root.dataset.theme !== "dark";
    if (dark) root.dataset.theme = "dark"; else delete root.dataset.theme;
    try { localStorage.setItem("muse-theme", dark ? "dark" : "light"); } catch (e) {}
    paintThemeButton();
    document.dispatchEvent(new CustomEvent("themechange"));
  });

  // Mobile menu -----------------------------------------------------------
  const menuBtn = document.querySelector(".menu-toggle");
  const nav = document.getElementById("site-nav");
  if (menuBtn && nav) menuBtn.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    menuBtn.setAttribute("aria-expanded", String(open));
  });

  // File pickers ----------------------------------------------------------
  document.querySelectorAll("[data-dropzone]").forEach((zone) => {
    const input = zone.querySelector("input[type=file]");
    const label = zone.querySelector("[data-filename]");
    const show = () => { label.textContent = input.files.length ? input.files[0].name : ""; };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach((e) => zone.addEventListener(e, () => zone.classList.add("drag")));
    ["dragleave", "drop"].forEach((e) => zone.addEventListener(e, () => zone.classList.remove("drag")));
    input.addEventListener("drop", () => setTimeout(show, 0));
  });

  // Range outputs -----------------------------------------------------------
  document.querySelectorAll("input[type=range][data-output]").forEach((r) => {
    const out = document.getElementById(r.dataset.output);
    const fmt = r.dataset.format;
    const update = () => { out.textContent = (fmt === "signed" && r.value > 0 ? "+" : "") + r.value + (r.dataset.suffix || ""); };
    r.addEventListener("input", update);
    update();
  });

  // Tabs ------------------------------------------------------------------------
  document.querySelectorAll("[data-tabs]").forEach((group) => {
    const buttons = group.querySelectorAll("[role=tab]");
    const select = (btn, focus) => {
      buttons.forEach((b) => {
        const on = b === btn;
        b.setAttribute("aria-selected", String(on));
        b.tabIndex = on ? 0 : -1;
        document.getElementById(b.getAttribute("aria-controls")).hidden = !on;
      });
      if (focus) btn.focus();
      const field = group.dataset.field && document.querySelector(`[name="${group.dataset.field}"]`);
      if (field) field.value = btn.dataset.value;
      // only the visible panel's file inputs are required
      group.parentElement.querySelectorAll(".tab-panel").forEach((p) => {
        p.querySelectorAll("[data-required]").forEach((el) => { el.required = !p.hidden; });
      });
    };
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => select(b));
      b.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          const next = buttons[(i + (e.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length];
          select(next, true);
        }
      });
    });
    const initial = group.querySelector("[aria-selected=true]") || buttons[0];
    select(initial);
  });

  // Toast -----------------------------------------------------------------------
  window.museToast = function (text) {
    const t = document.createElement("div");
    t.className = "toast";
    t.setAttribute("role", "status");
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3200);
  };

  // Jobs ----------------------------------------------------------------------------
  const overlay = document.getElementById("progress-overlay");
  const stageEl = document.getElementById("progress-stage");
  const bar = overlay && overlay.querySelector(".progress");
  function showProgress(fraction, stage) {
    overlay.hidden = false;
    if (stage) stageEl.textContent = stage;
    const pct = Math.round(100 * (fraction || 0));
    bar.firstElementChild.style.width = pct + "%";
    bar.setAttribute("aria-valuenow", String(pct));
  }

  async function poll(statusUrl, resultUrl, onError) {
    for (;;) {
      await new Promise((r) => setTimeout(r, 700));
      let job;
      try {
        const res = await fetch(statusUrl, { cache: "no-store" });
        job = await res.json();
      } catch (e) { continue; }
      if (job.status === "done") { showProgress(1, "Opening the result"); window.location.href = resultUrl; return; }
      if (job.status === "error") { overlay.hidden = true; onError(job.error || "The job failed."); return; }
      showProgress(job.progress, job.stage);
    }
  }

  function uploadWithProgress(form) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", form.action);
      xhr.responseType = "json";
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable && e.total > 200000) showProgress(0, `Uploading ${Math.round((100 * e.loaded) / e.total)}%`);
      });
      xhr.addEventListener("load", () => resolve({ status: xhr.status, body: xhr.response || {} }));
      xhr.addEventListener("error", () => reject(new Error("The upload failed. Check that MuseAssist is still running.")));
      xhr.send(new FormData(form));
    });
  }

  document.querySelectorAll("form[data-job-form]").forEach((form) => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const errorEl = form.querySelector(".form-error") || (() => {
        const p = document.createElement("p"); p.className = "form-error"; p.setAttribute("role", "alert"); form.appendChild(p); return p;
      })();
      errorEl.textContent = "";
      const button = form.querySelector("button[type=submit]");
      if (button) button.disabled = true;
      const fail = (msg) => { errorEl.textContent = msg; if (button) button.disabled = false; };
      showProgress(0, "Uploading");
      try {
        const { status, body } = await uploadWithProgress(form);
        if (status !== 202) { overlay.hidden = true; fail(body.error || `The server answered with an error (${status}).`); return; }
        showProgress(0.01, "Starting");
        poll(body.status_url, body.result_url, fail);
      } catch (err) {
        overlay.hidden = true;
        fail(err.message);
      }
    });
  });

  // a result page that is still being made
  const waiting = document.querySelector("[data-waiting]");
  if (waiting) {
    showProgress(0, "Working on it");
    poll(waiting.dataset.status, waiting.dataset.result, (msg) => { waiting.querySelector(".form-error").textContent = msg; });
  }

  // Library ---------------------------------------------------------------------------
  const filters = document.querySelector("[data-lib-filters]");
  if (filters) {
    filters.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      filters.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      document.querySelectorAll("[data-kind]").forEach((row) => {
        row.hidden = b.dataset.filter !== "all" && row.dataset.kind !== b.dataset.filter;
      });
    });
  }
  document.querySelectorAll("[data-delete]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (btn.dataset.armed !== "1") {  // ask once, inline, instead of a blocking dialog
        btn.dataset.armed = "1";
        const label = btn.textContent;
        btn.textContent = "Click again to delete";
        setTimeout(() => { btn.dataset.armed = ""; btn.textContent = label; }, 4000);
        return;
      }
      const res = await fetch(btn.dataset.delete, { method: "POST" });
      if (res.ok) {
        const row = btn.closest("tr");
        if (row) row.remove();
        window.museToast("Deleted");
        if (btn.dataset.after) window.location.href = btn.dataset.after;
      } else {
        window.museToast("Could not delete this result");
      }
    });
  });
})();
