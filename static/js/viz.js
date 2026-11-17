/* Piano roll and waveform views, both following an <audio> element. */
(function () {
  "use strict";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const BLACK = [1, 3, 6, 8, 10];

  function sizeCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return g;
  }

  function follow(audio, draw) {
    let raf = null;
    const loop = () => { draw(); raf = audio.paused ? null : requestAnimationFrame(loop); };
    ["play", "seeked", "timeupdate"].forEach((e) => audio.addEventListener(e, () => { if (!raf) loop(); }));
  }

  class PianoRoll {
    constructor(box, notes, opts = {}) {
      this.box = box;
      this.notes = notes;
      // follows whichever of its audio elements was played last
      this.audios = (opts.audios || (opts.audio ? [opts.audio] : [])).filter(Boolean);
      this.audio = this.audios[0] || null;
      this.offset = opts.offset || 0;  // audio time = note time + offset
      this.marker = opts.marker ?? null;
      this.markerLabel = opts.markerLabel || "";
      this.height = opts.height || 260;
      this.duration = Math.max(1, ...notes.map((n) => n[2])) + 0.5;
      const pitches = notes.map((n) => n[0]);
      this.lo = Math.min(...pitches) - 2;
      this.hi = Math.max(...pitches) + 2;
      if (this.hi - this.lo < 14) { const mid = (this.hi + this.lo) / 2; this.lo = Math.floor(mid - 7); this.hi = Math.ceil(mid + 7); }
      this.canvas = document.createElement("canvas");
      box.appendChild(this.canvas);
      this.fit();
      box.addEventListener("scroll", () => this.draw());
      this.canvas.addEventListener("click", (e) => {
        if (!this.audio) return;
        const x = e.offsetX;
        this.audio.currentTime = Math.max(0, x / this.pps + this.offset);
        this.draw();
      });
      this.audios.forEach((a) => {
        a.addEventListener("play", () => {
          this.audio = a;
          this.audios.forEach((o) => { if (o !== a) o.pause(); });
        });
        follow(a, () => { if (a === this.audio) this.draw(true); });
      });
      window.addEventListener("resize", () => this.fit());
      document.addEventListener("themechange", () => this.draw());
    }
    fit() {
      const avail = this.box.clientWidth || 600;
      this.minPps = avail / this.duration;
      // fit the whole piece when it is short; scroll at a readable 14 px per second when long
      if (!this.ppsSet) this.pps = Math.max(this.minPps, 14);
      this.pps = Math.min(this.pps, 30000 / this.duration);
      this.width = Math.max(avail, this.duration * this.pps);
      this.g = sizeCanvas(this.canvas, this.width, this.height);
      this.draw();
    }
    zoom(f) {
      this.ppsSet = true;
      const t = (this.box.scrollLeft + this.box.clientWidth / 2) / this.pps;
      this.pps = Math.max(this.minPps, Math.min(this.pps * f, 400));
      this.fit();
      this.box.scrollLeft = t * this.pps - this.box.clientWidth / 2;
    }
    draw(followHead) {
      const g = this.g, W = this.width, H = this.height;
      const range = this.hi - this.lo + 1;
      const rowH = H / range;
      const y = (p) => H - (p - this.lo + 1) * rowH;
      g.clearRect(0, 0, W, H);
      g.fillStyle = css("--sunk");
      g.fillRect(0, 0, W, H);
      g.fillStyle = css("--line");
      for (let p = this.lo; p <= this.hi; p++) {
        if (BLACK.includes(((p % 12) + 12) % 12)) { g.globalAlpha = 0.35; g.fillRect(0, y(p), W, rowH); g.globalAlpha = 1; }
        if (p % 12 === 0) { g.fillRect(0, y(p) + rowH - 1, W, 1); }
      }
      // seconds grid
      g.fillStyle = css("--line");
      const every = this.pps > 60 ? 1 : this.pps > 20 ? 5 : 10;
      for (let t = 0; t < this.duration; t += every) g.fillRect(t * this.pps, 0, 1, H);
      const now = this.audio ? this.audio.currentTime - this.offset : -1;
      const colors = [css("--seed"), css("--ai")];
      for (const [p, s, e, v, part] of this.notes) {
        const x = s * this.pps, w = Math.max(2, (e - s) * this.pps - 1);
        const playing = now >= s && now < e;
        g.fillStyle = colors[part || 0];
        g.globalAlpha = playing ? 1 : 0.45 + 0.45 * Math.min(1, (v || 90) / 110);
        g.fillRect(x, y(p) + 0.5, w, Math.max(rowH - 1, 2));
        if (playing) { g.globalAlpha = 1; g.strokeStyle = css("--ink"); g.lineWidth = 1; g.strokeRect(x + 0.5, y(p) + 0.5, w - 1, Math.max(rowH - 1, 2)); }
      }
      g.globalAlpha = 1;
      if (this.marker !== null) {
        const mx = this.marker * this.pps;
        g.strokeStyle = css("--ai");
        g.setLineDash([5, 4]);
        g.beginPath(); g.moveTo(mx, 0); g.lineTo(mx, H); g.stroke();
        g.setLineDash([]);
        g.fillStyle = css("--ai");
        g.font = "600 12px " + css("--font");
        g.fillText(this.markerLabel, mx + 6, 14);
      }
      // octave labels at the visible left edge
      g.font = "11px " + css("--font");
      g.fillStyle = css("--muted");
      const left = this.box.scrollLeft;
      for (let p = Math.ceil(this.lo / 12) * 12; p <= this.hi; p += 12) g.fillText("C" + (p / 12 - 1), left + 4, y(p) + rowH - 3);
      if (now >= 0) {
        const px = now * this.pps;
        g.fillStyle = css("--brass-deep");
        g.fillRect(px - 1, 0, 2, H);
        if (followHead && this.audio && !this.audio.paused) {
          const view = this.box.clientWidth;
          if (px < left || px > left + view * 0.85) this.box.scrollLeft = px - view * 0.15;
        }
      }
    }
  }

  class Waveform {
    constructor(box, data, opts = {}) {
      this.box = box;
      this.peaks = data.peaks;
      this.duration = data.duration;
      this.beats = data.beats || [];
      this.chords = data.chords || [];
      this.audio = opts.audio;
      this.chordNow = opts.chordNow || null;
      this.height = 150;
      this.lane = this.chords.length ? 30 : 0;
      this.canvas = document.createElement("canvas");
      box.appendChild(this.canvas);
      this.fit();
      window.addEventListener("resize", () => this.fit());
      document.addEventListener("themechange", () => this.draw());
      this.canvas.addEventListener("click", (e) => {
        if (!this.audio) return;
        this.audio.currentTime = (e.offsetX / this.width) * this.duration;
        this.draw();
      });
      if (this.audio) follow(this.audio, () => this.draw());
    }
    fit() {
      this.width = this.box.clientWidth || 600;
      this.g = sizeCanvas(this.canvas, this.width, this.height + this.lane);
      this.draw();
    }
    draw() {
      const g = this.g, W = this.width, H = this.height, L = this.lane;
      g.clearRect(0, 0, W, H + L);
      const mid = H / 2;
      const now = this.audio ? this.audio.currentTime : 0;
      const n = this.peaks.length;
      const ink = css("--ink"), seed = css("--seed");
      for (let i = 0; i < n; i++) {
        const x = (i / n) * W;
        const [lo, hi] = this.peaks[i];
        g.fillStyle = (i / n) * this.duration <= now ? seed : ink;
        g.globalAlpha = (i / n) * this.duration <= now ? 0.95 : 0.55;
        g.fillRect(x, mid - hi * mid * 0.95, Math.max(W / n, 1), Math.max((hi - lo) * mid * 0.95, 1));
      }
      g.globalAlpha = 1;
      g.fillStyle = css("--brass-deep");
      g.globalAlpha = 0.5;
      for (const b of this.beats) g.fillRect((b / this.duration) * W, H - 8, 1, 8);
      g.globalAlpha = 1;
      if (L) {
        g.font = "600 12px " + css("--font");
        let current = "";
        this.chords.forEach((c, i) => {
          const x0 = (c.start / this.duration) * W, x1 = (c.end / this.duration) * W;
          const active = now >= c.start && now < c.end;
          if (active) current = c.chord;
          g.fillStyle = i % 2 ? css("--sunk") : css("--surface");
          if (active) g.fillStyle = css("--brass");
          g.fillRect(x0, H + 2, x1 - x0, L - 4);
          g.fillStyle = active ? css("--brass-ink") : css("--muted");
          if (x1 - x0 > 26 && c.chord !== "N") g.fillText(c.chord, x0 + 4, H + L / 2 + 4);
        });
        if (this.chordNow) this.chordNow.textContent = current && current !== "N" ? current : "–";
      }
      g.fillStyle = css("--brass-deep");
      g.fillRect((now / this.duration) * W - 1, 0, 2, H + L);
    }
  }

  function spectroHead(img, audio, duration) {
    const head = img.parentElement.querySelector(".ph");
    if (!head) return;
    const move = () => { head.style.left = `${(100 * audio.currentTime) / duration}%`; };
    follow(audio, move);
    img.addEventListener("click", (e) => { audio.currentTime = (e.offsetX / img.clientWidth) * duration; move(); });
  }

  window.Muse = Object.assign(window.Muse || {}, { PianoRoll, Waveform, spectroHead });
})();
