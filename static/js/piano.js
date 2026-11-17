/* Web Audio piano voice and an on-screen keyboard, shared by Home and Practice. */
(function () {
  "use strict";
  const NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const noteName = (m) => NAMES[m % 12] + (Math.floor(m / 12) - 1);
  const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);

  let ctx = null, master = null;
  function audio() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
      master = ctx.createDynamicsCompressor();
      master.threshold.value = -14;
      master.connect(ctx.destination);
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  /** Play a note; returns a function that releases it. */
  function play(midi, opts = {}) {
    const ac = audio();
    if (!ac) return () => {};
    const when = opts.when ?? ac.currentTime;
    const vel = opts.velocity ?? 0.8;
    const out = ac.createGain();
    const filter = ac.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(Math.min(hz(midi) * 6 + 800, 12000), when);
    filter.frequency.exponentialRampToValueAtTime(Math.min(hz(midi) * 2.5 + 400, 8000), when + 1.2);
    filter.connect(out);
    out.connect(master);
    const f = hz(midi);
    const oscs = [["triangle", 1, 0.55], ["sine", 2, 0.22], ["sine", 3, 0.08]].map(([type, mult, g]) => {
      const o = ac.createOscillator();
      const og = ac.createGain();
      o.type = type;
      o.frequency.value = f * mult;
      og.gain.value = g;
      o.connect(og).connect(filter);
      o.start(when);
      return o;
    });
    const decay = 1.6 * Math.pow(2, -(midi - 60) / 24);
    out.gain.setValueAtTime(0.0001, when);
    out.gain.exponentialRampToValueAtTime(0.32 * vel, when + 0.006);
    out.gain.exponentialRampToValueAtTime(0.12 * vel, when + 0.006 + decay * 0.5);
    out.gain.exponentialRampToValueAtTime(0.0001, when + 0.006 + decay * 4);
    let released = false;
    const release = (at) => {
      if (released) return;
      released = true;
      const t = Math.max(at ?? ac.currentTime, when + 0.02);
      out.gain.cancelScheduledValues(t);
      out.gain.setTargetAtTime(0.0001, t, 0.08);
      oscs.forEach((o) => o.stop(t + 0.6));
    };
    if (opts.duration) release(when + opts.duration);
    else oscs.forEach((o) => o.stop(when + 0.006 + decay * 4 + 0.1));
    return release;
  }

  function click(when, accent) {
    const ac = audio();
    if (!ac) return;
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.frequency.value = accent ? 1760 : 1175;
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(accent ? 0.6 : 0.35, when + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, when + 0.05);
    o.connect(g).connect(master);
    o.start(when);
    o.stop(when + 0.06);
  }

  const KEYMAP = "awsedftgyhujkolp;'";  // C to F a little over an octave up

  class Keyboard {
    constructor(el, opts = {}) {
      this.el = el;
      this.low = opts.low ?? 48;
      this.high = opts.high ?? 72;
      this.labels = opts.labels ?? "c";
      this.onNote = opts.onNote || (() => {});
      this.base = opts.keyboardBase ?? 60;
      this.held = new Map();
      this.render();
      if (opts.computerKeys) this.bindKeys();
    }
    render() {
      this.el.innerHTML = "";
      this.el.classList.add("keyboard");
      this.el.setAttribute("role", "group");
      const whites = [];
      for (let m = this.low; m <= this.high; m++) if (![1, 3, 6, 8, 10].includes(m % 12)) whites.push(m);
      const w = 100 / whites.length;
      this.keys = new Map();
      for (let m = this.low; m <= this.high; m++) {
        const black = [1, 3, 6, 8, 10].includes(m % 12);
        const k = document.createElement("div");
        k.className = "key " + (black ? "black" : "white");
        k.dataset.midi = m;
        k.setAttribute("aria-label", noteName(m));
        if (!black && (this.labels === "all" || (this.labels === "c" && m % 12 === 0))) k.textContent = noteName(m);
        if (black) {
          const idx = whites.indexOf(m - 1);
          k.style.left = `calc(${(idx + 1) * w}% - ${w * 0.32}%)`;
          k.style.width = `${w * 0.64}%`;
        }
        this.el.appendChild(k);
        this.keys.set(m, k);
      }
      let pointerNote = null;
      this.el.addEventListener("pointerdown", (e) => {
        const k = e.target.closest(".key");
        if (!k) return;
        this.el.setPointerCapture(e.pointerId);
        pointerNote = +k.dataset.midi;
        this.down(pointerNote);
      });
      this.el.addEventListener("pointermove", (e) => {
        if (pointerNote === null) return;
        const k = document.elementFromPoint(e.clientX, e.clientY)?.closest?.(".key");
        if (k && +k.dataset.midi !== pointerNote) {
          this.up(pointerNote);
          pointerNote = +k.dataset.midi;
          this.down(pointerNote);
        }
      });
      const end = () => { if (pointerNote !== null) this.up(pointerNote); pointerNote = null; };
      this.el.addEventListener("pointerup", end);
      this.el.addEventListener("pointercancel", end);
    }
    down(m) {
      if (this.held.has(m)) return;
      this.held.set(m, play(m));
      this.keys.get(m)?.classList.add("down");
      this.onNote(m, true);
    }
    up(m) {
      const rel = this.held.get(m);
      if (rel) rel();
      this.held.delete(m);
      this.keys.get(m)?.classList.remove("down");
      this.onNote(m, false);
    }
    flash(m, ms = 350) {
      const k = this.keys.get(m);
      if (!k) return;
      k.classList.add("down");
      setTimeout(() => k.classList.remove("down"), ms);
    }
    light(midis, root, cls = "lit") {
      this.clear(cls);
      midis.forEach((m) => {
        const k = this.keys.get(m);
        if (!k) return;
        k.classList.add(cls);
        if (m === root) k.classList.add("root");
      });
    }
    clear(cls = "lit") { this.keys.forEach((k) => k.classList.remove(cls, "root")); }
    bindKeys() {
      const isTyping = (e) => /input|textarea|select/i.test(e.target.tagName) || e.target.isContentEditable;
      window.addEventListener("keydown", (e) => {
        if (e.repeat || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
        const k = e.key.toLowerCase();
        if (k === "z" || k === "x") {
          this.base = Math.min(Math.max(this.base + (k === "z" ? -12 : 12), this.low), this.high - 12);
          window.museToast && window.museToast(`Keys now start at ${noteName(this.base)}`);
          return;
        }
        const i = KEYMAP.indexOf(k);
        if (i < 0) return;
        const m = this.base + i;
        if (m < this.low || m > this.high) return;
        e.preventDefault();
        this.down(m);
      });
      window.addEventListener("keyup", (e) => {
        const i = KEYMAP.indexOf(e.key.toLowerCase());
        if (i >= 0) this.up(this.base + i);
      });
      window.addEventListener("blur", () => [...this.held.keys()].forEach((m) => this.up(m)));
    }
  }

  window.Muse = Object.assign(window.Muse || {}, { audio, play, click, Keyboard, noteName, hz, NAMES });
})();
