/* Practice page: chord and scale explorer, ear training, tuner, metronome. */
(function () {
  "use strict";
  const { Keyboard, play, click, audio, noteName, NAMES } = window.Muse;
  const $ = (id) => document.getElementById(id);
  const kb = new Keyboard($("kb"), { low: 48, high: 84, labels: "c", computerKeys: true });
  if (window.matchMedia("(max-width: 600px)").matches) { kb.low = 55; kb.high = 76; kb.render(); }

  const ROOTS = NAMES.map((n, i) => [n, i]);
  const CHORDS = [
    ["major", "", "major", [0, 4, 7]],
    ["minor", "m", "minor", [0, 3, 7]],
    ["dim", "dim", "diminished", [0, 3, 6]],
    ["aug", "aug", "augmented", [0, 4, 8]],
    ["sus2", "sus2", "suspended 2nd", [0, 2, 7]],
    ["sus4", "sus4", "suspended 4th", [0, 5, 7]],
    ["7", "7", "dominant 7th", [0, 4, 7, 10]],
    ["maj7", "maj7", "major 7th", [0, 4, 7, 11]],
    ["m7", "m7", "minor 7th", [0, 3, 7, 10]],
    ["m7b5", "m7♭5", "half-diminished 7th", [0, 3, 6, 10]],
    ["dim7", "dim7", "diminished 7th", [0, 3, 6, 9]],
    ["6", "6", "major 6th", [0, 4, 7, 9]],
    ["m6", "m6", "minor 6th", [0, 3, 7, 9]],
    ["9", "9", "dominant 9th", [0, 4, 7, 10, 14]],
    ["add9", "add9", "added 9th", [0, 4, 7, 14]],
  ];
  const INTERVAL_NAMES = ["unison", "minor 2nd", "major 2nd", "minor 3rd", "major 3rd", "perfect 4th", "tritone",
    "perfect 5th", "minor 6th", "major 6th", "minor 7th", "major 7th", "octave", "minor 9th", "9th"];
  const SCALES = [
    ["Major", [2, 2, 1, 2, 2, 2, 1]],
    ["Natural minor", [2, 1, 2, 2, 1, 2, 2]],
    ["Harmonic minor", [2, 1, 2, 2, 1, 3, 1]],
    ["Melodic minor (rising)", [2, 1, 2, 2, 2, 2, 1]],
    ["Major pentatonic", [2, 2, 3, 2, 3]],
    ["Minor pentatonic", [3, 2, 2, 3, 2]],
    ["Blues", [3, 2, 1, 1, 3, 2]],
    ["Dorian", [2, 1, 2, 2, 2, 1, 2]],
    ["Phrygian", [1, 2, 2, 2, 1, 2, 2]],
    ["Lydian", [2, 2, 2, 1, 2, 2, 1]],
    ["Mixolydian", [2, 2, 1, 2, 2, 1, 2]],
    ["Locrian", [1, 2, 2, 1, 2, 2, 2]],
    ["Chromatic", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
  ];
  const fill = (sel, items) => { sel.innerHTML = items.map(([label, v]) => `<option value="${v}">${label}</option>`).join(""); };
  const pc = (m) => NAMES[((m % 12) + 12) % 12];

  function playChord(midis, arpeggio) {
    const ac = audio();
    if (!ac) return;
    const t = ac.currentTime + 0.03;
    midis.forEach((m, i) => {
      const when = arpeggio ? t + i * 0.32 : t + i * 0.012;
      play(m, { when, duration: arpeggio ? 0.9 : 1.6, velocity: 0.75 });
      setTimeout(() => kb.flash(m, arpeggio ? 300 : 900), (when - ac.currentTime) * 1000);
    });
  }

  // Chords ----------------------------------------------------------------
  fill($("chord-root"), ROOTS);
  fill($("chord-quality"), CHORDS.map((c) => [`${c[2][0].toUpperCase()}${c[2].slice(1)}`, c[0]]));
  $("chord-root").value = "0";
  function currentChord() {
    const root = +$("chord-root").value;
    const [, sym, long, ivs] = CHORDS.find((c) => c[0] === $("chord-quality").value);
    let notes = ivs.map((i) => 60 + root + i);
    if (root >= 7) notes = notes.map((m) => m - 12);  // keep it near middle C
    const inv = Math.min(+$("chord-inv").value, notes.length - 1);
    for (let i = 0; i < inv; i++) notes.push(notes.shift() + 12);
    return { root: notes.find((m) => (m - root) % 12 === 0), notes, sym, long, ivs, rootName: NAMES[root] };
  }
  function showChord() {
    const c = currentChord();
    $("chord-name").textContent = `${c.rootName}${c.sym}: ${c.notes.map(pc).join("  ")}`;
    $("chord-detail").textContent = `${c.rootName} ${c.long}. From the root: ${c.ivs.map((i) => (i === 0 ? "root" : INTERVAL_NAMES[i])).join(", ")}.`;
    kb.light(c.notes, c.root);
  }
  ["chord-root", "chord-quality", "chord-inv"].forEach((id) => $(id).addEventListener("change", () => { showChord(); playChord(currentChord().notes, false); }));
  $("chord-play").addEventListener("click", () => playChord(currentChord().notes, false));
  $("chord-arp").addEventListener("click", () => playChord(currentChord().notes, true));

  // Scales ----------------------------------------------------------------
  fill($("scale-root"), ROOTS);
  fill($("scale-type"), SCALES.map((s, i) => [s[0], i]));
  function currentScale() {
    const root = +$("scale-root").value;
    const [name, steps] = SCALES[+$("scale-type").value];
    const start = 60 + root - (root >= 7 ? 12 : 0);
    const notes = [start];
    steps.forEach((s) => notes.push(notes[notes.length - 1] + s));
    return { name, steps, notes, root: start };
  }
  function showScale() {
    const s = currentScale();
    $("scale-notes").textContent = s.notes.slice(0, -1).map(pc).join("  ");
    const word = { 1: "H", 2: "W", 3: "W+H" };
    $("scale-steps").textContent = `${NAMES[s.root % 12]} ${s.name.toLowerCase()}. Steps: ${s.steps.map((x) => word[x] || x).join(" ")} (W = whole tone, H = half tone).`;
    kb.light(s.notes, s.root);
    kb.keys.get(s.notes[s.notes.length - 1])?.classList.add("root");
  }
  ["scale-root", "scale-type"].forEach((id) => $(id).addEventListener("change", showScale));
  $("scale-play").addEventListener("click", () => {
    const s = currentScale();
    const ac = audio();
    if (!ac) return;
    const seq = s.notes.concat(s.notes.slice(0, -1).reverse());
    const t = ac.currentTime + 0.05;
    seq.forEach((m, i) => {
      play(m, { when: t + i * 0.28, duration: 0.3, velocity: 0.7 });
      setTimeout(() => kb.flash(m, 240), (t + i * 0.28 - ac.currentTime) * 1000);
    });
  });
  document.querySelectorAll("#t-chords, #t-scales").forEach((t) => t.addEventListener("click", () => (t.id === "t-chords" ? showChord() : showScale())));
  showChord();

  // Ear training --------------------------------------------------------------
  const EAR = {
    interval: INTERVAL_NAMES.slice(1, 13).map((n, i) => ({ id: "i" + (i + 1), label: n, semis: i + 1, on: i + 1 <= 7 || i + 1 === 12 })),
    chord: CHORDS.filter((c) => ["major", "minor", "dim", "aug", "7", "maj7", "m7"].includes(c[0]))
      .map((c, i) => ({ id: "c" + c[0], label: c[2], ivs: c[3], on: i < 4 })),
  };
  let mode = "interval", question = null, answered = false, right = 0, total = 0, streak = 0;
  const pool = () => EAR[mode].filter((x) => x.on);
  function renderPool() {
    $("ear-pool").innerHTML = EAR[mode].map((x) =>
      `<label class="check"><input type="checkbox" data-id="${x.id}" ${x.on ? "checked" : ""}><span>${x.label}</span></label>`).join("");
    $("ear-pool").querySelectorAll("input").forEach((cb) => cb.addEventListener("change", () => {
      const item = EAR[mode].find((x) => x.id === cb.dataset.id);
      item.on = cb.checked;
      if (pool().length < 2) { item.on = true; cb.checked = true; window.museToast("Keep at least two to choose from"); }
      renderAnswers();
    }));
  }
  function renderAnswers() {
    $("ear-answers").innerHTML = pool().map((x) => `<button type="button" data-id="${x.id}">${x.label}</button>`).join("");
    $("ear-answers").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => answer(b)));
  }
  function notesFor(q) {
    if (mode === "interval") {
      const dir = $("ear-dir").value;
      return dir === "down" ? [q.base + q.item.semis, q.base] : [q.base, q.base + q.item.semis];
    }
    return q.item.ivs.map((i) => q.base + i);
  }
  function playQuestion() {
    const notes = notesFor(question);
    const together = mode === "chord" || $("ear-dir").value === "together";
    const ac = audio();
    if (!ac) return;
    const t = ac.currentTime + 0.05;
    notes.forEach((m, i) => play(m, { when: together ? t : t + i * 0.7, duration: together ? 1.6 : 0.8, velocity: 0.75 }));
  }
  $("ear-new").addEventListener("click", () => {
    const items = pool();
    const item = items[Math.floor(Math.random() * items.length)];
    question = { item, base: 52 + Math.floor(Math.random() * 14) };
    answered = false;
    kb.clear("answer");
    $("ear-feedback").textContent = "What did you hear?";
    $("ear-answers").querySelectorAll("button").forEach((b) => (b.className = ""));
    $("ear-again").disabled = false;
    playQuestion();
  });
  $("ear-again").addEventListener("click", () => question && playQuestion());
  function answer(btn) {
    if (!question) { $("ear-feedback").textContent = "Press “New question” first."; return; }
    if (answered) return;
    answered = true;
    total++;
    const ok = btn.dataset.id === question.item.id;
    if (ok) { right++; streak++; } else streak = 0;
    btn.classList.add(ok ? "right" : "wrong");
    $("ear-answers").querySelector(`[data-id="${question.item.id}"]`).classList.add("right");
    const notes = notesFor(question);
    $("ear-feedback").textContent = ok
      ? `Right: ${question.item.label} (${notes.map(noteName).join(", ")}).`
      : `It was ${question.item.label === "tritone" ? "a tritone" : "a " + question.item.label} (${notes.map(noteName).join(", ")}).`;
    kb.light(notes, null, "answer");
    $("ear-right").textContent = right;
    $("ear-total").textContent = total;
    $("ear-streak").textContent = streak;
  }
  document.querySelectorAll("input[name=ear-mode]").forEach((r) => r.addEventListener("change", () => {
    mode = r.value;
    $("ear-dir-field").hidden = mode !== "interval";
    question = null;
    $("ear-again").disabled = true;
    $("ear-feedback").textContent = "Press “New question” and name what you hear.";
    renderPool();
    renderAnswers();
  }));
  renderPool();
  renderAnswers();

  // Tuner --------------------------------------------------------------------------
  let stream = null, raf = null, analyser = null, buf = null;
  function detectPitch(data, sampleRate) {
    // normalised autocorrelation; picks the first strong peak (McLeod-style)
    let rms = 0;
    for (let i = 0; i < data.length; i++) rms += data[i] * data[i];
    rms = Math.sqrt(rms / data.length);
    if (rms < 0.01) return null;
    const n = data.length, maxLag = Math.floor(sampleRate / 50), minLag = Math.floor(sampleRate / 1400);
    const nsdf = new Float32Array(maxLag + 2);
    for (let lag = minLag; lag <= maxLag + 1; lag++) {
      let ac = 0, m = 0;
      for (let i = 0; i < n - lag; i++) { ac += data[i] * data[i + lag]; m += data[i] * data[i] + data[i + lag] * data[i + lag]; }
      nsdf[lag] = m > 0 ? (2 * ac) / m : 0;
    }
    let best = -1, bestVal = 0, globalMax = 0;
    for (let lag = minLag; lag <= maxLag; lag++) globalMax = Math.max(globalMax, nsdf[lag]);
    for (let lag = minLag + 1; lag <= maxLag; lag++) {
      if (nsdf[lag] > nsdf[lag - 1] && nsdf[lag] >= nsdf[lag + 1] && nsdf[lag] > 0.9 * globalMax && nsdf[lag] > 0.5) {
        best = lag; bestVal = nsdf[lag]; break;
      }
    }
    if (best < 0) return null;
    const a = nsdf[best - 1], b = nsdf[best], c = nsdf[best + 1];
    const shift = (a - c) / (2 * (a - 2 * b + c) || 1);
    return { freq: sampleRate / (best + shift), clarity: bestVal };
  }
  let smooth = null;
  function tunerLoop() {
    analyser.getFloatTimeDomainData(buf);
    const r = detectPitch(buf, analyser.context.sampleRate);
    if (r) {
      const a4 = +$("tuner-a4").value || 440;
      const midiFloat = 69 + 12 * Math.log2(r.freq / a4);
      smooth = smooth === null || Math.abs(midiFloat - smooth) > 0.6 ? midiFloat : smooth * 0.7 + midiFloat * 0.3;
      const nearest = Math.round(smooth);
      const cents = Math.round((smooth - nearest) * 100);
      $("tuner-note").textContent = noteName(nearest);
      $("tuner-needle").style.left = `${50 + cents}%`;
      $("tuner-needle").classList.toggle("in-tune", Math.abs(cents) <= 5);
      $("tuner-detail").textContent = `${r.freq.toFixed(1)} Hz, ${cents > 0 ? "+" : ""}${cents} cents${Math.abs(cents) <= 5 ? ", in tune" : cents < 0 ? ", tune up" : ", tune down"}`;
    }
    raf = requestAnimationFrame(tunerLoop);
  }
  $("tuner-toggle").addEventListener("click", async () => {
    if (stream) {
      cancelAnimationFrame(raf);
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      $("tuner-toggle").textContent = "Start the tuner";
      $("tuner-note").textContent = "–";
      return;
    }
    const ac = audio();
    if (!ac || !navigator.mediaDevices?.getUserMedia) { $("tuner-detail").textContent = "This browser can't use the microphone here."; return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (e) {
      $("tuner-detail").textContent = "The microphone is blocked. Allow it in the browser's address bar, then try again.";
      return;
    }
    const src = ac.createMediaStreamSource(stream);
    analyser = ac.createAnalyser();
    analyser.fftSize = 4096;
    buf = new Float32Array(analyser.fftSize);
    src.connect(analyser);
    $("tuner-toggle").textContent = "Stop the tuner";
    $("tuner-detail").textContent = "Listening. Play one note at a time.";
    tunerLoop();
  });

  // Metronome ------------------------------------------------------------------------
  let bpm = 100, running = false, nextTime = 0, beat = 0, timer = null, taps = [];
  const beatsEl = $("metro-beats");
  function drawBeats() {
    const n = +$("metro-meter").value;
    beatsEl.innerHTML = Array.from({ length: n }, (_, i) => `<i class="${i === 0 ? "first" : ""}"></i>`).join("");
  }
  function setBpm(v) {
    bpm = Math.max(30, Math.min(240, Math.round(v)));
    $("metro-bpm").textContent = bpm;
    $("metro-tempo").value = bpm;
  }
  function schedule() {
    const ac = audio();
    const n = +$("metro-meter").value;
    while (nextTime < ac.currentTime + 0.12) {
      const b = beat % n;
      click(nextTime, b === 0 && $("metro-accent").checked);
      const delay = Math.max(0, (nextTime - ac.currentTime) * 1000);
      setTimeout(() => {
        beatsEl.querySelectorAll("i").forEach((el, i) => el.classList.toggle("on", i === b));
      }, delay);
      nextTime += 60 / bpm;
      beat++;
    }
  }
  $("metro-toggle").addEventListener("click", () => {
    const ac = audio();
    if (!ac) return;
    running = !running;
    $("metro-toggle").textContent = running ? "Stop" : "Start";
    if (running) {
      beat = 0;
      nextTime = ac.currentTime + 0.06;
      schedule();
      timer = setInterval(schedule, 25);
    } else {
      clearInterval(timer);
      beatsEl.querySelectorAll("i").forEach((el) => el.classList.remove("on"));
    }
  });
  $("metro-tempo").addEventListener("input", (e) => setBpm(+e.target.value));
  $("metro-down").addEventListener("click", () => setBpm(bpm - 1));
  $("metro-up").addEventListener("click", () => setBpm(bpm + 1));
  $("metro-meter").addEventListener("change", () => { drawBeats(); beat = 0; });
  $("metro-tap").addEventListener("click", () => {
    const now = performance.now();
    taps = taps.filter((t) => now - t < 3000);
    taps.push(now);
    if (taps.length >= 2) {
      const gaps = taps.slice(1).map((t, i) => t - taps[i]);
      setBpm(60000 / (gaps.reduce((a, b) => a + b, 0) / gaps.length));
    }
  });
  drawBeats();
})();
