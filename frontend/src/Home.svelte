<script>
  import { onMount } from 'svelte';
  
  export let navigate;

  onMount(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('reveal-visible');
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -60px 0px' });

    document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));

    return () => observer.disconnect();
  });

  function startAnalysis() {
    navigate('/chat');
  }
</script>

<div class="landing-container">
  <header class="hero-section">
    <div class="hero-bg"></div>
    <div class="grain-overlay"></div>

    <!-- Meta bar to mimic the reference top meta stats -->
    <div class="meta-bar animate-fade-in">
    <span class="meta-item"><span class="dot" style="background: var(--accent);"></span> REAL-TIME ENGINE</span>
    <span class="meta-item">PHASE 2 FUSION</span>
    <span class="meta-item">92.8% ACCURACY</span>
    <span class="meta-item" style="color: var(--primary);">MODEL: ROBERTA V2</span>
  </div>

  <main class="hero-grid">
    <div class="hero-left">
      <h1 class="super-title animate-fade-in" style="animation-delay: 0.1s">
        INF<span class="font-highlight">O</span><br/>
        <span class="outline-text">INTEGRITY</span><br/>
        <span class="font-highlight" style="color: var(--primary);">V.2.0</span>
      </h1>
      
      <div class="cta-actions animate-fade-in" style="animation-delay: 0.3s">
        <button class="btn-minimal" on:click={startAnalysis}>
          Open Detector ↗
        </button>
      </div>
    </div>

    <div class="hero-right">
      <p class="subtitle animate-fade-in" style="animation-delay: 0.2s">
        Real-time contextual analysis powered by a dual-phase classification approach.<br/><br/>
        Unprecedented accuracy for modern <span class="font-highlight" style="color: var(--accent);">threat</span> intelligence.
      </p>
    </div>
  </main>
  </header>

  <div class="scrolling-marquee">
    <div class="marquee-content">
      <div class="marquee-track">
        <span>&#9672; ROBERTA V2</span>
        <span>&#9672; DUAL-PHASE CLASSIFICATION</span>
        <span>&#9672; 240,000 SAMPLES</span>
        <span>&#9672; 99.4% F1 FUSION</span>
        <span>&#9672; GPT-2 STAT FEATURE HEAD</span>
        <span>&#9672; REAL-TIME DEPLOYMENT</span>
      </div>
      <div class="marquee-track" aria-hidden="true">
        <span>&#9672; ROBERTA V2</span>
        <span>&#9672; DUAL-PHASE CLASSIFICATION</span>
        <span>&#9672; 240,000 SAMPLES</span>
        <span>&#9672; 99.4% F1 FUSION</span>
        <span>&#9672; GPT-2 STAT FEATURE HEAD</span>
        <span>&#9672; REAL-TIME DEPLOYMENT</span>
      </div>
    </div>
  </div>

  <section class="training-section" id="metrics-section">
    <div class="train-left reveal-on-scroll">
      <h2 class="section-title">
        MODEL<br/>
        <span class="outline-text">ARCHITECTURE</span><br/>
        & METRICS
      </h2>
      <img src="/src/assets/art.png" alt="Vigil AI Structural Art" class="arch-art" />
    </div>
    
    <div class="train-right reveal-on-scroll" style="transition-delay: 0.2s;">
      <p class="train-description">
        Trained extensively over 240,000 cross-platform samples (Real/Fake 1:1 ratio) on an NVIDIA RTX A6000. 
        <br/><br/>
        <strong>Phase 1 (Linguistic Extractor):</strong> Utilizes a RoBERTa Sequence Classification backbone to extract deep contextual semantics, pushing the baseline accuracy to a raw 98.0%. 
        <br/><br/>
        <strong>Phase 2 (Statistical Fusion Head):</strong> Because sophisticated threats circumvent generic language models, the stream is fed into a localized GPT-2 style statistical projection head. This layers raw metadata and attention density atop the structural language cues—terminating in a highly resilient 99.4% F1-Score devoid of zero-shot hallucinations.
      </p>
      
      <div class="metrics-row">
        <div class="metric">
          <span class="metric-val" style="color: var(--primary);">99.4%</span>
          <span class="metric-label">TEST F1</span>
        </div>
        <div class="metric">
          <span class="metric-val" style="color: var(--accent);">0.211</span>
          <span class="metric-label">FINAL LOSS</span>
        </div>
        <div class="metric">
          <span class="metric-val">1.9M</span>
          <span class="metric-label">TOKENS/S</span>
        </div>
      </div>
      
      <div class="css-graph-container">
        <div class="css-graph">
          <div class="graph-bar" style="height: 100%;"></div>
          <div class="graph-bar" style="height: 80%;"></div>
          <div class="graph-bar" style="height: 75%;"></div>
          <div class="graph-bar" style="height: 68%;"></div>
          <div class="graph-bar" style="height: 60%;"></div>
          <div class="graph-bar" style="height: 58%;"></div>
          <div class="graph-bar" style="height: 52%; background: var(--accent);"></div>
        </div>
        <span class="graph-label">Epoch 1-7 RoBERTa Training Loss Trajectory (0.234 &rarr; 0.211)</span>
      </div>
      
      <div class="arch-diagram">
        <div class="arch-node">RoBERTa Backbone Core</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-split">
          <div class="arch-subnode" style="color: var(--primary); border-color: var(--primary);">768 Semantic</div>
          <div class="arch-plus">+</div>
          <div class="arch-subnode">3 Stat Vector</div>
        </div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-node" style="border-style: dashed;">2-Layer MLP Fusion Engine</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-node final-node">Phase 2 Verification Identity</div>
      </div>

    </div>
  </section>

  <section class="training-section" style="padding-top: 2rem;">
    <div class="train-left reveal-on-scroll" style="display: flex; justify-content: center; align-items: flex-start;">
      <div class="arch-diagram" style="margin-top: 0; width: 100%;">
        <div class="arch-node" style="border-color: var(--accent);">Gemma-3 Generative Core</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-split">
          <div class="arch-subnode" style="color: var(--highlight-high); border-color: var(--highlight-high);">LLM Claim Extraction</div>
          <div class="arch-plus">+</div>
          <div class="arch-subnode" style="color: var(--success); border-color: var(--success);">Live GNews Verification</div>
        </div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-node" style="border-style: dotted;">Cross-Reference via Prompt Fusion</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-node final-node" style="background: rgba(0, 255, 128, 0.05); border-color: var(--success); color: var(--success);">Final Threat Intelligence Stream</div>
      </div>
    </div>
    <div class="train-right reveal-on-scroll phase3-right" style="transition-delay: 0.2s;">
      <h2 class="section-title" style="font-size: 3rem; margin-bottom: 2rem;">PHASE 3<br/><span style="color: var(--highlight-high);">MULTI-AGENT SYNTHESIS</span></h2>
      <p class="train-description">
        While RoBERTa isolates structural linguistic mimicry, it cannot fact-check real-world events natively. Truth Shield resolves this by routing both the raw text and ML confidence scores through a live GenAI agentic pipeline.
      </p>
      <p class="train-description">
        The agent extracts core claims, queries <strong>GNews</strong> for live global corroboration, and fuses both signals into a final verdict using a calibrated weighting model.
      </p>
      <div class="weight-breakdown">
        <div class="weight-row">
          <span class="weight-label">Live Web Verification</span>
          <div class="weight-bar-wrap">
            <div class="weight-bar" style="width: 60%; background: var(--success);"></div>
          </div>
          <span class="weight-pct" style="color: var(--success);">60%</span>
        </div>
        <div class="weight-row">
          <span class="weight-label">Custom RoBERTa Model</span>
          <div class="weight-bar-wrap">
            <div class="weight-bar" style="width: 40%; background: var(--primary);"></div>
          </div>
          <span class="weight-pct" style="color: var(--primary);">40%</span>
        </div>
      </div>
      <p class="train-description" style="margin-bottom: 0; font-size: 0.9rem; color: var(--text-muted);">
        Live evidence takes precedence. The custom model acts as a structural tiebreaker — especially effective when web sources are delayed on breaking events.
      </p>
    </div>
  </section>

  <footer class="page-footer">
    <div class="footer-content">
      <span>TRUTH SHIELD STRUCTURES</span>
      <span style="color: var(--primary);">EST. 2026 // INFORMATION INTEGRITY</span>
      <span>SYSTEM <span style="color: var(--accent);">ONLINE</span></span>
    </div>
  </footer>
</div>

<style>
  :global(.reveal-on-scroll) {
    opacity: 0;
    transform: translateY(40px);
    transition: opacity 0.9s cubic-bezier(0.16, 1, 0.3, 1), transform 0.9s cubic-bezier(0.16, 1, 0.3, 1);
  }

  :global(.reveal-visible) {
    opacity: 1 !important;
    transform: translateY(0) !important;
  }

  .landing-container {
    display: flex;
    flex-direction: column;
    position: relative;
    overflow-x: hidden;
  }

  .hero-section {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 6rem 8% 2rem 8%;
    position: relative;
  }

  .hero-bg {
    position: absolute;
    inset: 0;
    background-image: url('/hero.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    opacity: 0.05;
    z-index: 0;
    pointer-events: none;
  }

  .grain-overlay {
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    opacity: 0.06;
    mix-blend-mode: overlay;
  }

  .meta-bar {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-top: 1rem;
    position: relative;
    z-index: 10;
    font-size: 0.75rem;
    font-family: inherit;
    letter-spacing: 2px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .meta-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .hero-grid {
    flex: 1;
    display: grid;
    grid-template-columns: 2fr 1fr;
    position: relative;
    z-index: 10;
    margin-top: 8rem;
    gap: 4rem;
  }

  .hero-left {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }

  .super-title {
    font-size: 8rem;
    line-height: 0.9;
    margin-bottom: 4rem;
    letter-spacing: -2px;
    opacity: 0;
    color: var(--text-main);
  }

  .outline-text {
    color: transparent;
    -webkit-text-stroke: 1px var(--text-muted);
  }

  .font-highlight {
    /* CookConthic font applied via app.css global class */
  }

  .cta-actions {
    display: flex;
    gap: 2rem;
    opacity: 0;
    margin-top: auto;
    margin-bottom: 6rem;
  }

  .btn-minimal {
    background: transparent;
    border: none;
    color: var(--text-main);
    font-size: 1.1rem;
    font-family: inherit;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid var(--primary);
    padding-bottom: 0.2rem;
    transition: all 0.2s;
  }

  .btn-minimal:hover {
    color: var(--primary);
    letter-spacing: 1px;
  }

  .hero-right {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    text-align: right;
    padding-top: 2rem;
  }

  .subtitle {
    font-size: 1.25rem;
    color: var(--text-muted);
    line-height: 1.8;
    max-width: 380px;
    margin-left: auto;
    opacity: 0;
  }



  /* Marquee */
  .scrolling-marquee {
    position: relative;
    width: 100%;
    margin-top: 0;
    padding: 1.5rem 0;
    overflow: hidden;
    background: transparent;
    border-top: 1px solid var(--border-strong);
    border-bottom: 1px solid var(--border-strong);
    z-index: 10;
    display: flex;
  }

  .marquee-content {
    display: flex;
    white-space: nowrap;
    width: max-content;
    animation: scroll-left 30s linear infinite;
  }

  .marquee-track {
    display: flex;
    gap: 3rem;
    padding-right: 3rem;
  }

  .marquee-track span {
    font-size: 0.85rem;
    font-weight: 600;
    font-family: inherit;
    color: var(--text-muted);
    letter-spacing: 2px;
  }

  .marquee-track span:nth-child(even) {
    color: var(--primary);
  }

  @keyframes scroll-left {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }

  /* Training Section */
  .training-section {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    padding: 6rem 8%;
    gap: 6rem;
    position: relative;
    z-index: 10;
    margin-top: 0;
  }

  .section-title {
    font-size: 6rem;
    line-height: 0.9;
    letter-spacing: -2px;
    font-family: 'CookConthic', sans-serif;
  }

  .train-right {
    display: flex;
    flex-direction: column;
    padding-top: 1rem;
  }

  .arch-art {
    margin-top: 9rem;
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: var(--radius-lg);
    opacity: 0.9;
  }

  .train-description {
    font-size: 1.05rem;
    line-height: 1.8;
    color: var(--text-muted);
    margin-bottom: 3rem;
  }

  .metrics-row {
    display: flex;
    gap: 4rem;
    border-bottom: 1px solid var(--border-strong);
    padding-bottom: 2rem;
    margin-bottom: 3rem;
  }

  .metric {
    display: flex;
    flex-direction: column;
  }

  .metric-val {
    font-size: 3.5rem;
    font-weight: normal;
    font-family: 'CookConthic', sans-serif;
    line-height: 1;
    margin-bottom: 0.5rem;
  }

  .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: var(--text-muted);
  }

  /* Verdict Weight Breakdown */
  .weight-breakdown {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
    margin: 1.5rem 0;
    padding: 1.2rem;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    background: rgba(255,255,255,0.02);
  }

  .weight-row {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .weight-label {
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    min-width: 160px;
  }

  .weight-bar-wrap {
    flex: 1;
    height: 4px;
    background: var(--border-strong);
    border-radius: 2px;
    overflow: hidden;
  }

  .weight-bar {
    height: 100%;
    border-radius: 2px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .weight-pct {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    min-width: 36px;
    text-align: right;
  }

  /* CSS Graph */
  .css-graph-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .css-graph {
    display: flex;
    align-items: flex-end;
    gap: 0.8rem;
    height: 100px;
    width: 100%;
    border-bottom: 1px dashed var(--border-strong);
    padding-bottom: 0.5rem;
  }

  .graph-bar {
    flex: 1;
    background: var(--border-strong);
    border-radius: 2px 2px 0 0;
    transition: background 0.3s;
  }

  .graph-bar:hover {
    background: var(--primary);
  }

  .graph-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* Architecture Diagram */
  .arch-diagram {
    margin-top: 4rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    background: transparent;
    padding: 2rem;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
  }

  .arch-node {
    padding: 0.8rem 2rem;
    border: 1px solid var(--text-muted);
    border-radius: 4px;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-main);
    font-weight: 600;
  }

  .final-node {
    background: var(--text-main);
    color: var(--bg-color);
    border-color: var(--text-main);
  }

  .arch-arrow {
    font-size: 1.2rem;
    color: var(--text-muted);
  }

  .arch-split {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .arch-subnode {
    padding: 0.6rem 1.2rem;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .arch-plus {
    color: var(--accent);
    font-weight: bold;
    font-size: 1.2rem;
  }

  /* Footer */
  .page-footer {
    width: 100%;
    border-top: 1px solid var(--border-strong);
    padding: 2rem 8%;
    margin-top: 6rem;
    position: relative;
    z-index: 10;
  }

  .footer-content {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    letter-spacing: 2px;
    color: var(--text-muted);
    font-family: inherit;
    text-transform: uppercase;
  }

  @media (max-width: 1024px) {
    .super-title {
      font-size: 5rem;
    }
    .training-section {
      grid-template-columns: 1fr;
      padding: 4rem 5%;
      gap: 3rem;
    }
    .section-title {
      font-size: 4rem;
    }
    .arch-art {
      margin-top: 4rem;
    }
  }

  @media (max-width: 768px) {
    .meta-bar {
      flex-direction: column;
      gap: 1rem;
    }
    .hero-grid {
      grid-template-columns: 1fr;
      margin-top: 4rem;
    }
    .hero-right {
      text-align: left;
      padding-top: 0;
    }
    .subtitle {
      margin-left: 0;
      max-width: 100%;
    }
    .super-title {
      font-size: 4rem;
      margin-bottom: 2rem;
    }
    .cta-actions {
      margin-bottom: 2rem;
    }
    .arch-split {
      flex-direction: column;
      gap: 0.5rem;
    }
    .footer-content {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }
    .phase3-right {
      padding-left: 0;
    }
  }

  @media (max-width: 480px) {
    .super-title {
      font-size: 14vw;
      margin-bottom: 1.5rem;
    }
    .section-title {
      font-size: 12vw;
    }
    .metric-val {
      font-size: 12vw;
    }
    .metrics-row {
      gap: 1.5rem;
      flex-direction: column;
    }
    .arch-node {
      padding: 0.8rem 1rem;
      font-size: 0.75rem;
      text-align: center;
      width: 100%;
      box-sizing: border-box;
    }
    .arch-subnode {
      padding: 0.6rem 0.5rem;
      font-size: 0.65rem;
      text-align: center;
      width: 100%;
      box-sizing: border-box;
    }
    .arch-diagram {
      padding: 1rem;
    }
  }
</style>
