<script>
  import Home from "./Home.svelte";
  import Chat from "./Chat.svelte";
  import logo from "./assets/logo.png";

  let currentPath = window.location.pathname;
  let mobileMenuOpen = false;

  window.addEventListener("popstate", () => {
    currentPath = window.location.pathname;
  });

  function navigate(path) {
    if (currentPath === path) return;
    window.history.pushState({}, "", path);
    window.dispatchEvent(new Event("popstate"));
    mobileMenuOpen = false;
  }

  function toggleMenu() {
    mobileMenuOpen = !mobileMenuOpen;
  }
</script>

<div class="saas-layout">
  <!-- Top Nav -->
  <header class="top-nav">
    <div class="nav-left">
      <button
        class="icon-btn menu-btn"
        on:click={toggleMenu}
        aria-label="Toggle menu"
      >
        {#if mobileMenuOpen}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            ><line x1="18" y1="6" x2="6" y2="18"></line><line
              x1="6"
              y1="6"
              x2="18"
              y2="18"
            ></line></svg
          >
        {:else}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            ><line x1="3" y1="12" x2="21" y2="12"></line><line
              x1="3"
              y1="6"
              x2="21"
              y2="6"
            ></line><line x1="3" y1="18" x2="21" y2="18"></line></svg
          >
        {/if}
      </button>

      <div class="brand" on:click={() => navigate("/")}>
        <img src={logo} alt="Truth Shield Logo" class="logo" />
        <h2>TRUTH SHIELD</h2>
      </div>
    </div>

    <div class="nav-center">
      <!-- Centered space intentionally empty -->
    </div>

    <div class="nav-right">
      <button
        class="nav-link {currentPath === '/' ? 'active' : ''}"
        on:click={() => navigate("/")}>Overview</button
      >
      <button
        class="nav-link {currentPath === '/chat' ? 'active' : ''}"
        on:click={() => navigate("/chat")}>Detector</button
      >
    </div>
  </header>

  <!-- Mobile Drawer -->
  {#if mobileMenuOpen}
    <div class="mobile-overlay" on:click={toggleMenu}></div>
    <nav class="mobile-drawer">
      <div class="mobile-brand">
        <img src={logo} alt="Truth Shield Logo" class="logo" />
        <span>TRUTH SHIELD</span>
      </div>
      <button
        class="mobile-nav-link {currentPath === '/' ? 'active' : ''}"
        on:click={() => navigate("/")}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          ><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"
          ></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg
        >
        Overview
      </button>
      <button
        class="mobile-nav-link {currentPath === '/chat' ? 'active' : ''}"
        on:click={() => navigate("/chat")}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          ><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg
        >
        Detector
      </button>
    </nav>
  {/if}

  <!-- Main Content Area -->
  <main class="main-content">
    <div class="view-wrapper">
      {#if currentPath === "/chat"}
        <Chat />
      {:else}
        <Home {navigate} />
      {/if}
    </div>
  </main>
</div>

<style>
  .saas-layout {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background-color: var(--bg-color);
  }

  .top-nav {
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    border-bottom: 1px solid var(--border-strong);
    background-color: rgba(10, 11, 14, 0.95);
    backdrop-filter: blur(8px);
    flex-shrink: 0;
    position: absolute;
    top: 0;
    width: 100%;
    z-index: 100;
    box-sizing: border-box;
  }

  .nav-left {
    display: flex;
    align-items: center;
    gap: 1.2rem;
  }

  .menu-btn {
    cursor: pointer;
    background: transparent;
    border: none;
    color: var(--text-muted);
    padding: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.2s;
  }

  .menu-btn:hover {
    color: var(--text-main);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    cursor: pointer;
  }

  .logo {
    width: 26px;
    height: 26px;
    object-fit: contain;
  }

  .brand h2 {
    font-size: 1rem;
    font-weight: normal;
    color: var(--text-main);
    letter-spacing: 2px;
    white-space: nowrap;
  }

  .nav-right {
    display: flex;
    align-items: center;
    gap: 2rem;
  }

  .nav-link {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-family: inherit;
    font-size: 0.95rem;
    cursor: pointer;
    transition: color 0.2s;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-decoration: underline;
    text-underline-offset: 6px;
    text-decoration-color: transparent;
  }

  .nav-link:hover {
    color: var(--text-main);
    text-decoration-color: var(--border-strong);
  }

  .nav-link.active {
    color: var(--primary);
    text-decoration-color: var(--primary);
  }

  .main-content {
    flex: 1;
    height: calc(100vh - 60px);
    display: flex;
    flex-direction: column;
    min-width: 0;
    position: relative;
  }

  .view-wrapper {
    flex: 1;
    position: relative;
    overflow-y: auto;
    overflow-x: hidden;
  }

  /* Mobile Drawer */
  .mobile-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    z-index: 200;
    backdrop-filter: blur(2px);
  }

  .mobile-drawer {
    position: fixed;
    top: 0;
    left: 0;
    width: 280px;
    height: 100vh;
    background: var(--panel-bg);
    border-right: 1px solid var(--border-strong);
    z-index: 300;
    padding: 2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    animation: slideIn 0.25s ease;
  }

  @keyframes slideIn {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(0);
    }
  }

  .mobile-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-strong);
    font-family: "CookConthic", sans-serif;
    font-size: 1rem;
    letter-spacing: 2px;
    color: var(--text-main);
  }

  .mobile-nav-link {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-family: inherit;
    font-size: 0.9rem;
    cursor: pointer;
    padding: 0.8rem 1rem;
    border-radius: var(--radius-md);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    transition: all 0.2s;
    text-align: left;
  }

  .mobile-nav-link:hover {
    background: var(--border);
    color: var(--text-main);
  }

  .mobile-nav-link.active {
    color: var(--primary);
    background: var(--primary-light);
  }

  /* Responsive hide desktop nav on small screens */
  @media (max-width: 768px) {
    .nav-right {
      display: none;
    }

    .brand h2 {
      font-size: 0.85rem;
    }

    .top-nav {
      padding: 0 1rem;
    }
  }
</style>
