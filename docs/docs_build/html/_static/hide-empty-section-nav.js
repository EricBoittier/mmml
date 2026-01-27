// If the left "Section Navigation" has no children,
// populate it with the right-hand "On this page" toc instead of leaving it blank.
(function () {
  function populateLeftWithPageTOC() {
    const leftNav = document.querySelector('nav.bd-docs-nav.bd-links');
    if (!leftNav) return;
    const leftTitle = leftNav.querySelector('.bd-links__title');
    const leftContainer = leftNav.querySelector('.bd-toc-item');
    if (!leftContainer) return;

    // If there are already section links (a child toctree), keep the default behavior
    const hasSectionLinks = leftContainer.querySelector('a, .nav-link, li');
    if (hasSectionLinks) {
      // Ensure default title is shown
      if (leftTitle) leftTitle.textContent = 'Section Navigation';
      leftNav.setAttribute('aria-label', 'Section Navigation');
      leftContainer.dataset.populatedFromPageToc = 'false';
      return;
    }

    // Find the right-hand page toc
    const rightPageToc = document.querySelector('.bd-toc-nav.page-toc');
    if (!rightPageToc) return;
    const list = rightPageToc.querySelector('ul');
    if (!list) return;

    // Avoid duplicating content on repeated calls
    if (leftContainer.dataset.populatedFromPageToc === 'true') return;

    // Clone the page toc list and inject it into the left container
    const cloned = list.cloneNode(true);
    // Normalize classes for left nav styling
    cloned.classList.add('navbar-nav');
    leftContainer.innerHTML = '';
    leftContainer.appendChild(cloned);

    // Update title and ARIA to "On this page"
    if (leftTitle) leftTitle.textContent = 'On this page';
    leftNav.setAttribute('aria-label', 'On this page');
    leftContainer.dataset.populatedFromPageToc = 'true';
  }

  function init() {
    // Initial population
    populateLeftWithPageTOC();

    // Observe changes in the right toc (e.g., when theme populates it lazily)
    const right = document.querySelector('.bd-toc-nav.page-toc');
    if (right) {
      const obs = new MutationObserver(() => populateLeftWithPageTOC());
      obs.observe(right, { childList: true, subtree: true });
    }

    // Also observe the left nav in case toctrees are injected dynamically
    const left = document.querySelector('nav.bd-docs-nav.bd-links');
    if (left) {
      const obsLeft = new MutationObserver(() => populateLeftWithPageTOC());
      obsLeft.observe(left, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
