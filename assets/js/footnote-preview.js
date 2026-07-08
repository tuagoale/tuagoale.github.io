(() => {
  const refSelector = 'a.footnote-ref[href^="#fn"], a[role="doc-noteref"][href^="#fn"]';
  const backlinkSelector = '.footnote-backref, [role="doc-backlink"], a[href^="#fnref"]';
  const visibleClass = 'is-visible';
  const viewportMargin = 12;

  let tooltip;
  let tooltipLabel;
  let tooltipContent;
  let activeRef;
  let hideTimer;
  let positionFrame;
  const previews = new WeakMap();

  function cloneFootnoteContent(note) {
    const clone = note.cloneNode(true);
    clone.removeAttribute('id');

    clone.querySelectorAll(backlinkSelector).forEach((backlink) => {
      const previous = backlink.previousSibling;
      backlink.remove();

      if (previous && previous.nodeType === Node.TEXT_NODE) {
        previous.textContent = previous.textContent.replace(/\s+$/, '');
      }
    });

    clone.querySelectorAll('[id]').forEach((node) => node.removeAttribute('id'));

    return clone.innerHTML.trim();
  }

  function footnoteForRef(ref) {
    const id = ref.hash ? ref.hash.slice(1) : '';
    if (!id) {
      return null;
    }

    try {
      return document.getElementById(decodeURIComponent(id)) || document.getElementById(id);
    } catch {
      return document.getElementById(id);
    }
  }

  function createTooltip() {
    tooltip = document.createElement('div');
    tooltip.id = 'footnote-popover';
    tooltip.className = 'footnote-popover';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.hidden = true;

    tooltipLabel = document.createElement('div');
    tooltipLabel.className = 'footnote-popover-label';

    tooltipContent = document.createElement('div');
    tooltipContent.className = 'footnote-popover-content';

    tooltip.append(tooltipLabel, tooltipContent);
    document.body.appendChild(tooltip);

    tooltip.addEventListener('mouseenter', cancelHide);
    tooltip.addEventListener('mouseleave', scheduleHide);
  }

  function cancelHide() {
    window.clearTimeout(hideTimer);
  }

  function scheduleHide() {
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(hideTooltip, 140);
  }

  function hideTooltip() {
    window.clearTimeout(hideTimer);
    window.cancelAnimationFrame(positionFrame);

    if (activeRef) {
      activeRef.removeAttribute('aria-describedby');
    }

    activeRef = null;
    tooltip.classList.remove(visibleClass);
    tooltip.hidden = true;
  }

  function typesetTooltip() {
    if (typeof window.renderMathInElement !== 'function' || tooltipContent.querySelector('.katex')) {
      return;
    }

    try {
      window.renderMathInElement(tooltipContent, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        ignoredClasses: ['katex'],
        throwOnError: false,
      });
    } catch {
      // The regular page renderer remains the source of truth if preview math cannot render.
    }
  }

  function positionTooltip() {
    if (!activeRef || tooltip.hidden) {
      return;
    }

    const refRect = activeRef.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const preferredLeft = refRect.left + refRect.width / 2 - tooltipRect.width / 2;
    const maxLeft = window.innerWidth - tooltipRect.width - viewportMargin;
    const left = Math.max(viewportMargin, Math.min(preferredLeft, maxLeft));

    let placement = 'top';
    let top = refRect.top - tooltipRect.height - 10;

    if (top < viewportMargin) {
      placement = 'bottom';
      top = refRect.bottom + 10;
    }

    const maxTop = window.innerHeight - tooltipRect.height - viewportMargin;
    top = Math.max(viewportMargin, Math.min(top, maxTop));

    tooltip.dataset.placement = placement;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.style.setProperty('--footnote-arrow-left', `${refRect.left + refRect.width / 2 - left}px`);
  }

  function requestPosition() {
    window.cancelAnimationFrame(positionFrame);
    positionFrame = window.requestAnimationFrame(positionTooltip);
  }

  function showTooltip(ref) {
    const previewHtml = previews.get(ref);
    if (!previewHtml) {
      return;
    }

    cancelHide();
    activeRef = ref;
    tooltipLabel.textContent = `#${ref.textContent.trim() || '?'}`;
    tooltipContent.innerHTML = previewHtml;
    tooltip.hidden = false;
    tooltip.classList.remove(visibleClass);
    ref.setAttribute('aria-describedby', tooltip.id);

    typesetTooltip();
    positionTooltip();

    window.requestAnimationFrame(() => {
      positionTooltip();
      tooltip.classList.add(visibleClass);
    });
  }

  function enhanceRef(ref) {
    const note = footnoteForRef(ref);
    if (!note) {
      return;
    }

    const previewHtml = cloneFootnoteContent(note);
    if (!previewHtml) {
      return;
    }

    previews.set(ref, previewHtml);
    ref.classList.add('has-footnote-preview');

    const sup = ref.closest('sup');
    if (sup) {
      sup.classList.add('footnote-preview-trigger');
    }

    ref.addEventListener('mouseenter', () => showTooltip(ref));
    ref.addEventListener('mouseleave', scheduleHide);
    ref.addEventListener('focus', () => showTooltip(ref));
    ref.addEventListener('blur', scheduleHide);
    ref.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        hideTooltip();
      }
    });
  }

  function init() {
    const refs = [...document.querySelectorAll(refSelector)];
    if (!refs.length) {
      return;
    }

    createTooltip();
    refs.forEach(enhanceRef);

    window.addEventListener('scroll', requestPosition, { passive: true });
    window.addEventListener('resize', requestPosition);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
