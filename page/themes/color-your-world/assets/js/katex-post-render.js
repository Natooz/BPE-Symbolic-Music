function getAll(selector) {
  return Array.prototype.slice
    .call(document.querySelectorAll(selector), 0);
}

// tabindex hack
function addTabIndex() {
  'use strict';

  const katexBlocks = getAll('span.katex-display');
  
  katexBlocks.forEach(function(el) {
    el.tabIndex = 0;
  });
}

addTabIndex();
