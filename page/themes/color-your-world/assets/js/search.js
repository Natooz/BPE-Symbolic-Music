// =================================================
// Basic search functionality via Fuse.js
// Based on: https://gist.github.com/eddiewebb/735feb48f50f0ddd65ae5606a1cb41ae#gistcomment-2987774
// =================================================

'use strict';

const fuseOptions = {
  keys: [
    { name: 'title',        weight: .4 },
    { name: 'tags',         weight: .3 },
    { name: 'description',  weight: .2 },
    { name: 'content',      weight: .1 }
  ],
  ignoreLocation: true,
  minMatchCharLength: {{ .Site.Params.Search.minLength | default .Site.Data.default.search.minLength }},
  shouldSort: false,
  threshold: 0
}

const searchResults = document.querySelector('#search-results');

// Populate results
function populateResults(output) {
  output.forEach((value) => {
    
    const el = value.item;

    const postTitle = el.title;
    const postDate = el.date;
    
    const htmlPostTitle = document.createElement('p');
    htmlPostTitle.textContent = postTitle;
              
    // Pull HTML template
    const resultsTemplate = document.querySelector('#search-results-template')
      .content.cloneNode(true);
      
    const postLink = resultsTemplate.querySelector('.btn');
    
    // Replace values
    postLink.setAttribute('href', el.permalink);
    postLink.setAttribute('title', postTitle);

    if (postDate) {
      const htmlPostDate = document.createElement('time');
      htmlPostDate.setAttribute('datetime', postDate);
      htmlPostDate.textContent = postDate;

      const htmlPostWithDate = document.createDocumentFragment();
      htmlPostWithDate.appendChild(htmlPostTitle);
      htmlPostWithDate.appendChild(htmlPostDate);

      postLink.setAttribute('title', postTitle + ' (' + postDate + ')');
      postLink.appendChild(htmlPostWithDate);
    } else {
      postLink.setAttribute('title', postTitle);
      postLink.appendChild(htmlPostTitle);
    }

    searchResults.appendChild(resultsTemplate);
  });
}


// Search info section
const searchInfo = document.querySelector('#search-info');

// Show message
function report(message, type) {
  const el = document.createElement('p');
  
  el.textContent = message;
  
  if (type) {
    el.classList.add(type);
  }
  
  searchInfo.appendChild(el);
}


function executeSearch(query) {
  fetch(searchResults.getAttribute('data-search-index'))
  .then((response) => {
    return response.json();
  })
  .then((data) => {
    
    // Limit results and throw an error if too many pages are found
    const limit = {{ .Site.Params.Search.maxResults | default 30 }};

    import(
      '/libs/fuse.js@' +
      searchResults.getAttribute('data-lib-version') +
      '/dist/fuse.basic.esm.min.js'
    )
    .then((fuseBasic) => {
      const fuse = new fuseBasic.default(data, fuseOptions);
      return fuse.search(query);
    })
    .then((output) => {
      searchInfo.firstElementChild.remove();
      report('{{ T "searchResultsFor" }}: ' + query);

      const matches = output.length;
      
      if (matches > 0) {
        if (matches === 1) {
          report('{{ T "searchOnePageFound" }}.');
        } else if (1 < matches && matches < limit + 1) {
          report(matches + ' {{ T "searchPagesFound" }}.');
        } else {
          report('{{ T "searchTooMany" }}', 'error');
        }
      } else {
        report('{{ T "searchNoPageFound" }}', 'error');
      }
      
      if (0 < matches && matches < limit + 1) {
        populateResults(output);
      }
    });
  });
}


// Sanitize
function getUrlParameter(string) {
  string = string.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
  const regex = new RegExp('[\\?&]' + string + '=([^&#]*)');
  const results = regex.exec(location.search);
  return results === null ? '' : decodeURIComponent(results[1].replace(/\+/g, ' '));
}

// Capture input
const searchQuery = getUrlParameter('q');

if (searchQuery) {

  // Transfer text to search field
  document.querySelector('.search-box input')
    .value = searchQuery;
  
  executeSearch(searchQuery);
  report('{{ T "searchProcessing" }}');
  
} else {
  report('{{ T "searchAwaitingSearch" }}');
}
