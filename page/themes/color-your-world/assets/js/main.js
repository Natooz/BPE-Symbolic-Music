'use strict';

// Get default accent colors
{{ $defaultDarkAccent   := .Site.Params.Style.darkAccent  | default .Site.Data.default.style.darkAccent }}
{{ $defaultLightAccent  := .Site.Params.Style.lightAccent | default .Site.Data.default.style.lightAccent }}

// Get CSS transition
{{ $changeTransition := .Site.Params.Style.changeTransition | default .Site.Data.default.style.changeTransition }}

// =================================================
// Mode switcher + Custom accent color
// Based on: https://gist.github.com/regpaq/04c67e8aceecbf0fd819945835412d1f
// =================================================

const rootElement = document.documentElement;

// Set the dark mode
function setDark() {
  rootElement.setAttribute('data-mode', 'dark');
}

// Set the light mode
function setLight() {
  rootElement.setAttribute('data-mode', 'light');
}

/*
 * Initialization triggers dark/light mode based on 3 things
 * The priority follows:
 * 
 * 1. Local preference (localStorage)
 * 2. System settings (prefers-color-scheme)
 * 3. HTML data-* attribute (data-mode)
 */

const localMode = localStorage.getItem('mode');

{{ if not .Site.Params.Style.ignoreSystemSettings }}
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
  const prefersLight = window.matchMedia('(prefers-color-scheme: light)');
{{ end }}

if (localMode === 'dark') {
  setDark();
} else if (localMode === 'light') {
  setLight();

  {{ if not .Site.Params.Style.ignoreSystemSettings }}
    } else if (prefersDark.matches) {
      setDark();
    } else if (prefersLight.matches) {
      setLight();
  {{ end }}

}


{{ if .Site.IsServer }}

  function capitalize(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }

  const currentModeTitle = capitalize(rootElement.getAttribute('data-mode'));

  if (localMode !== null) {
    console.log(currentModeTitle +
    ' mode loaded via local preference (localStorage).');
  } else if (typeof prefersDark !== 'undefined') {
    console.log(currentModeTitle +
    ' mode loaded via system settings (prefers-color-scheme).');
  } else {
    console.log(currentModeTitle +
    ' mode loaded via HTML data-* attribute (data-mode).');
  }
  
{{ end }}


function getAccent() {
  
  const currentMode = rootElement.getAttribute('data-mode');
  
  const localDarkAccent = localStorage.getItem('darkAccent');
  const localLightAccent = localStorage.getItem('lightAccent');
  
  let currentAccent;

  if (currentMode === 'dark') {
  
    if (localDarkAccent === null) {
      currentAccent = '{{ $defaultDarkAccent }}';
    } else {
      currentAccent = localDarkAccent;
    }
    
  } else {

    if (localLightAccent === null) {
      currentAccent = '{{ $defaultLightAccent }}';
    } else {
      currentAccent = localLightAccent;
    }
    
  }
  
  {{ if .Site.IsServer }}
  
    if (
      (currentMode === 'dark') &&
      (localStorage.getItem('darkAccent') !== null) ||
      (currentMode === 'light') &&
      (localStorage.getItem('lightAccent') !== null)
    ) {
      console.log('Custom accent color defined. Loading custom ' +
      currentMode + ' accent (' + currentAccent + ').');
    } else {
      console.log('Custom accent color NOT defined. Loading default ' +
      currentMode + ' accent (' + currentAccent + ').');
    }
    
  {{ end }}
  
  return currentAccent;
}

const activeAccent = getAccent();

// Set the active accent color for these right after setting dark/light mode
// Should mitigate any flashing/flickering
const rootStyle = rootElement.style;

rootStyle.setProperty('--accent', activeAccent);


// Also meta-theme cuz, why not
const metaThemeColor = document.querySelector('meta[name=theme-color]');

metaThemeColor.setAttribute('content', activeAccent);



document.addEventListener('DOMContentLoaded', function() {

  const colorPicker = document.querySelector('footer input');
  
  function updateAccent() {
    const activeAccent = getAccent();

    rootStyle.setProperty('--accent', activeAccent);
    colorPicker.value = activeAccent;
    metaThemeColor.setAttribute('content', activeAccent);
  }

  colorPicker.onchange = function() {

    const selectedAccent = colorPicker.value;

    rootStyle.setProperty('--accent', selectedAccent);
    
    if (rootElement.getAttribute('data-mode') === 'dark') {
      localStorage.setItem('darkAccent', selectedAccent);
    } else {
      localStorage.setItem('lightAccent', selectedAccent);
    }
    
    updateAccent();
  }

  // Update the color picker with the active accent color
  colorPicker.value = activeAccent;

  // Smooth transition, only when changing modes (and not loading pages)
  function smoothTransition() {
    document.body.style.transition =
    document.querySelector('header').style.transition =
    document.querySelector('footer').style.transition =
    '{{ printf "background-color %[1]s, color %[1]s" $changeTransition }}';
  }
  
  // Change mode via localStorage
  function localModeChange() {
  
    smoothTransition();

    if (rootElement.getAttribute('data-mode') === 'light') {
      setDark();
      localStorage.setItem('mode', 'dark');
    } else {
      setLight();
      localStorage.setItem('mode', 'light');
    }
    
    {{ if .Site.IsServer }}
      console.log('Local: ' +
      capitalize(localStorage.getItem('mode') + ' mode set.'));
    {{ end }}
    
    updateAccent();
  }

  
  {{ if .Site.IsServer }}
  
    // TEST
    // Keyboard shortcut for mode change, here for testing purposes only
    // CTRL + ALT + M
    document.addEventListener('keydown', (event) => {
      const e = event || window.event;
      if (e.keyCode === 77 && e.ctrlKey && e.altKey) {
        localModeChange();
        return;
      }
    }, false);
    
  {{ end }}


  {{ if not .Site.Params.Style.ignoreSystemSettings }}
  
    // Change mode via system settings
    function systemModeChange() {
    
      smoothTransition();
      
      if (prefersDark.matches) {
        setDark();
      } else {
        setLight();
      }
      
      {{ if .Site.IsServer }}
        console.log('System: ' +
        capitalize(rootElement.getAttribute('data-mode')) + ' mode set.');
      {{ end }}
      
      updateAccent();
      
      // System settings do not require localStorage
      if (localMode !== null) {
        localStorage.removeItem('mode');
      }

    }

    // System settings listener
    prefersDark.addEventListener('change', systemModeChange);
  
  {{ end }}

  // Mode change button listener
  document.querySelector('footer button')
    .addEventListener('click', localModeChange);
});
