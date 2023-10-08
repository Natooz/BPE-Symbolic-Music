// KaTeX's auto-render extension with custom options
// See the page below for more details on how to configure it
// https://katex.org/docs/autorender.html

renderMathInElement(document.body, {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '\\[', right: '\\]', display: true },
    { left: '$', right: '$', display: false },
    { left: '\\(', right: '\\)', display: false }
  ]
});
