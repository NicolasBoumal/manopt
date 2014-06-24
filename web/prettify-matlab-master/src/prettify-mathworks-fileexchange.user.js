// ==UserScript==
// @name           Mathworks FileExchange: MATLAB highlighter
// @namespace      MathworksFileExchange_GoogleCodePrettify_MATLAB
// @description    Adds simple MATLAB syntax highlighting on Mathworks FileExchange
// @author         Amro <amroamroamro@gmail.com>
// @version        1.1
// @license        MIT License
// @icon           http://www.mathworks.com/favicon.ico
// @include        http://www.mathworks.com/matlabcentral/fileexchange/*
// @run-at         document-end
// ==/UserScript==

(function () {
	// create and inject a <script> element into page's DOM, with func source inlined.
	// It will be executed in the page scope, not the Greasemonkey sandbox
	// REFERENCE : http://wiki.greasespot.net/Content_Script_Injection
	function script_inject(func) {
		var script = document.createElement('script');
		script.setAttribute('type', 'text/javascript');
		script.textContent = '(' + func.toString() + ')();';
		document.body.appendChild(script);		// Insert script into page, so it will run
		//document.body.removeChild(script);	// immediately remove it to clean up
	}

	// GM_addStyle
	function style_inject(css) {
		var style = document.createElement('style');
		style.setAttribute('type', 'text/css');
		style.textContent = css.toString();
		document.getElementsByTagName('head')[0].appendChild(style);
	}
	function style_inject_byURL(cssURL) {
		var style = document.createElement('link');
		style.setAttribute('rel', 'stylesheet');
		style.setAttribute('type', 'text/css');
		style.setAttribute('href', cssURL);
		document.getElementsByTagName('head')[0].appendChild(style);
	}

	// activate only on an actual question page (ignore question lists, and such)
	if ( !/^\/matlabcentral\/fileexchange\/\d+/.test(window.location.pathname) ) {
		return;
	}

	// insert our custom CSS styles
	style_inject_byURL('http://google-code-prettify.googlecode.com/svn/trunk/src/prettify.css');
	style_inject([
		//=INSERT_FILE_QUOTED= ../css/lang-matlab.css
		'/* use horizontal scrollbars instead of wrapping long lines */',
		'pre.prettyprint { white-space: pre !important; overflow: auto !important; }',
		'/* add borders around code, give it a background color, and make it slightly indented */',
		'pre.prettyprint { padding: 4px; margin-left: 1em; background-color: #EEEEEE; }'
	].join(""));

	// insert out JS code
	script_inject(function () {
		// we require jQuery to be already loaded in the page
		if (typeof jQuery == 'undefined') { return; }

		// use jQuery Deferred to load prettify JS library, then execute our code
		jQuery.ajax({
			cache: true,	// use $.ajax instead of $.getScript to set cache=true (allows broswer to cache the script)
			async: true,
			dataType: 'script',
			url: 'http://google-code-prettify.googlecode.com/svn/trunk/src/prettify.js',
		}).done(function () {
			// register the new language handlers
			RegisterMATLABLanguageHandlers();

			// on DOMContentLoaded
			jQuery(document).ready(function () {
				// for each <pre.matlab-code> blocks
				var blocks = document.getElementsByTagName('pre');
				for (var i = 0; i < blocks.length; ++i) {
					if (blocks[i].className.indexOf('matlab-code') != -1) {
						// apply prettyprint class, and set the language to MATLAB
						blocks[i].className = 'prettyprint lang-matlab';
					}
				}

				// apply highlighting
				prettyPrint();
			});
		});

		function RegisterMATLABLanguageHandlers() {
			//=RENDER_FILE= ./_main.js
		}
	});
})();
