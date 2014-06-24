// ==UserScript==
// @name           StackOverflow: switch language of syntax highlighting
// @namespace      StackExchange_GoogleCodePrettify_SwitchLanguage
// @description    Allows switching the language of syntax highlighting on StackOverflow
// @author         Amro <amroamroamro@gmail.com>
// @version        1.0
// @license        MIT License
// @icon           http://cdn.sstatic.net/stackoverflow/img/favicon.ico
// @include        http://stackoverflow.com/questions/*
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

	// activate only on an actual question page (ignore question lists, tag pages, and such)
	if ( !/^\/questions\/(\d+|ask)/.test(window.location.pathname) ) {
		return;
	}

	// insert our custom CSS styles
	style_inject([
		//=INSERT_FILE_QUOTED= ../css/switch_lang.css
	].join(""));

	script_inject(function () {
		// add to onReady queue of SE (a stub for $.ready)
		StackExchange.ready(function () {
			add_language_selection_menu();
		});

		//=INSERT_FILE= ./_switch_lang.js
	});
})();
