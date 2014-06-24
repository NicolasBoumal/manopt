// REFERENCE: http://userscripts.org/scripts/show/71052
// REFERENCE: http://davidwalsh.name/twitter-dropdown-jquery
function add_language_selection_menu() {
	"use strict";

	// we require jQuery to be already loaded in the page
	if (typeof jQuery == 'undefined') { return; }

	// languages CSS classes
	var languages = ["default", "lang-html", "lang-c", "lang-java", "lang-cs",
		"lang-sh", "lang-pl", "lang-py", "lang-rb", "lang-js", "lang-matlab"];

	// return closure
	var makeClickHandler = function (pre, lang) {
		// create closure
		return function (e) {
			// remove existing formatting inside <code> tag, by setting content to plain text again
			unprettify(pre.children('code'));

			// set new prettify class
			pre.removeClass();
			pre.addClass("prettyprint " + lang);

			// change language indicated
			$(this).parent().prev(".pp-lang-link").children("span").text(lang);

			// hide languge menu
			//$(this).parent().slideToggle();

			// re-apply syntax highlighting
			prettyPrint();

			// stop default link-clicking behaviour
			e.preventDefault();
		};
	};

	// go through each <pre> block, and add language selection menu
	$("pre.prettyprint").each(function () {
		// <pre> block
		var code = $(this);

		// current language used
		var currLang = $.trim(this.className.replace('prettyprint', ''));
		if (!currLang) { currLang = "default"; }

		// create <div> of language selector button
		var button = $('<div class="pp-lang-button" title="choose language"></div>');

		// create and add toggle link
		var link = $('<a class="pp-lang-link"><span>' + currLang + '</span></a>').appendTo(button);

		// create dropmenu and add to button
		var menu = $('<div class="pp-lang-menu"></div>').appendTo(button);

		// transparency animation on hover
		button.hover(
			function () { $(this).animate({opacity: 1.0}, 'fast'); },
			function () { $(this).animate({opacity: 0.7}); }
		);

		// button is click event
		link.click(function (e) {
			// set button as active/non-active
			$(this).toggleClass('pp-link-active');

			// show/hide menu
			$(this).next('.pp-lang-menu').slideToggle();

			// stop default link-clicking behaviour
			e.preventDefault();
		});

		// populate it with entries for every language
		for (var i = 0; i < languages.length; i++) {
			// create link, hook up the click event, and add it to menu
			$('<a title="set language to: ' + languages[i] + '">' + languages[i] + '</a>')
				.css({'cursor': 'pointer', 'font-size': 'small'})
				.click( makeClickHandler(code, languages[i]) ).appendTo(menu);
		}

		// add button to DOM just before the <pre> block
		button.insertBefore(code);
	});
}

function unprettify(codeNode) {
	// Note: el.innerHTML, el.textContent vs. $(el).html(), $(el).text()
	var code = $(codeNode);
	var encodedStr = code.html().replace(/<br[^>]*>/g, "\n").replace(/&nbsp;/g, " ");	// html encoded
	var decodedStr = $("<div/>").html(encodedStr).text();	// decode html entities
	code.text(decodedStr);		// text() replaces special characters like `<` with `&lt;`
}

/*
<div class="pp-lang-button">
	<a class="pp-lang-link"><span>Language</span></a>
	<div class="pp-lang-menu">
		<a></a>
		<a></a>
	</div>
</div>
<pre class="prettyprint">
	<code></code>
</pre>
*/
