/*
	PR_PLAIN: plain text
	PR_STRING: string literals
	PR_KEYWORD: keywords
	PR_COMMENT: comments
	PR_TYPE: types
	PR_LITERAL: literal values (1, null, true, ..)
	PR_PUNCTUATION: punctuation string
	PR_SOURCE: embedded source
	PR_DECLARATION: markup declaration such as a DOCTYPE
	PR_TAG: sgml tag
	PR_ATTRIB_NAME: sgml attribute name
	PR_ATTRIB_VALUE: sgml attribute value
*/
var PR_IDENTIFIER = "ident",
	PR_CONSTANT = "const",
	PR_FUNCTION = "fun",
	PR_FUNCTION_TOOLBOX = "fun_tbx",
	PR_SYSCMD = "syscmd",
	PR_CODE_OUTPUT = "codeoutput",
	PR_ERROR = "err",
	PR_WARNING = "wrn",
	PR_TRANSPOSE = "transpose",
	PR_LINE_CONTINUATION = "linecont";

// Refer to: http://www.mathworks.com/help/techdoc/ref/f16-6011.html
var coreFunctions = [
	//=INSERT_FILE_QUOTED_CONCATED= ./functions/core.txt
].join("|");
var statsFunctions = [
	//=INSERT_FILE_QUOTED_CONCATED= ./functions/stats.txt
].join("|");
var imageFunctions = [
	//=INSERT_FILE_QUOTED_CONCATED= ./functions/image.txt
].join("|");
var optimFunctions = [
	//=INSERT_FILE_QUOTED_CONCATED= ./functions/optim.txt
].join("|");

// identifiers: variable/function name, or a chain of variable names joined by dots (obj.method, struct.field1.field2, etc..)
// valid variable names (start with letter, and contains letters, digits, and underscores).
// we match "xx.yy" as a whole so that if "xx" is plain and "yy" is not, we dont get a false positive for "yy"
//var reIdent = '(?:[a-zA-Z][a-zA-Z0-9_]*)';
//var reIdentChain = '(?:' + reIdent + '(?:\.' + reIdent + ')*' + ')';

// patterns that always start with a known character. Must have a shortcut string.
var shortcutStylePatterns = [
	// whitespaces: space, tab, carriage return, line feed, line tab, form-feed, non-break space
	[PR.PR_PLAIN, /^[ \t\r\n\v\f\xA0]+/, null, " \t\r\n\u000b\u000c\u00a0"],

	// block comments
	//TODO: chokes on nested block comments
	//TODO: false positives when the lines with %{ and %} contain non-spaces
	//[PR.PR_COMMENT, /^%(?:[^\{].*|\{(?:%|%*[^\}%])*(?:\}+%?)?)/, null],
	[PR.PR_COMMENT, /^%\{[^%]*%+(?:[^\}%][^%]*%+)*\}/, null],

	// single-line comments
	[PR.PR_COMMENT, /^%[^\r\n]*/, null, "%"],

	// system commands
	[PR_SYSCMD, /^![^\r\n]*/, null, "!"]
];

// patterns that will be tried in order if the shortcut ones fail. May have shortcuts.
var fallthroughStylePatterns = [
	// line continuation
	[PR_LINE_CONTINUATION, /^\.\.\.\s*[\r\n]/, null],

	// error message
	[PR_ERROR, /^\?\?\? [^\r\n]*/, null],

	// warning message
	[PR_WARNING, /^Warning: [^\r\n]*/, null],

	// command prompt/output
	//[PR_CODE_OUTPUT, /^>>\s+[^\r\n]*[\r\n]{1,2}[^=]*=[^\r\n]*[\r\n]{1,2}[^\r\n]*/, null],		// full command output (both loose/compact format): `>> EXP\nVAR =\n VAL`
	[PR_CODE_OUTPUT, /^>>\s+/, null],			// only the command prompt `>> `
	[PR_CODE_OUTPUT, /^octave:\d+>\s+/, null],	// Octave command prompt `octave:1> `

	// identifier (chain) or closing-parenthesis/brace/bracket, and IS followed by transpose operator
	// this way we dont misdetect the transpose operator ' as the start of a string
	["lang-matlab-operators", /^((?:[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*|\)|\]|\}|\.)')/, null],

	// identifier (chain), and NOT followed by transpose operator
	// this must come AFTER the "is followed by transpose" step (otherwise it chops the last char of identifier)
	["lang-matlab-identifiers", /^([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*)(?!')/, null],

	// single-quoted strings: allow for escaping with '', no multilines
	//[PR.PR_STRING, /(?:(?<=(?:\(|\[|\{|\s|=|;|,|:))|^)'(?:[^']|'')*'(?=(?:\)|\]|\}|\s|=|;|,|:|~|<|>|&|-|\+|\*|\.|\^|\|))/, null],	// string vs. transpose (check before/after context using negative/positive lookbehind/lookahead)
	[PR.PR_STRING, /^'(?:[^']|'')*'/, null],	// "'"

	// floating point numbers: 1, 1.0, 1i, -1.1E-1
	[PR.PR_LITERAL, /^[+\-]?\.?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?[ij]?/, null],

	// parentheses, braces, brackets
	[PR.PR_TAG, /^(?:\{|\}|\(|\)|\[|\])/, null],	// "{}()[]"

	// other operators
	[PR.PR_PUNCTUATION, /^(?:<|>|=|~|@|&|;|,|:|!|\-|\+|\*|\^|\.|\||\\|\/)/, null]
];

var identifiersPatterns = [
	// list of keywords (`iskeyword`)
	[PR.PR_KEYWORD, /^\b(?:break|case|catch|classdef|continue|else|elseif|end|for|function|global|if|otherwise|parfor|persistent|return|spmd|switch|try|while)\b/, null],

	// some specials variables/constants
	[PR_CONSTANT, /^\b(?:true|false|inf|Inf|nan|NaN|eps|pi|ans|nargin|nargout|varargin|varargout)\b/, null],

	// some data types
	[PR.PR_TYPE, /^\b(?:cell|struct|char|double|single|logical|u?int(?:8|16|32|64)|sparse)\b/, null],

	// commonly used builtin functions from core MATLAB and a few popular toolboxes
	[PR_FUNCTION, new RegExp('^\\b(?:' + coreFunctions + ')\\b'), null],
	[PR_FUNCTION_TOOLBOX, new RegExp('^\\b(?:' + statsFunctions + ')\\b'), null],
	[PR_FUNCTION_TOOLBOX, new RegExp('^\\b(?:' + imageFunctions + ')\\b'), null],
	[PR_FUNCTION_TOOLBOX, new RegExp('^\\b(?:' + optimFunctions + ')\\b'), null],

	// plain identifier (user-defined variable/function name)
	[PR_IDENTIFIER, /^[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*/, null]
];

var operatorsPatterns = [
	// forward to identifiers to match
	["lang-matlab-identifiers", /^([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*)/, null],

	// parentheses, braces, brackets
	[PR.PR_TAG, /^(?:\{|\}|\(|\)|\[|\])/, null],	// "{}()[]"

	// other operators
	[PR.PR_PUNCTUATION, /^(?:<|>|=|~|@|&|;|,|:|!|\-|\+|\*|\^|\.|\||\\|\/)/, null],

	// transpose operators
	[PR_TRANSPOSE, /^'/, null]
];

PR.registerLangHandler(
	PR.createSimpleLexer([], identifiersPatterns),
	["matlab-identifiers"]
);
PR.registerLangHandler(
	PR.createSimpleLexer([], operatorsPatterns),
	["matlab-operators"]
);
PR.registerLangHandler(
	PR.createSimpleLexer(shortcutStylePatterns, fallthroughStylePatterns),
	["matlab"]
);
