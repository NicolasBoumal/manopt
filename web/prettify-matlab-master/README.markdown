MATLAB syntax highlighting for google code prettify
===================================================

This script provides MATLAB syntax highlighting for the
[google-code-prettify][1] project. Intended to be used on [Stack Overflow][2]
and other Stack Exchange sites.

Currently the code recognizes the following constructs:

 - single line as well as block comments (`% this is a comment`)
 - quoted string (`'hello world'`)
 - number literals (`1`, `-2.5`, `1i`, `2.9E-5`, etc...)
 - system commands (`!touch filename`)
 - line continuation characters (`...`)
 - transpose operator (single quote) vs. strings
 - command prompt/command output (`>> now`)
 - error/warning messages (`??? Error in ...` and `Warning: ...`)
 - parentheses, brackets, braces (`()`, `[]`, `{}`)
 - other operators (`<>=~@&;,:!-+*^.|\/`)
 - MATLAB language keywords (`if`, `else`, `end`, etc...)
 - some special variables and constants (`inf`, `nan`, `varargin`, etc..)
 - over 1300 builtin functions from core MATLAB (`cos`, `plot`, etc...)
 - additional functions from popular toolboxes ([Statistics][6], [IPT][7], and [Optimization][8])
 - user-defined indentifiers (function and variable names not matched in previous steps)

See the wiki for sample screenshots.


BUILDING
--------

If the files don't already exist, run `rake SO:build` to create the output
javascript files in the `js` directory using the templates sources from `src`.
(This step requires Rake, a Make-like build tool for Ruby, used here to provide
basic template processing).


Stack Overflow
--------------

To obtain MATLAB syntax highlighting on Stack Overflow, install the
`js/prettify-matlab.user.js` userscript with your preferred browser (see [this
page][3] for some instructions).

The script is only activated on questions tagged [`matlab`][4].

In addition, a separate userscript is included to allow switching the language
used by the prettifier. It adds a small button to the top-right corner of each
code block, with an attached drop-down menu to allow language selection.
To add this functionality, install the `js/switch-lang.user.js` userscript.


MATLAB Answers / File Exchange
------------------------------

Similarly, you can apply the syntax hightighting on both [MATLAB Answers][5] 
and [File Exchange][9] websites from MathWorks. Simply install the userscripts
`js/prettify-mathworks-answers.user.js` or `js/prettify-mathworks-fileexchange.user.js`
respectively.


google-code-prettify extension
------------------------------

To apply the MALTAB syntax highlighting on code snippets in your own web pages,
first include the prettify scripts and stylesheets in your document (as explained
in the [prettify][1] project documentation). To use the MATLAB language extension,
include the `js/lang-matlab.js` script, and place your source code inside a
preformatted HTML tag as follows:

    <pre class="prettyprint lang-matlab">
        <code>
       	% example code
       	x = [1,2,3];
       	sum(x.^2)
        </code>
    </pre>

Upon calling `prettyPrint()`, this will automatically be pretty printed, and the
default styles will be applied. You can customize them with your own, or use the
provided CSS file in `css/lang-matlab.css` which resembles the color scheme of
the MATLAB IDE (with some additions of my own).

Check the `demo/index.html` file for a demonstration.


[1]: http://code.google.com/p/google-code-prettify/
[2]: http://stackoverflow.com/
[3]: http://stackapps.com/tags/script/info
[4]: http://stackoverflow.com/questions/tagged/matlab
[5]: http://www.mathworks.com/matlabcentral/answers/
[6]: http://www.mathworks.com/products/statistics/
[7]: http://www.mathworks.com/products/image/
[8]: http://www.mathworks.com/products/optimization/
[9]: http://www.mathworks.com/matlabcentral/fileexchange/

LICENSE
-------

Copyright (c) 2012 by Amro &lt;amroamroamro@gmail.com&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
