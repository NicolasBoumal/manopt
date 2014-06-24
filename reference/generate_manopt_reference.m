clear all;
close all;
clc;

% Add m2html tool to the Matlab path
cd m2html/;
addpath(pwd);
cd ..;

% Go back to the main directory hosting manopt files.
cd ..;

% Setup options
options = struct();
options.verbose = 'on';
options.mFiles = {'manopt', 'examples'};
options.htmlDir = 'web/reference/';
options.recursive = 'on';
options.source = 'on';
options.download = 'off';
options.syntaxHighlighting = 'on';
options.tabs = 4;
options.globalHypertextLinks = 'on';
options.graph = 'on';
options.todo = 'off';
% options.load = 0;
% options.save = 'off';
options.search = 'off';
options.helptocxml = 'off';
options.indexFile = 'index';
options.extension = '.html';
options.template = 'blue';
options.ignoredDir = {'.svn', 'cvs', '.git'};
options.language = 'english';

% Call the magician
m2html(options);
