% Compiles weingarten_omega.c into a binary mex file.
% Requires Matlab to have access to a compatible C compiler.
% If need be, run 'mex -setup' for help installing a compiler.

% This file is part of Manopt: www.manopt.org.
% Original author: Michael Psenka, Nov. 24, 2020.
% Contributors: Nicolas Boumal
% Change log:

mex -lmwlapack -lmwblas -largeArrayDims weingarten_omega.c 
