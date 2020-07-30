function X = sylvester_nochecks(A, B, C)
% Solve Sylvester equation without input checks.
%
% function X = sylvester_nochecks(A, B, C)
%
% Solves the Sylvester equation A*X + X*B = C, where A is an m-by-m matrix,
% B is an n-by-n matrix, and X and C are two m-by-n matrices.
%
% This is a stripped-down version of Matlab's own sylvester function that
% bypasses any input checks. This is significantly faster for small m and
% n, which is often useful in Manopt.
%
% See also: sylvester lyapunov_symmetric

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 19, 2018
% Contributors: This is a modification of Matlab's built-in sylvester.
% Change log: 
%   July 30, 2020 (NB):
%       Changed call from builtin('_sylvester_tri', ...) to
%       matlab.internal.math.sylvester_tri(...), which seems necessary for
%       more recent versions of Matlab.
%       Also had to remove the second output: the 'unique' flag.

    flag = 'real';
    if ~isreal(A) || ~isreal(B) || ~isreal(C)
        flag = 'complex';
    end

    [QA, TA] = schur(A, flag);
    [QB, TB] = schur(B, flag);
    
    % Solve Sylvester Equation TA*Y + Y*TB = QA'*C*QB.
    Y = matlab.internal.math.sylvester_tri(TA, TB, QA'*C*QB, ...
                                                     'I', 'I', 'notransp');
    % Use this call instead for older versions of Matlab:
    % [Y, info] = builtin('_sylvester_tri', TA, TB, QA'*C*QB);

    X = QA*Y*QB';
    
end
