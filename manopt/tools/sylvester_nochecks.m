function X = sylvester_nochecks(A, B, C)
% Solve Sylvester equation without input checks.
%
% function X = sylvester_nochecks(A, B, C)
%
% Solves the Sylvester equation A*X + X*B = C, where A is a m-by-m matrix,
% B is a n-by-n matrix, and X and C are m-by-n matrices.
%
% This is a stripped-down version of Matlab's own sylvester function that
% bypasses any input checks. This is significantly faster for small m and
% n, which is often useful in Manopt.

    flag = 'real';
    if ~isreal(A) || ~isreal(B) || ~isreal(C)
        flag = 'complex';
    end

    [QA, TA] = schur(A, flag);
    [QB, TB] = schur(B, flag);
    
    % Solve Sylvester Equation TA*Y + Y*TB = QA'*C*QB.
    [Y, info] = builtin('_sylvester_tri', TA, TB, QA'*C*QB);
    
    if info == 1
        error(message('MATLAB:sylvester:solutionNotUnique'));
    end

    X = QA*Y*QB';
    
end
