function [Q, R] = stiefelgeneralized_retraction_MGS_twice(B, X, U, t)
% Retraction for generalized Stiefel based on Modified Gram-Schmidt.
% More accurate but twice as slow as stiefelgeneralized_retraction_MGS.
% NB, Dec. 16, 2018.
    if ~exist('t', 'var') || isempty(t)
        t = 1;
    end
    [Q1, R1] = stiefelgeneralized_retraction_MGS(B, X, U, t);
    [Q2, R2] = stiefelgeneralized_retraction_MGS(B, Q1, zeros(size(U)), 0);
    Q = Q2;
    R = R2*R1;
end
