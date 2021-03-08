function [B, d_2, d_1] = doubly_stochastic_general(A, p, q, maxiter, checkperiod)
% Project a matrix to the doubly stochastic matrices (Sinkhorn's algorithm)
%
% function [B, d_2, d_1] = doubly_stochastic_general(A, p, q)
% function [B, d_2, d_1] = doubly_stochastic_general(A, p, q, maxiter)
% function [B, d_2, d_1] = doubly_stochastic_general(A, p, q, [], checkperiod)
% function [B, d_2, d_1] = doubly_stochastic_general(A, p, q, maxiter, checkperiod)
%
% Given an element-wise non-negative matrix A of size nxm, returns a
% matrix B of size nxn by applying Sinkhorn's algorithm
% to A.
% 
% maxiter (optional):
%    Strictly positive integer representing the maximum 
%    number of iterations of Sinkhorn's algorithm. 
%    The default value of maxiter is n*m.
% checkperiod (optional):
%    Only check stopping criteria every checkperiod iterations,
%    to reduce computational burden.
%
% The file is based on developments in the research paper
% Philip A. Knight, "The Sinkhorn–Knopp Algorithm: Convergence and 
% Applications" in SIAM Journal on Matrix Analysis and Applications 30(1), 
% 261-275, 2008.
%
% Please cite the Manopt paper as well as the above research paper.
% 
%
% See also doubly_stochastic

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Oct 30, 2020.
% Contributors:
% Change log:

    tol = eps;

    n = size(A, 1);
    m = size(A, 2);
    
    if ~exist('p', 'var') || isempty(p)
        p = (1/n)*ones(n,1);
    end

    if ~exist('q', 'var') || isempty(q)
        q = (1/m)*ones(m,1);
    end

    if ~exist('maxiter', 'var') || isempty(maxiter)
        maxiter = n*m;
    end
    
    if ~exist('checkperiod', 'var') || isempty(checkperiod)
        checkperiod = 100;
    end
        
    C = A;
    

    iter = 0;
    d_1 = (q')./sum(C); % row vector of size m
    d_2 = p./(C * d_1.'); % column vector of size n
    gap = inf;
    while iter < maxiter
        iter = iter + 1;
        row = d_2.' * C;

        % Check gap condition only at checkperiod intervals.
        % It saves computations for large-scale scenarios.
        if mod(iter, checkperiod) == 0 
            gap = max(abs(row .* d_1 - q'));
            if isnan(gap)
                break;
            end
            if gap <= tol
                break;
            end
        end
        
        d_1_prev = d_1;
        d_2_prev = d_2;

        d_1 = (q')./row;
        d_2 = p./(C * d_1.');

        if any(isinf(d_2)) || any(isnan(d_2)) || any(isinf(d_1)) || any(isnan(d_1))
            warning('DoublyStochasticGeneralProjection:NanInfEncountered', ...
                    'Nan or Inf occured at iter %d \n', iter);
            d_1 = d_1_prev;
            d_2 = d_2_prev;
            break;
        end
    end
    
    B = C .* (d_2 * d_1);

    fprintf('Iter %d, gap %e \n', iter, gap);

end
