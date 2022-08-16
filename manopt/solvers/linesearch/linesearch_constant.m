function [stepsize, newx, newkey, lsstats] = ...
                  linesearch_constant(problem, x, d, ~, ~, ~, storedb, ~)
% Forces a constant multiplier on the descent direction chosen by the algorithm.
% 
% This is meant to be used by the steepestdescent or conjugategradients solvers.
% To use this method, specify linesearch_constant as an option, and your chosen
% constant alpha > 0 in the problem structure, as follows:
%
%  problem.linesearch = @(x, d) 1.0;     % choose any positive real number
%  options.linesearch = @linesearch_constant;
%  x = steepestdescent(problem, [], options);
%
% The effective step (that is, the vector the optimization algorithm retracts)
% is constructed as alpha*d, and the step size is the norm of that vector.
% Thus: stepsize = alpha*norm_d.
% The step is executed by retracting the vector alpha*d from the current
% point x, which gives newx (the returned point).
% This line-search method does not require any cost function evaluations,
% as there is effectively no search involved.
%
% Inputs
%
%  problem : structure holding the description of the optimization problem
%  x : current point on the manifold problem.M
%  d : tangent vector at x (descent direction)
%  storedb : StoreDB object (handle class: passed by reference) for caching
%
%  storedb is optional.
%
% Outputs
%
%  stepsize : norm of the vector retracted to reach newx from x.
%  newx : next iterate using the constant stepsize, such that
%         the retraction at x of the vector alpha*d reaches newx.
%  newkey : key associated to newx in storedb
%  lsstats : statistics about the line-search procedure
%            (costevals, stepsize, alpha).
%
% See also: steepestdescent conjugategradients linesearch

% This file is part of Manopt: www.manopt.org.
% Original author: Victor Liao, June 13, 2022.
% Contributors: 
% Change log: 

    % Allow omission of storedb.
    if ~exist('storedb', 'var')
        storedb = StoreDB();
    end

    % Obtain user-specified alpha if it exists.
    % User should specify their intended alpha by:
    % problem.linesearch = @(x,d) alpha;
    if canGetLinesearch(problem)
        alpha = getLinesearch(problem, x, d);
    else
        alpha = 1; % default alpha value.
    end

    % Make the chosen step and compute the cost there.
    newx = problem.M.retr(x, d, alpha);
    newkey = storedb.getNewKey();
    
    % As seen outside this function, stepsize is the size of the vector we
    % retract to make the step from x to newx. Since the step is alpha*d:
    norm_d = problem.M.norm(x, d);
    stepsize = alpha * norm_d;
    
    % Return some statistics also, for possible analysis.
    % Return some statistics also, for possible analysis.
    lsstats.costevals = 0;
    lsstats.stepsize = stepsize;
    lsstats.alpha = alpha;
end
