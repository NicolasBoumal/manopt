function [stepsize, newx, newkey, lsstats] = ...
                  linesearch_constant(problem, x, d, ~, ~, ~, storedb, ~)
% Constant stepsize (no line-search) algorithm for descent methods.
% 
% Note: to use linesearch_constant the user should define their intended 
% constant alpha > 0 and use the following lines of code:
%  problem.linesearch = @(x, d) alpha;
%  options.linesearch = @linesearch_constant;
%
% Below, the step is constructed as alpha*d, and the step size is the norm
% of that vector, thus: stepsize = alpha*norm_d. The step is executed by
% retracting the vector alpha*d from the current point x, giving newx.
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
%            (stepsize).
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
    lsstats.stepsize = stepsize;
end
