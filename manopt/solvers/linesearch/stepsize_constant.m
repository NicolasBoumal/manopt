function [stepsize, newx, newkey, lsstats] = ...
                  stepsize_constant(problem, x, d, ~, ~, options, storedb, ~)
% Constant stepsize algorithm for descent methods
%
% function [stepsize, newx, newkey, lsstats] = 
%            stepsize_constant(problem, x, d, ~, ~, options, storedb, ~)
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
%  f0 : cost value at x
%  df0 : directional derivative at x along d
%  options : options structure (see in code for usage):
%     stepsize_init (0.01)
%         The constant stepsize alpha that is used at every iteration.
%  storedb : StoreDB object (handle class: passed by reference) for caching
%
%  options and storedb are optional.
%
% Outputs
%
%  stepsize : norm of the vector retracted to reach newx from x.
%  newx : next iterate using the constant stepsize, such that
%         the retraction at x of the vector alpha*d reaches newx.
%  newkey : key associated to newx in storedb
%  lsstats : statistics about the line-search procedure
%            (stepsize, number of trials etc).
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

    % Constant stepsize default parameters. These can be overwritten in the
    % options structure which is passed to the solver.
    default_options.stepsize_init = 0.01;

    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(default_options, options);
    
    alpha = options.stepsize_init;

    % Make the chosen step and compute the cost there.
    newx = problem.M.retr(x, d, alpha);
    newkey = storedb.getNewKey();
    cost_evaluations = 1;
    
    % As seen outside this function, stepsize is the size of the vector we
    % retract to make the step from x to newx. Since the step is alpha*d:
    norm_d = problem.M.norm(x, d);
    stepsize = alpha * norm_d;
    
    % Return some statistics also, for possible analysis.
    lsstats.costevals = cost_evaluations;
    lsstats.stepsize = stepsize;
    lsstats.alpha = alpha;
end
