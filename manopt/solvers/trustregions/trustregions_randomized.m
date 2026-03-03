function [x, cost, info, options] = trustregions_randomized(problem, x0, options)
% Riemannian trust-regions solver for optimization on manifolds, randomized
%
% function [x, cost, info, options] = trustregions_randomized(problem)
% function [x, cost, info, options] = trustregions_randomized(problem, x0)
% function [x, cost, info, options] = trustregions_randomized(problem, x0, options)
% function [x, cost, info, options] = trustregions_randomized(problem, [], options)
%
% This calls the Riemannian Trust-Regions solver in Manopt, with options
% set for a randomized version of RTR, designed to escape saddle points.
%
% See also: trustregions trs_tCG_randomized

% This file is part of Manopt: www.manopt.org.
% Original authors: Nicolas Boumal and Radu Dragomir, 2024--2026.
% Contributors: Xiaowen Jiang and Bonan Sun
% Change log: 

    if ~exist('x0', 'var')
        x0 = [];
    end

    if ~exist('options', 'var')
        options = struct();
    end

    % Force the subproblem solver to be the randomized one.
    if isfield(options, 'subproblemsolver')
        warning('manopt:rtrrandomized:trs', ...
                'Overwriting options.subproblemsolver.');
    end
    options.subproblemsolver = @trs_tCG_randomized;

    % If unset, set miniter to a positive number to make sure the
    % randomized algorithm has a chance to escape if it is initialized at
    % or near a saddle point (but let the user force any value).
    if ~isfield(options, 'miniter')
        options.miniter = 3;
    end

    % If unset, set mininner to zero (because the randomization should be
    % good enough to do the right thing), but let the user force any value.
    if ~isfield(options, 'mininner')
        options.mininner = 0;
    end

    % If unset, disable rho regularization because the randomization
    % mechanism has its own regularization built in, but let the user
    % decide if they want to add extra regularization on top of that.
    if ~isfield(options, 'rho_regularization')
        options.rho_regularization = 0;
    end

    % If unset, allow cost function value to increase from x_k to x_{k+1}
    % because randomization sometimes uses that to escape saddle points.
    % (But let the user override.)
    if ~isfield(options, 'allowcostincrease')
        options.allowcostincrease = true;
    end

    [x, cost, info, options] = trustregions(problem, x0, options);

end
