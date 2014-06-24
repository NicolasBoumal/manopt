function [stepsize newx storedb lsmem lsstats] = ...
                linesearch(problem, x, d, f0, df0, options, storedb, lsmem)
% Standard line search algorithm (step size selection) for descent methods.
%
% function [stepsize newx storedb lsmem lsstats] = 
%               linesearch(problem, x, d, f0, df0, options, storedb, lsmem)
%
% Crude linesearch algorithm for descent methods, based on a simple
% backtracking method. If the direction provided is not a descent
% direction, as indicated by a positive df0, the search direction will be
% reversed.
%
% Inputs
%
%  problem : optimization problem description structure
%  x : current point on the manifold
%  d : tangent vector at x with norm 1 (descent direction)
%  f0 : cost value at x
%  df0 : directional derivative at x along d
%  options : options structure used by the solver
%  storedb : store database structure for caching purposes
%  lsmem : a special memory holder to give the linesearch the opportunity
%          to "remember" what happened in the previous calls.
%
% Outputs
%
%  stepsize : step size proposed by the line search algorithm.
%  newx : next iterate suggested by the line search algorithm, such that
%         the retraction at x of the vector stepsize*d reaches newx (unless
%         d was not a descent direction, in which case it would be
%         -stepsize*d, but that is not the normal usage for this function).
%  storedb : the (possibly updated) store database structure.
%  lsmem : the (possibly updated) lsmem memory holder.
%  lsstats : statistics about the line search procedure (stepsize, number
%            of trials etc).
% 
% About lsmem : it can be anything, and will typically be a matrix or a
%               structure. When first calling the line search, it should be
%               passed as the empty matrix []. For subsequent calls
%               (pertaining to the same solver call), the previously
%               returned lsmem should be passed as input. This memory
%               holder gives the line search a chance to exploit knowledge
%               of previous decisions to make new decisions.
%
% See also: linesearch_adaptive

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


% TODO: implement nice linesearchs from Nocedal&Wright (pp 56 sq.)
% TODO: rename this with explicit reference to the type of algorithm in use
% TODO: give control over the parameters via the options structure.


    % Backtracking parameters
    contraction_factor = .5;
    suff_decr = 1e-4;
    max_ls_steps = 25;
    
    % If we didn't get a descent direction, reverse it.
    if df0 > 0
        if options.verbosity >= 3
            fprintf(['Line search warning: got an ascent direction ' ...
                     '(df0 = %2e), went the other way.\n'], df0);
        end
        df0 = -df0;
        d = problem.M.lincomb(x, -1, d);
    end
    
    % Initial guess for the step size, as inspired from Nocedal&Wright, p59
    if isstruct(lsmem) && isfield(lsmem, 'f0')
        % Pick initial stepsize based on where we were last time,
        stepsize = 2*(f0-lsmem.f0)/df0;
        % and go look a little further, just in case.
        stepsize = stepsize / contraction_factor;
        % In case this gave a zero stepsize, try to force a larger one.
        if stepsize <= eps
            stepsize = abs(df0);
        end
    else
        % The initial choice of stepsize is necessarily disputable. The
        % optimal step size is invariant under rescaling of the cost
        % function, but df0, on the other hand, will scale like f. Hence,
        % using df0 to guess a stepsize may be considered ill-advised. It
        % is not so if one further assumes that f is "well conditionned".
        % At any rate, to prevent this initial step size guess from
        % exploding, we limit it to 1 (arbitrarily).
        stepsize = min(abs(df0), 1);
    end
    
    % Backtrack
    for iter = 1 : max_ls_steps
        
        % Look further down the line
        newx = problem.M.retr(x, d, stepsize);
        [newf storedb] = getCost(problem, newx, storedb);
        
        % If that point is not satisfactory, reduce the stepsize and retry.
        if newf > f0 + suff_decr*stepsize*df0
            stepsize = contraction_factor * stepsize;
            
        % Otherwise, stop here.
        else
            break;
        end
        
    end
    
    % If we got here without obtaining a decrease, we reject the step.
    if newf > f0
        stepsize = 0;
        newx = x;
    end

    % Save the situtation faced now so that at the next iteration, we will
    % know something about the previous decision.
    lsmem.f0 = f0;
    lsmem.df0 = df0;
    lsmem.stepsize = stepsize;
    
    % Save some statistics also, for possible analysis.
    lsstats.costevals = iter;
    lsstats.stepsize = stepsize;
    
end
