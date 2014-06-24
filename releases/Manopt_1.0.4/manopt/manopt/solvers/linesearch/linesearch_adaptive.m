function [stepsize newx storedb lsmem lsstats] = ...
       linesearch_adaptive(problem, x, d, f0, df0, options, storedb, lsmem)
% Adaptive line search algorithm (step size selection) for descent methods.
%
% function [stepsize newx storedb lsmem lsstats] = 
%      linesearch_adaptive(problem, x, d, f0, df0, options, storedb, lsmem)
%
% Adaptive linesearch algorithm for descent methods, based on a simple
% backtracking method. On average, this line search intends to do only one
% or two cost evaluations. If the direction provided is not a descent
% direction, as indicated by a positive df0, the search direction will be
% reversed.
%
% Inputs/Outputs : see help for linesearch
%
% See also : linesearch

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 


    % Backtracking parameters
    contraction_factor = .5;
    suff_decr = 0.5; 1e-4;
    max_ls_steps = 10;
    
    % If we didn't get a descent direction, reverse it.
    if df0 > 0
        if options.verbosity >= 3
            fprintf(['Line search warning: got an ascent direction ' ...
                     '(df0 = %2e), went the other way.\n'], df0);
        end
        df0 = -df0;
        d = problem.M.lincomb(x, -1, d);
    end
    
    
    % Initial guess for the step size
    if ~isempty(lsmem)
        initial_stepsize =  lsmem.initial_stepsize;
    else
        % The initial choice of stepsize is necessarily disputable. The
        % optimal step size is invariant under rescaling of the cost
        % function, but df0, on the other hand, will scale like f. Hence,
        % using df0 to guess a stepsize may be considered ill-advised. It
        % is not so if one further assumes that f is "well conditionned".
        % At any rate, to prevent this initial step size guess from
        % exploding, we limit it to 1 (arbitrarily).
        initial_stepsize = min(abs(df0), 1);
    end
    
    % Backtrack
    stepsize = initial_stepsize;
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
    
    % On average we intend to do only one extra cost evaluation
    if iter == 1
        lsmem.initial_stepsize = 2 * initial_stepsize;
    elseif iter == 2
        lsmem.initial_stepsize = stepsize;
    else
        lsmem.initial_stepsize = 2 * stepsize;
    end
    
    % Save some statistics also, for possible analysis.
    lsstats.costevals = iter;
    lsstats.stepsize = stepsize;
    
end
