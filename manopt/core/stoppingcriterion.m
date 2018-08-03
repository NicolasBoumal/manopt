function [stop, reason] = stoppingcriterion(problem, x, options, info, last)
% Checks for standard stopping criteria, as a helper to solvers.
%
% function [stop, reason] = stoppingcriterion(problem, x, options, info, last)
%
% Executes standard stopping criterion checks, based on what is defined in
% the info(last) stats structure and in the options structure.
%
% The returned number 'stop' is 0 if none of the stopping criteria
% triggered, and a (strictly) positive integer otherwise. The integer
% identifies which criterion triggered:
%  0 : Nothing triggered;
%  1 : Cost tolerance reached;
%  2 : Gradient norm tolerance reached;
%  3 : Max time exceeded;
%  4 : Max iteration count reached;
%  6 : User defined stopfun criterion triggered.
%
% The output 'reason' is a string describing the triggered event.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   Apr. 2, 2015 (NB):
%       'reason' now contains the option (name and value) that triggered.
%
%   Aug. 3, 2018 (NB):
%       Removed check for costevals, as it was never used, and the new
%       manopt counters allow to do this in a more transparent way.
%       Furthermore, now, options.stopfun can have 1 or 2 outputs: the
%       first is a boolean indicating whether or not to stop, and the
%       (optional) second output is a string indicating the reason.


    stop = 0;
    reason = '';
    
    stats = info(last);

    % Target cost attained
    if isfield(stats, 'cost') && isfield(options, 'tolcost') && ...
       stats.cost <= options.tolcost
        reason = sprintf('Cost tolerance reached; options.tolcost = %g.', options.tolcost);
        stop = 1;
        return;
    end

    % Target gradient norm attained
    if isfield(stats, 'gradnorm') && isfield(options, 'tolgradnorm') && ...
       stats.gradnorm < options.tolgradnorm
        reason = sprintf('Gradient norm tolerance reached; options.tolgradnorm = %g.', options.tolgradnorm);
        stop = 2;
        return;
    end

    % Allotted time exceeded
    if isfield(stats, 'time') && isfield(options, 'maxtime') && ...
       stats.time >= options.maxtime
        reason = sprintf('Max time exceeded; options.maxtime = %g.', options.maxtime);
        stop = 3;
        return;
    end

    % Allotted iteration count exceeded
    if isfield(stats, 'iter') && isfield(options, 'maxiter') && ...
       stats.iter >= options.maxiter
        reason = sprintf('Max iteration count reached; options.maxiter = %g.', options.maxiter);
        stop = 4;
        return;
    end

    % Check whether the possibly user defined stopping criterion
    % triggers or not.
    if isfield(options, 'stopfun')
        % options.stopfun can have 1 or 2 outputs, but typically we cannot
        % check this using nargout because it will often be an anonymous
        % function. Thus, we try with 2 outputs, and if it fails we try
        % again with 1.
        try
            [userstop, reason] = options.stopfun(problem, x, info, last);
        catch up
            % If the exception was indeed about the number of outputs...
            if strcmp(up.identifier, 'MATLAB:maxlhs')
                try
                    % Try again with a single output
                    userstop = options.stopfun(problem, x, info, last);
                    reason = ['User defined stopfun criterion triggered;' ...
                              ' see options.stopfun.'];
                catch e
                    % Something went wrong anyway...
                    warning('manopt:stoppingcriterion:stopfunoutputs', ...
                            'options.stopfun must have 1 or 2 outputs.');
                    rethrow(e);
                end
            % Otherwise, something else went wrong: pass it on.
            else
                rethrow(up);
            end
        end
        if userstop
            stop = 6;
            return;
        end
    end

end
