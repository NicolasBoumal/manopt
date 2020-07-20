function [newx, newkey, info, hooked] = applyHook(problem, x, storedb, key, options, info, last)
% Apply the hook function to possibly replace the current x (for solvers).
%
% function [newx, newkey, info, hooked] = 
%                  applyHook(problem, x, storedb, key, options, info, last)
%
% Applies the options.hook user-supplied function (if there is one) to the
% current x of a solver. If this leads to a change, than the boolean
% 'hooked' is true, and newx, newkey are different from x, key. Otherwise,
% newx, newkey are equal to x, key, and the boolean 'hooked' is false.
%
% storedb is a StoreDB object, key is the StoreDB key to point x; likewise
% for newkey and newx.
%
% info and last work the same way as in stoppingcriterion.
%
% The hook is called at the beginning of each iteration, after saving the
% stats information, but before evaluating stopping criteria. Time spent in
% the hook is included in the solver's reported computation time.
%
% This function takes care of logging the boolean 'hooked' in the info
% struct-array. (This requires the field 'hooked' to exist in the first
% place: applyStatsfun ensures this.)
%
% The options.hook function handle can have these prototypes:
%
%   [newx, hooked]                = hook(problem, x)
%   [newx, newkey, hooked]        = hook(problem, x, storedb, key)
%   [newx, newkey, hooked, stats] = hook(problem, x, storedb, key, stats)
%
% See also: applyStatsfun stoppingcriterion

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 19, 2020.
% Contributors: Eitan Levin
% Change log: 

    if isfield(options, 'hook')
        
        nin = nargin(options.hook);
        nout = nargout(options.hook);
        
        if nin == 2 && nout == 2
            [newx, hooked] = options.hook(problem, x);
            if hooked
                storedb.remove(key);
                newkey = storedb.getNewKey();
            else
                newkey = key;
            end
        elseif nin == 4 && nout == 3
            [newx, newkey, hooked] = options.hook(problem, x, storedb, key);
            if hooked
                storedb.removefirstifdifferent(key, newkey);
            end
        elseif nin == 5 && nout == 4
            stats = info(last);
            [newx, newkey, hooked, stats] = ...
                             options.hook(problem, x, storedb, key, stats);
            info(last) = stats;
            if hooked
                storedb.removefirstifdifferent(key, newkey);
            end
        else
            newx = x;
            newkey = key;
            hooked = false;
            warning('manopt:hook', ...
                    'options.hook unused: wrong number of inputs/outputs');
        end
        
    else
        newx = x;
        newkey = key;
        hooked = false;
    end
    
    % Always register whether or not the point was hooked (i.e., changed).
    % This field is first created in applyStatsfun.
    stats = info(last);
    stats.hooked = hooked;
    info(last) = stats;

end
