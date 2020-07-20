function stats = applyStatsfun(problem, x, storedb, key, options, stats)
% Apply the statsfun function to a stats structure (for solvers).
%
% function stats = applyStatsfun(problem, x, storedb, key, options, stats)
%
% Applies the options.statsfun user supplied function (if it was provided)
% to the stats structure, and returns the (possibly) modified stats
% structure.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% Note: if statsfun accepts a store structure as input, this structure can
% be read but not modified (modifications will be lost) ; the store
% structure will contain the store.shared field.
%
% See also: applyHook

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, April 3, 2013.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.
%
%   July 19, 2020 (NB):
%       Creates a field called 'hooked' in stats structures, for applyHook.

    if isfield(options, 'statsfun')
        
        switch nargin(options.statsfun)
            case 3
                stats = options.statsfun(problem, x, stats);
            case 4
                % Obtain, pass along, and save the store for x.
                % get/setWithShared must come in pairs.
                store = storedb.getWithShared(key);
                stats = options.statsfun(problem, x, stats, store);
                storedb.setWithShared(store, key);
            otherwise
                warning('manopt:statsfun', ...
                        'statsfun unused: wrong number of inputs');
        end
    end
    
    % Always create a field called 'hooked' for the benefit of applyHook.
    % This is to be set to either true or false by applyHook.
    stats.hooked = NaN;

end
