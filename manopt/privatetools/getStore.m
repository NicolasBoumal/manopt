function store = getStore(problem, x, storedb)
% Extracts a store struct. pertaining to a point from the storedb database.
%
% function store = getStore(problem, x, storedb)
%
% Queries the storedb database of structures (itself a structure) and
% returns the store structure corresponding to the point x. If there is no
% record for the point x, returns an empty structure.
%
% That structure is then complemented with a field called 'permanent'. That
% field can also be modified, and will be the same regardless of x. In
% other words: it makes it possible to store and pass information around
% between different points.
%
% See also: setStore purgeStoredb

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log:
%
%   April 2, 2015, NB:
%       storedb may now contain a special field, called 'permanent', which
%       is then added to the returned store. This special field is not
%       associated to a specific store; it is passed around from call to
%       call, to create a 'permanent memory' that spans, for example, all
%       iterations of a solver's execution.
   
    % Construct the fieldname (key) associated to the queried point x.
    key = problem.M.hash(x);
    
    % If there is a value stored for this key, extract it.
    % Otherwise, create an empty structure.
    if isfield(storedb, key)
        store = storedb.(key);
    else
        store = struct();
    end
    
    % If there is a permanent memory, add it.
    % Otherwise, add an empty structure.
    if isfield(storedb, 'permanent')
        store.permanent = storedb.permanent;
    else
        store.permanent = struct();
    end

end
