function storedb = setStore(problem, x, storedb, store)
% Updates the store struct. pertaining to a point in the storedb database.
%
% function storedb = setStore(problem, x, storedb, store)
%
% Updates the storedb database of structures such that the structure
% corresponding to the point x will be replaced by store. If there was no
% record for the point x, it is created and set to store. The updated
% storedb database is returned. The lastset__ field of the store structure
% keeps track of which stores were updated latest.
%
% If store contains a field called "permanent", its contents are placed in
% storedb.permanent, and the field is removed from the store structre
% before storage.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   Dec. 6, 2013, NB:
%       Now using a persistent uint32 counter instead of cputime to track
%       the most recently modified stores.
%
%   April 2, 2015, NB:
%       storedb may now contain a special field, called 'permanent', which
%       is updated from the given store. This special field is not
%       associated to a specific store; it is passed around from call to
%       call, to create a 'permanent memory' that spans, for example, all
%       iterations of a solver's execution.

    % This persistent counter is used to keep track of the order in which
    % store structures are updated. This is used by purgeStoredb to erase
    % the least recently useful store structures first when garbage
    % collecting.
    persistent counter;
    if isempty(counter)
        counter = uint32(0);
    end

    assert(nargout == 1, ...
           'The output of setStore should replace your storedb.');
   
    % If there is a permanent memory, extract it.
    if isfield(store, 'permanent')
        storedb.permanent = store.permanent;
        % Removing a field appears to be rather slow.
        % store = rmfield(store, 'permanent');
        % Instead, we simply empty the contents:
        store.permanent = struct();
    end
    
    % Add / update a last-set flag.
    store.lastset__ = counter;
    counter = counter + 1;
       
    % Construct the fieldname (key) associated to the current point x.
    key = problem.M.hash(x);
    
    % Put the store in storage at that key.
    storedb.(key) = store;

end
