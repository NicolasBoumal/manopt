function storedb = setStore(problem, x, storedb, store)
% Updates the store struct. pertaining to a point in the storedb database.
%
% function storedb = setStore(problem, x, storedb, store)
%
% Updates the storedb database of structures such that the structure
% corresponding to the point x will be replaced by store. If there was no
% record for the point x, it is created and set to store. The updated
% storedb database is returned. The lastset field of the store structure
% will be set to the current time.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    assert(nargout == 1, ...
           'The output of setStore should replace your storedb.');
   
    % Construct the fieldname (key) associated to the current point x.
    key = problem.M.hash(x);
    
    % Set the value associated to that key to store.
    storedb.(key) = store;
    
    % Add / update a last-set flag
    storedb.(key).lastset = cputime();

end
