function storedb = purgeStoredb(storedb, storedepth)
% Makes sure the storedb database does not exceed some maximum size.
%
% function storedb = purgeStoredb(storedb, storedepth)
%
% Trim the store database storedb such that it contains at most storedepth
% elements (store structures). The 'lastset' field of the store structures
% is used to delete the oldest elements first.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    if storedepth <= 0
        storedb = struct();
        return;
    end

    % Get list of field names (keys)
    keys = fieldnames(storedb);
    nkeys = length(keys);
    
    % If we need to remove some of the elements in the database
    if nkeys > storedepth
    
        % Get current time (should be same source of time as that used in
        % setStore for the lastset flag).
        nowcpu = cputime();
        
        % Get the 'age' of each element since last set
        ages = zeros(nkeys, 1);
        for i = 1 : nkeys
            store = storedb.(keys{i});
            ages(i) = nowcpu - store.lastset;
        end
        
        % Sort the ages and determine the threshold above which the field
        % needs to be removed
        sortages = sort(ages, 1, 'ascend');
        maxage = sortages(storedepth);
        
        % Remove all fields that are too old
        storedb = rmfield(storedb, keys(ages > maxage));
        
    end
    
end
