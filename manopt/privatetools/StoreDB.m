classdef StoreDB < handle_light
% The StoreDB class is a handle class to manage caching in Manopt.

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, April 3, 2015.
% Contributors: 
% Change log: 

% TODO : deal with storedepth
% TODO : protect get/setWithShared calls: limit to one, and forbid access
%        to shared memory while it has not been returned.
%        Do think of the applyStatsFun case : calls a getWithShared, does
%        not need a setWithShared.
    
    properties(Access = public)
       
        % This memory is meant to be shared at all times. Users can modify
        % this at will. It is the same for all points x.
        shared = struct();
        
        % This memory is used by the toolbox for, e.g., automatic caching
        % and book keeping. Users should not overwrite this. It is the
        % same for all points x.
        internal = struct();
        
    end
    
    properties(Access = private)
        
        % This structure holds separate memories for individual points.
        % Use get and set to interact with this. The field name 'shared' is
        % reserved, for use with get/setWithShared.
        history = struct();
        
        % This internal counter is used to obtain unique key's for points.
        counter = uint32(0);
        
    end
    
    
    methods(Access = public)
        
%         % Constructor
%         function storedb = StoreDB()
%             % do nothing special.
%         end
        
        % Return the store associated to a given key.
        % If the key is unknown, returns an empty strucrure.
        function store = get(storedb, key)
            if isfield(storedb.history, key)
                store = storedb.history.(key);
            else
                store = struct();
            end
        end
        
        % Same as get, but adds the shared memory in store.shared.
        function store = getWithShared(storedb, key)
            store = storedb.get(key);
            store.shared = storedb.shared;
        end
        
        % Save the given store at the given key. If no key is provided, a
        % new key is generated for this store (i.e., it is assumed this
        % store pertains to a new point). The key is returned in all cases.
        function key = set(storedb, store, key)
            if nargin < 3
                key = getNewKey(storedb);
            end
            storedb.history.(key) = store;
        end
        
        % Same as set, but extracts the shared memory and saves it.
        % The stored store will still have a 'shared' field, but it will be
        % empty.
        function key = setWithShared(storedb, store, key)
            storedb.shared = store.shared;
            store.shared = [];
            key = storedb.set(store, key);
        end
        
        % Generate a unique key and return it. This should be called
        % everytime a new point is generated / stored. Keys are valid field
        % names for structures.
        function key = getNewKey(storedb)
            key = sprintf('z%d', storedb.counter);
            storedb.counter = storedb.counter + 1;
        end
        
    end
    
end
