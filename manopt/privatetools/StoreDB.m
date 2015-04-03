classdef StoreDB < handle_light
% The StoreDB class is a handle class to handle caching in Manopt.
% TODO : deal with storedepth
    
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
        % Use get and set to interact with this.
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
        
        % Save the given store at the given key. If no key is provided, a
        % new key is generated for this store (i.e., it is assumed this
        % store pertains to a new point). The key is returned in all cases.
        function key = set(storedb, store, key)
            if nargin < 3
                key = getNewKey(storedb);
            end
            storedb.history.(key) = store;
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
