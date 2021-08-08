function Xrepmat = crepmat(X,varargin)

    if isstruct(X) && isfield(X,'real')
        Xrepmat.real = repmat(X.real,varargin{:});
        Xrepmat.imag = repmat(X.imag,varargin{:});
        
    elseif isnumeric(X)
        Xrepmat = repmat(X,varargin{:});
        
    else
        ME = MException('crespmat:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end
      

end