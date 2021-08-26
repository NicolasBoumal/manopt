function Xreshape = creshape(X,varargin)

    if isstruct(X) && isfield(X,'real')
        Xreshape.real = reshape(X.real,varargin{:});
        Xreshape.imag = reshape(X.imag,varargin{:});
        
    elseif isnumeric(X)
        Xreshape = reshape(X,varargin{:});
        
    else
        ME = MException('creshape:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end
end