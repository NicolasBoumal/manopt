function Xreal = creal(X)
    
    if isstruct(X) && isfield(X,'real')
        Xreal = X.real;
    elseif isnumeric(X)
        Xreal = real(X);
    else
        ME = MException('creal:inputError', ...
        'Input does not have the expected format.');
        throw(ME);


end