function Xreal = creal(X)
    
    if isstruct(X) && isfield(X,'real')
        Xreal = X.real;
    elseif isnumeric(X) && ~isdlarray(X)
        Xreal = real(X);
    elseif isdlarray(X)
        Xreal = X;
    else
        ME = MException('creal:inputError', ...
        'Input does not have the expected format.');
        throw(ME);


end