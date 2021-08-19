function Xcol = cmat2col(X)

    if isstruct(X) && isfield(X,'real')
        Xreal = X.real;
        Ximag = X.imag;
        Xcol.real = Xreal(:);
        Xcol.imag = Ximag(:);
    elseif isnumeric(X)
        Xcol = X(:);
    else
        ME = MException('cmat2col:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end



end