function Xtransp = ctransp(X)

    if isstruct(X) && isfield(X,'real')
        Xreal = X.real;
        Ximag = X.imag;
        Xtransp.real = Xreal';
        Xtransp.imag = -Ximag';
    elseif isnumeric(X)
        Xtransp = X';
    else
        ME = MException('ctransp:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end

end