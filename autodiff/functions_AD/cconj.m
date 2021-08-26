function xconj = cconj(x)
    
    if isstruct(x) && isfield(x,'real')
        xconj.real = x.real;
        xconj.imag = -x.imag;
    elseif isnumeric(x)
        xconj = conj(x);
    else
        ME = MException('cconj:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end






end