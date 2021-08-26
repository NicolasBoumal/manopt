function traceA = ctrace(A)
    
    if isstruct(A) && isfield(A,'real')
        assert(size(A.real,1)==size(A.real,2),'Input should be a square matrix')
        n = size(A.real,1);
        realA = A.real;
        imagA = A.imag;
        traceA.real = sum(realA(1:n+1:end));
        traceA.imag = sum(imagA(1:n+1:end));
        
    elseif isnumeric(A)
        assert(size(A,1)==size(A,2),'Input should be a square matrix')
        n = size(A,1);
        traceA = sum(A(1:n+1:end));
    else
        ME = MException('ctrace:inputError', ...
        'Input does not have the expected format.');
        throw(ME);

end