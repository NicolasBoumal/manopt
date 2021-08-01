function traceA = ctrace(A)
    
    assert(size(A,1)==size(A,2),'Input should be a square matrix')
    if isstruct(A) && isfield(A,'real')
        n = size(A.real,1);
        realA = A.real;
        imagA = A.imag;
        traceA.real=0;
        traceA.imag=0;
        for i = 1:n
            traceA.real = traceA.real + realA(i,i);
            traceA.imag = traceA.imag + imagA(i,i);
        end
    elseif isnumeric(A)
        n = size(A,1);
        traceA = 0;
        for i = 1:n
            traceA = traceA + A(i,i);
        end
    else
        ME = MException('ctrace:inputError', ...
        'Input does not have the expected format.');
        throw(ME);

end