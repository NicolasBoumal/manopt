function traceA = ctrace(A)
    
    if isstruct(A) && isfield(A,'real')
        assert(length(size(A.real))==2,'Input should be a 2-D array')
        m = size(A.real,1);
        n = size(A.real,2);
        realA = A.real;
        imagA = A.imag;
        
        if n >= m
            traceA.real = sum(realA(1:m+1:m^2));
            traceA.imag = sum(imagA(1:m+1:m^2));
        else
            traceA.real = sum(realA(1:m+1:m*n-m+n));
            traceA.imag = sum(imagA(1:m+1:m*n-m+n));
        end
        
    elseif isnumeric(A)
        assert(length(size(A))==2,'Input should be a 2-D array')
        m = size(A,1);
        n = size(A,2);
        if n >= m
            traceA = sum(A(1:m+1:m^2));
        else
            traceA = sum(A(1:m+1:m*n-m+n));
        end

    else
        ME = MException('ctrace:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end

end