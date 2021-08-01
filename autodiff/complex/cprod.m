function prodAB = cprod(A,B)

    if ~isstruct(A) && isstruct(B) && isfield(B,'real')
        realA = real(A);
        imagA = imag(A);
        prodAB.real = realA*B.real-imagA*B.imag;
        prodAB.imag = realA*B.imag+imagA*B.real;
        
    elseif isstruct(A) && ~isstruct(B) && isfield(A,'real')
        realB = real(B);
        imagB = imag(B);
        prodAB.real = A.real*realB-A.imag*imagB;
        prodAB.imag = A.real*imagB+A.imag*realB;
        
    elseif ~isstruct(A) && ~isstruct(B)
        prodAB.real = real(A*B);
        prodAB.imag = imag(A*B);
        
    elseif isstruct(A) && isstruct(B) && isfield(A,'real') && isfield(B,'real')
        prodAB.real = A.real*B.real-A.imag*B.imag;
        prodAB.imag = A.real*B.imag+A.imag*B.real;
        
    else
        ME = MException('cprod:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end