function plusAB = cplus(A,B)

    if isnumeric(A) && isstruct(B) && isfield(B,'real')
        realA = real(A);
        imagA = imag(A);
        plusAB.real = realA + B.real;
        plusAB.imag = imagA + B.imag;
        
    elseif isstruct(A) && isnumeric(B) && isfield(A,'real')
        realB = real(B);
        imagB = imag(B);
        plusAB.real = realB + A.real;
        plusAB.imag = imagB + A.imag;
    
    elseif isnumeric(A) && isnumeric(B)
        plusAB = A+B;
    
    elseif isstruct(A) && isstruct(B) && isfield(A,'real') && isfield(B,'real')
        plusAB.real = A.real+B.real;
        plusAB.imag = A.imag+B.imag;
    
    else
        ME = MException('cplus:inputError', ...
        'Input does not have the expected format.');
        throw(ME);

end