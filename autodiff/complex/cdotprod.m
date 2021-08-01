function cdotprodAB = cdotprod(A,B)

    if ~isstruct(A) && isstruct(B) && isfield(B,'real')
        realA = real(A);
        imagA = imag(A);
        cdotprodAB.real = realA.*B.real-imagA.*B.imag;
        cdotprodAB.imag = realA.*B.imag+imagA.*B.real;
        
    elseif isstruct(A) && ~isstruct(B) && isfield(A,'real')
        realB = real(B);
        imagB = imag(B);
        cdotprodAB.real = A.real.*realB-A.imag.*imagB;
        cdotprodAB.imag = A.real.*imagB+A.imag.*realB;
        
    elseif ~isstruct(A) && ~isstruct(B)
        cdotprodAB.real = real(A.*B);
        cdotprodAB.imag = imag(A.*B);
        
    elseif isstruct(A) && isstruct(B) && isfield(A,'real') && isfield(B,'real')
        cdotprodAB.real = A.real.*B.real-A.imag.*B.imag;
        cdotprodAB.imag = A.real.*B.imag+A.imag.*B.real;
        
    else
        ME = MException('cprod:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end