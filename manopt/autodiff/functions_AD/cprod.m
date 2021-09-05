function prodAB = cprod(A,B)
% Computes the product of A and B
%
% function prodAB = cprod(A,B)
%
% Returns the product of A and B. This function can be seen as A*B but is
% compatible with dlarrays and structs with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isnumeric(A) && iscstruct(B)
        realA = real(A);
        imagA = imag(A);
        prodAB.real = realA*B.real - imagA*B.imag;
        prodAB.imag = realA*B.imag + imagA*B.real;
        
    elseif iscstruct(A) && isnumeric(B)
        realB = real(B);
        imagB = imag(B);
        prodAB.real = A.real*realB - A.imag*imagB;
        prodAB.imag = A.real*imagB + A.imag*realB;
        
    elseif isnumeric(A) && isnumeric(B)
        prodAB = A*B;
        
    elseif iscstruct(A) && iscstruct(B)
        prodAB.real = A.real*B.real - A.imag*B.imag;
        prodAB.imag = A.real*B.imag + A.imag*B.real;
        
    else
        ME = MException('cprod:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
        
    end

end
