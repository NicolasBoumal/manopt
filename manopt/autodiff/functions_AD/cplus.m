function plusAB = cplus(A,B)
% Computes the sum of A and B
%
% function plusAB = cplus(A,B)
%
% Returns the sum of A and B. This function can be seen as A+B but is
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
        plusAB.real = realA + B.real;
        plusAB.imag = imagA + B.imag;
        
    elseif iscstruct(A) && isnumeric(B)
        realB = real(B);
        imagB = imag(B);
        plusAB.real = realB + A.real;
        plusAB.imag = imagB + A.imag;
    
    elseif isnumeric(A) && isnumeric(B)
        plusAB = A + B;
    
    elseif iscstruct(A) && iscstruct(B)
        plusAB.real = A.real + B.real;
        plusAB.imag = A.imag + B.imag;
    
    else
        ME = MException('cplus:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end

end
