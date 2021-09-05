function miusAB = cminus(A,B)
% Computes the difference of A and B
%
% function miusAB = cminus(A,B)
%
% Returns the difference of A and B. This function can be seen as A-B but 
% is compatible with dlarrays and structs with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isnumeric(A) && iscstruct(B)
        realA = real(A);
        imagA = imag(A);
        miusAB.real = realA - B.real;
        miusAB.imag = imagA - B.imag;
        
    elseif iscstruct(A) && isnumeric(B)
        realB = real(B);
        imagB = imag(B);
        miusAB.real = realB - A.real;
        miusAB.imag = imagB - A.imag;
    
    elseif isnumeric(A) && isnumeric(B)
        miusAB = A - B;
    
    elseif iscstruct(A) && iscstruct(B)
        miusAB.real = A.real - B.real;
        miusAB.imag = A.imag - B.imag;
    
    else
        ME = MException('cminus:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end

end
