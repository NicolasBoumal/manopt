function cdottimesAB = cdottimes(A, B)
% Computes the element-wise multiplication between A and B
%
% function cdottimesAB = cdottimes(A, B)
%
% Returns the element-wise multiplication of A and B. The inputs A and B 
% can be either numeric arrays or structs with fields real and imag.
%
% See also: manoptAD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isnumeric(A) && isnumeric(B)
        cdottimesAB = A .* B;
        
    elseif iscstruct(A) || iscstruct(B)
        A = tocstruct(A);
        B = tocstruct(B);
        cdottimesAB.real = A.real .* B.real - A.imag .* B.imag;
        cdottimesAB.imag = A.real .* B.imag + A.imag .* B.real;
        
    else
        ME = MException('cprod:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
        
    end

end
