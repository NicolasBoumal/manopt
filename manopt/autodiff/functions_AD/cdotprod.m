function cdotprodAB = cdotprod(A,B)
% Computes the element-wise multiplication between A and B
%
% function cdotprodAB = cdotprod(A,B)
%
% Returns the element-wise multiplication of A and B. The input A and B 
% can be either a numeric array or a struct with fields real and imag.
%
% See also: manoptAD

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isnumeric(A) && isstruct(B) && isfield(B,'real')
        realA = real(A);
        imagA = imag(A);
        cdotprodAB.real = realA.*B.real-imagA.*B.imag;
        cdotprodAB.imag = realA.*B.imag+imagA.*B.real;
        
    elseif isstruct(A) && isnumeric(B) && isfield(A,'real')
        realB = real(B);
        imagB = imag(B);
        cdotprodAB.real = A.real.*realB-A.imag.*imagB;
        cdotprodAB.imag = A.real.*imagB+A.imag.*realB;
        
    elseif isnumeric(A) && isnumeric(B)
        cdotprodAB = A.*B;
        
    elseif isstruct(A) && isstruct(B) && isfield(A,'real') && isfield(B,'real')
        cdotprodAB.real = A.real.*B.real-A.imag.*B.imag;
        cdotprodAB.imag = A.real.*B.imag+A.imag.*B.real;
        
    else
        ME = MException('cprod:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
        
    end

end