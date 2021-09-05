function Xnormfro = cnormsqfro(X)
% Computes the squared Frobenius norm of X
%
% function Xnormfro = cnormsqfro(X)
%
% Returns the squared Frobenius norm of X. This function can be seen as 
% norm(...,'fro')^2 but is compatible with dlarrays and structs with fields
% real and imag. Supports both real and complex numbers.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if iscstruct(X)
        Xnormfro = cinnerprodgeneral(X, X);
        
    elseif isnumeric(X)
        if isreal(X) 
            Xnormfro = X(:)'*X(:);
        else
            Xnormfro = sum(real(conj(X(:)).*X(:)));
        end
        
    else
        ME = MException('cnormsqfro:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end

end
