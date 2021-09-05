function Xcol = cmat2col(X)
% Converts X into a column vector
%
% function Xcol = cmat2col(X)
%
% Returns X(:) where the input X can be either a numeric array or a 
% struct with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if iscstruct(X)
        Xreal = X.real;
        Ximag = X.imag;
        Xcol.real = Xreal(:);
        Xcol.imag = Ximag(:);
    elseif isnumeric(X)
        Xcol = X(:);
    else
        ME = MException('cmat2col:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end
    
end
