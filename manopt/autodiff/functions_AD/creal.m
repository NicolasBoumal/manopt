function Xreal = creal(X)
% Extracts the real part of x
%
% function Xreal = creal(X)
%
% Returns the real part of x. The input can be either a numeric array
% or a struct with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if iscstruct(X)
        Xreal = X.real;
    elseif isnumeric(X)
        Xreal = real(X);
    else
        ME = MException('creal:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end

end
