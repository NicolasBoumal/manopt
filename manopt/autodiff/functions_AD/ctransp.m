function Xtransp = ctransp(X)
% Computes the conjugate-transpose of X
%
% function Xtransp = ctransp(X)
%
% This function can be seen as the operator ' but is compatible with
% both dlarrays and structs with fields real and iamg.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:

    if iscstruct(X)
        Xreal = X.real;
        Ximag = X.imag;
        Xtransp.real = Xreal';
        Xtransp.imag = -Ximag';
    elseif isnumeric(X)
        Xtransp = X';
    else
        ME = MException('ctransp:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end

end
