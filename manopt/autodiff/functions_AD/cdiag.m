function diagX = cdiag(X)
% Extracts the diagonal elements of A.
%
% function diagX = cdiag(X)
%
% Returns the diagonal elements of A. The input A does not necessarily
% to be a square matrix. The function supports both numeric arrays and 
% structs with fields real and imag. Note that diag currently does
% not support dlarrays and cdiag can be seen as a backup function.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:

    if iscstruct(X)
        assert(length(size(X.real)) == 2, 'Input should be a 2-D array')
        m = size(X.real,1);
        n = size(X.real,2);
        realX = X.real;
        imagX = X.imag;
        if n >= m
            diagX.real = realX(1:m+1:m^2);
            diagX.imag = imagx(1:m+1:m^2);
        else
            diagX.real = realX(1:m+1:m*n-m+n);
            diagX.imag = imagX(1:m+1:m*n-m+n);
        end

    elseif isnumeric(X)
        assert(length(size(X)) == 2, 'Input should be a 2-D array')
        m = size(X,1);
        n = size(X,2);
        if n >= m
            diagX = X(1:m+1:m^2);
        else
            diagX = X(1:m+1:m*n-m+n);
        end

    else
        ME = MException('cdiag:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end    

end
