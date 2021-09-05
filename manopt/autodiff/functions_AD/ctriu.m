function Xtriu = ctriu(X, k)
% Extracts the upper triangular part of X.
%
% function Xtriu = ctriu(X,k)
%
% This function can be seen as triu(X,k) but is compatible with dlarrays
% and structs with fields real and imag.
%
% See also: manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:

    switch nargin
        case 1
            if iscstruct(X)
                index0 = find(triu(ones(size(X.real))) == 0);
                Xtriu = X;
                Xtriu.real(index0) = 0;
                Xtriu.imag(index0) = 0;
        
            elseif isnumeric(X) && ~isdlarray(X)
                Xtriu = triu(X);
            
            elseif isdlarray(X)
                Xtriu = dlarray(zeros(size(X)));
                index1 = find(triu(ones(size(X))) == 1);
                Xtriu(index1) = X(index1);
                
            else
                ME = MException('ctriu:inputError', ...
                                'Input does not have the expected format.');
                throw(ME);
            end
        case 2
            if iscstruct(X)
                index0 = find(triu(ones(size(X.real)), k) == 0);
                Xtriu = X;
                Xtriu.real(index0) = 0;
                Xtriu.imag(index0) = 0;
        
            elseif isnumeric(X) && ~isdlarray(X)
                Xtriu = triu(X,k);
            
            elseif isdlarray(X)
                Xtriu = dlarray(zeros(size(X)));
                index1 = find(triu(ones(size(X)), k) == 1);
                Xtriu(index1) = X(index1);
                
            else
                ME = MException('ctriu:inputError', ...
                                'Input does not have the expected format.');
                throw(ME);
            end
    otherwise
        error('Too many input arguments.');
    end
end
