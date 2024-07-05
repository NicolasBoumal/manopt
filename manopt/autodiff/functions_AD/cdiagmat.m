function X = cdiagmat(x)
% Maps vector x to diagonal entries of a diagonal matrix X, for manoptAD.
%
% function X = cdiagmat(x)
%
% Given a vector x of length n, outputs a matrix X of size n-by-n whose
% diagonal entries are those of x, and whose off-diagonal entries are zero.
% 
% This provides a replacement to Matlab's built-in X = diag(x) in a way
% that is compatible with automatic differentiation, see manoptAD.
%
% To map a matrix to a vector of its diagonal entries, see cdiag.
%
% See also: manoptAD manoptADhelp cdiag

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 5, 2024
% Contributors:
% Change log:

    if iscstruct(x)
        assert(isvector(x.real) && isvector(x.imag), ...
               'Input should encode a vector.');
        assert(numel(x.real) == numel(x.imag), ...
               'Real and imaginary parts should have the same length.');

        n = numel(x.real);
        X.real = dlarray(zeros(n, n));      % Call to dlarray is necessary.
        X.real(1:(n+1):end) = x.real(:);
        X.imag = dlarray(zeros(n, n));
        X.imag(1:(n+1):end) = x.imag(:);

    elseif isnumeric(x)
        assert(isvector(x), 'Input should be a vector.');

        n = numel(x);
        X = dlarray(zeros(n, n));           % Call to dlarray is necessary.
        X(1:(n+1):end) = x(:);

    else
        ME = MException('cdiagmat:inputError', ...
                        'Input does not have the expected format.');
        throw(ME);
    end    

end
