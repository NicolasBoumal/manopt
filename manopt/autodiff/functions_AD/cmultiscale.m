function Ascale = cmultiscale(scale, A)
% Multiplies the 2D slices in a 3D matrix by individual scalars.
%
% function Ascale = cmultiscale(scale, A)
%
% Basically the same as multiscale but is compatible with dlarrays
% and structs with fields real and imag.
%
% See also: manoptAD, multiscale

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    assert(ndims(A) == 3, ...
           'cmultiscale is only well defined for matrix arrays of 3.');
    if isstruct(A) && isfield(A,'real')
        [~, ~, N] = size(A.real);
        assert(numel(scale) == N, ...
           ['scale must be a vector whose length equals the third ' ...
            'dimension of A, that is, the number of 2D matrix slices ' ...
            'in the 3D array A.']);
        Ascale = cdotprod(A,reshape(scale,1,1,N));
    elseif isnumeric(A)
        [~, ~, N] = size(A);
        assert(numel(scale) == N, ...
           ['scale must be a vector whose length equals the third ' ...
            'dimension of A, that is, the number of 2D matrix slices ' ...
            'in the 3D array A.']);
        Ascale = A.*reshape(scale,1,1,N);
    else
        ME = MException('cmultiscale:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end

end
