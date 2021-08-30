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

    if isstruct(A) && isfield(A,'real')
        [~, ~, N] = size(A.real);
        Ascale = cdotprod(A,reshape(scale,1,1,N));
    elseif isnumeric(A)
        [~, ~, N] = size(A);
        Ascale = A.*reshape(scale,1,1,N);
    else
        ME = MException('cmultiscale:inputError', ...
        'Input does not have the expected format.');
        throw(ME);
    end

end