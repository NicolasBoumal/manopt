function Ascale = cmultiscale(scale, A)
% Multiplies the 2D slices in a 3D array by individual scalars.
%
% function Ascale = cmultiscale(scale, A)
%
% Basically the same as multiscale but is compatible with structs with
% fields real and imag, which means this can be used for automatic
% differentiation with complex arrays in older Matlab versions.
%
% See also: manoptADhelp multiscale

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if isnumeric(A) && isnumeric(scale)
        
        Ascale = multiscale(scale, A);
        
    else
        
        A = tocstruct(A);
        scale = tocstruct(scale);
        
        assert(ndims(A.real) == 3, ...
           'cmultiscale is only well defined for 3D arrays.');
       
        [~, ~, N] = size(A.real);
        
        assert(numel(scale.real) == N, ...
           ['scale must be a vector whose length equals the third ' ...
            'dimension of A, that is, the number of 2D matrix slices ' ...
            'in the 3D array A. It can also be a struct with fields ' ... 
            'real and imag with size as stated above.']);
        
        scale.real = reshape(scale.real, 1, 1, N);
        scale.imag = reshape(scale.imag, 1, 1, N);
        Ascale = cdottimes(A, scale);

    end

end


% Test code
% n = 3; m = 5; N = 17;
% A = randn(n, m, N);
% scale = randn(N, 1);
% Z = multiscale(scale, A) - cmultiscale(scale, A);
% norm(Z(:))
% A = randn(n, m, N) + 1i*randn(n, m, N);
% Z = multiscale(scale, A) - cmultiscale(scale, A);
% norm(Z(:))
% scale = randn(N, 1) + 1i*randn(N, 1);
% Z = multiscale(scale, A) - cmultiscale(scale, A);
% norm(Z(:))
% B.real = real(A); B.imag = imag(A);
% Z = cmultiscale(scale, B);
% Zr = real(multiscale(scale, A)) - Z.real;
% Zi = imag(multiscale(scale, A)) - Z.imag;
% norm(Zr(:))
% norm(Zi(:))
% scalebis = tocstruct(scale);
% Z = cmultiscale(scalebis, B);
% Zr = real(multiscale(scale, A)) - Z.real;
% Zi = imag(multiscale(scale, A)) - Z.imag;
% norm(Zr(:))
% norm(Zi(:))
