function x = TTeMPS_randn(r, n)
% TTEMPS_RANDN Create random TTeMPS tensor
%   X = TTEMPS_RANDN(R, N) creates a length(N)-dimensional TTeMPS tensor
%   of size N(1)*N(2)*...N(end) with ranks R by filling the cores with
%   iid standard Gaussian random numbers.
%   Note that the first and last entry of the rank vector must be 1.
%

%   This is a trivial modification of TTeMPS_rand, part of TTeMPS.
%
%   See also TTeMPS_rand
    
    if length(r) ~= length(n)+1
        error('Size mismatch in arguments')
    end

    U = cell(1, length(n));
    for i = 1:length(n)
        U{i} = randn(r(i), n(i), r(i+1));
    end
    x = TTeMPS(U);
end
