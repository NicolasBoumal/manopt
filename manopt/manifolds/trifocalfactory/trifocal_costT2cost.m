function val = trifocal_costT2cost(X, costT)
% Cost evaluation at X given function handle in the trifocal tensor T.
%
% function val = trifocal_costT2cost(X, costT)
%
% costT is the function handle for the cost function in T.
%
% See also: trifocal_egradT2egrad, trifocal_ehessT2rhess


val = costT(trifocal_getTensor(X));


end