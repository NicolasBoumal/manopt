% M = obliquecomplexfactory(5, 10, false);
% M = obliquefactory(5, 10, false);
% M = spherecomplexfactory(5, 10);
% M = spheresymmetricfactory(5);
% M = complexcirclefactory(5);
M = realphasefactory(5);

X = M.rand();
U = M.randvec(X);
U = U / M.norm(X, U);
t = 1e-9;
Y = M.exp(X, U, t);

abs(M.dist(X, Y) - t)/t

abs(M.norm(X, M.log(X, Y)) - t)/t