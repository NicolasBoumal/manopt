
M = spherefactory(5, 10);

x = M.rand();
v = M.randvec(x);
t = .1;
y = M.retr(x, v, t);

u1 = M.randvec(x);
u2 = M.randvec(x);

Tu1 = M.transp(x, y, u1);
Tu2 = M.transp(x, y, u2);
fprintf('Probably not zero: %g\n', (M.inner(x, u1, u2) - M.inner(y, Tu1, Tu2))/M.inner(x, u1, u2));

Tu1 = M.isotransp(x, y, u1);
Tu2 = M.isotransp(x, y, u2);
fprintf('Should be zero: %g\n', (M.inner(x, u1, u2) - M.inner(y, Tu1, Tu2))/M.inner(x, u1, u2));
