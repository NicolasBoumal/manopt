M = spherefactory(5);
x = M.rand();
v = M.randvec(x);
h = logspace(-20, 1, 251);
g = zeros(size(h));
for k = 1 : numel(h)
    g(k) = M.norm(x, M.log(x, M.exp(x, v, h(k))));
end
loglog(h, abs(g-h)./h);
xlabel('h');
title('Relative error: (norm(Log_x(Exp(x, h*v))) - h)/h');