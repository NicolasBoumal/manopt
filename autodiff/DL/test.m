n=10;
A = randn(n);
A = .5*(A+A');
x = dlarray(randn(n,1));
xdot = dlarray(randn(n,1));

manifold = spherefactory(n);
mycostfunction = @(x) -x'*(A*x);

problem.cost = mycostfunction; 
problem.M = manifold; 
ecompute = @(x) subautograd(problem,x);
dlfunc = dlaccelerate(ecompute);
egrad = dlfeval(dlfunc,x);
c = @(x) b(egrad,x,xdot);
u = dlfeval(c,x)


function egrad = subautograd(problem,x)
    mycostfunction = problem.cost;
    y = mycostfunction(x);
    egrad = dlgradient(y,x,'RetainData',true,'EnableHigherDerivatives',true);
end

function u = b(egrad,x,xdot)
    temp = sum(egrad.*xdot,'all');
    u = dlgradient(temp,x,'RetainData',true,'EnableHigherDerivatives',true);
end
