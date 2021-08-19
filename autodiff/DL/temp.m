n = 10000;
A = randn(n);
A = .5*(A+A');

manifold = spherefactory(n);
mycostfunction = @(x) -x'*(A*x);

problem.cost = mycostfunction; 
problem.M = manifold;   

x = manifold.rand();
store = struct();
autohessfunc = autohess(problem);
%tic;
%autogradfunc = autograd(problem);
%testgrad(A,manifold,problem,store,true);
%toc;



% time1 = timeit(@() testgrad(A,manifold,autogradfunc,true))
% time2 = timeit(@() testgrad(A,manifold,autogradfunc,false))
% proportion = time1 / time2

% x = manifold.rand();
% xdot = manifold.randvec(x);
% store = struct();
% [ehess,store] = ehesscompute_new(problem,x,xdot,store);

% time1 = timeit(@() testhess(A,manifold,x,store,problem,true))
% time2 = timeit(@() testhess(A,manifold,x,store,problem,false))
% proportion = time1 / time2
tic;
testhess(manifold,problem,store,true);
toc;

function testgrad(A,manifold,problem,store,egrad)
    
    if egrad == true
        for i = 1:200
            x = manifold.rand();
            z = egradcompute_new(problem,x,store);
            store = struct();
        end
    else
        for i = 1:200
            x = manifold.rand();
            z = -2*(A*x);
        end
    end
end

function testhess(manifold,problem,store,ehess)
    
    if ehess == true
        for i = 1:200
            x = manifold.rand();
            xdot = manifold.randvec(x);
            [ehess,store] = ehesscompute_new(problem,x,xdot,store);
            store = struct();
            fprintf('%dth',i);
        end
    else
        for i = 1:500
            xdot = manifold.randvec(x);
            z = -2*(A*xdot);
        end
    end
end

