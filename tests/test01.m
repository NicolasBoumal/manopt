function [cost, x, A, info] = test01(n, usestore)
% function [cost, x, A] = test01(n, usestore)
% All intputs are optional.
%
% Typical call:
%
% profile clear; profile on;
% test1(10000, true);
% profile off; profile report;
%
% If activated (search for 'work!' in the code):
% 'work!' is printed each time a matrix-vector product with A is computed.
% Observe how setting 'usestore' to true of false affects the number of
% products.
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    clc;
    reset(RandStream.getDefaultStream);
    randnfoo = randn(123456, 1); %#ok<NASGU>
    
    if ~exist('n', 'var') || isempty(n)
        n = 1042;
    end
    
    if ~exist('usestore', 'var') || isempty(usestore)
        usestore = true;
    end

    % Define the problem data
%     A = randn(n);
    A = magic(n)/n^3;
    A = (A+A')/2;
    
    % Create the problem structure
    problem.M = spherefactory(n);
    
    if usestore
        % These functions use the store capability
        problem.cost = @(x, store)    objective(A, x, store);
        problem.grad = @(x, store)    gradient (A, x, store);
        problem.hess = @(x, h, store) hessian  (A, x, h, store);
    else
        % These functions do not use the store capability
%         problem.cost = @(x)    objective(A, x, struct());
%         problem.grad = @(x)    gradient (A, x, struct());
        problem.costgrad = @(x) costgrad(A, x);
        problem.hess = @(x, h) hessian  (A, x, h, struct());
    end
    
    % Check consistency of cost, grad and hess.
    debug = 0;
    if debug
        checkgradient(problem);
        pause;
        checkhessian(problem);
        pause;
    end

    % Define a few (optional) options
    options.maxtime = 20; % [seconds]
    options.storedepth = 25;
    options.tolgradnorm = 1e-8;
    options.maxinner = 200;
    
%     options.stopfun = @stopfun;
    function stop = stopfun(problem, x, info, last)
        if mod(last, 150) == 0
            plot([info.time], [info.cost]);
            xlim([0 options.maxtime]);
            drawnow;
        end
        stop = false;
    end

    options.statsfun = @statsfun;
    function stats = statsfun(problem, x, stats)
        stats.pt = problem.M.hash(x);
    end


    % Solve
    x0 = (1:n)'; x0 = x0/norm(x0);
%     options.linesearch = @linesearch;
    options.linesearch = @linesearch_adaptive;
%     options.ls_max_steps = 10;
%     [x cost info] = steepestdescent(problem, x0, options);
    [x cost info] = conjugategradient(problem, x0, options);
    [x cost info] = trustregions(problem, [], options);
%     [x cost info] = pso(problem, [], options);
%     [x cost info] = neldermead(problem, [], options);

%     figure;
%     subplot(1, 2, 1);
%     semilogy([info.iter], [info.gradnorm], '.-');
%     subplot(1, 2, 2);
%     semilogy([info.time], [info.gradnorm], '.-');
    
    if isfield(info, 'linesearch')
        figure;
        lsstats = [info.linesearch];
        lscostevals = [lsstats.costevals];
        hist(lscostevals, min(lscostevals):max(lscostevals));
        title('Histogram of cost evaluations per line search');
%         keyboard;
    end
    
%     keyboard;
    
end

function [val store] = objective(A, x, store)

    if ~isfield(store, 'Ax')
        store.Ax = A*x; % disp('work!');
    end
    Ax = store.Ax;
    
    if ~isfield(store, 'val')
        store.val = -.5*(x'*Ax);
    end
    
    val = store.val;
    
end

function [grad store] = gradient(A, x, store)

    if ~isfield(store, 'Ax') || ~isfield(store, 'val')
        [~, store] = objective(A, x, store);
    end
    Ax = store.Ax;
    val = store.val;
    
    grad = -(2*val*x + Ax);
    
end

function [hess store] = hessian(A, x, h, store)

    if ~isfield(store, 'val')
        [~, store] = objective(A, x, store);
    end
    val = store.val;
    
    Ah = A*h; % disp('work!');
    hess = -(2*val*h + Ah);
    hess = hess - (x'*hess)*x;         % projection
    
end

function [cost grad] = costgrad(A, x)
    Ax = A*x;
    cost = -.5*(x'*Ax);
    if nargout == 2
        grad = -(2*cost*x + Ax);
    end
end