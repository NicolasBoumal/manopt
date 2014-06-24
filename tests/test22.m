function [cost, info, x, A] = test22(n)
% Test for the preconditioner support of RCG and RTR.
%
%

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 


    clc;
%     reset(RandStream.getDefaultStream);
%     randnfoo = randn(123456, 1); %#ok<NASGU>
    
    if ~exist('n', 'var') || isempty(n)
        n = 3000;
    end

    % Define the problem data : a random symmetric, positive definite
    % matrix with an ill-conditioned diagonal part.
    [Q, ~] = qr(randn(n));
    A = Q*diag(rand(n, 1))*Q';
    A = A + 150*diag(logspace(-3, 3, n));
%     A = -A;
    
    % Compute a preconditioner for A
%     P = diag(1./diag(A));
%     fprintf('Cond. of  A : %e\n', cond(A));
%     fprintf('Cond. of PA : %e\n', cond(A*P));
    
%     keyboard;
    
    % Create the problem structure
    M = spherefactory(n);
    problem.M = M;

    % The cost and gradient
    problem.costgrad = @costgrad;
    function [cost grad store] = costgrad(x, store)
        
        if ~isfield(store, 'cost')
            Ax = A*x;
            store.cost = -x'*Ax;
        end
        cost = store.cost;
        
        if ~isfield(store, 'grad')
            store.grad = -2*(Ax + cost*x);
        end
        grad = store.grad;
        
    end

    % The preconditioner, which is an approximation for the inverse of the
    % Hessian and should be a symmetric, positive definite linear operator.
    % See notes March 18, 2013
    problem.precon = @precon;
    function [Pu store] = precon(x, u, store)
        cost = store.cost;
%         approx_hess = 2*(diag(diag(A)) - cost*eye(n));
        approx_hess = 2*(-cost - diag(A));
        approx_hess(approx_hess < 1e-8) = 1;
%         if any(approx_hess < 0)
%             fprintf('oops\n');
%         end
%         Pu = M.proj(x, approx_hess\u);
        Pu = M.proj(x, u./approx_hess);
%         Pu = (-2*(A+cost*eye(n)))\u;
%         H = 2*(x*x'*A + (x'*A*x)*eye(n) - A)*(eye(n)-x*x');
%         keyboard;
%         Pu = H\u;
%         Pu = u;
%         Pu = M.proj(x, Pu);
%         hist(log10(approx_hess));
%         drawnow;
    end


    % If the optimization algorithms require Hessians, since we do not
    % provide it, it will go for a standard approximation of it. This line
    % tells Matlab not to issue a warning when this happens.
    warning('off', 'manopt:getHessian:approx');
    
    % Check gradient consistency.
%     checkgradient(problem);
    
    % Solve
    fprintf('\n ---- With preconditioning ---- \n');
    
    options = struct();
    options.maxinner = n;
    options.tolgradnorm = 1e-5;
    options.minstepsize = 1e-15;
    options.beta_type = 'H-Z';
    [x cost info] = trustregions(problem, [], options);
    fprintf('Inner work: %d\n', sum([info.numinner]));
    [x cost info] = conjugategradient(problem, [], options);

    fprintf('\n ---- Without preconditioning ---- \n');
    problem = rmfield(problem, 'precon');
    options.maxinner = n;
    [x cost info] = trustregions(problem, [], options);
    fprintf('Inner work: %d\n', sum([info.numinner]));
    [x cost info] = conjugategradient(problem, [], options);

% keyboard;
    
end
