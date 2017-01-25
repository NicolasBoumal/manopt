function problem_critpt = criticalpointfinder(problem)

    problem_critpt.M = problem.M;
    problem_critpt.costgrad = @costgrad;
    problem_critpt.approxhess = @approxhess;
    
    function [f, g] = costgrad(x, storedb, key)
        
        [~, grad] = getCostGrad(problem, x, storedb, key);
        Hess_grad = getHessian(problem, x, grad, storedb, key);
        
        f = .5*problem.M.norm(x, grad)^2;
        g = Hess_grad;
        
    end
    
    % This is not quite the Hessian because there should be a third-order
    % derivative term (which is innaccessible), but: at critical points
    % (where grad f(x) = 0 for the f of problem.cost) this Hessian is
    % exact, so it will allow for superlinear convergence.
    function HHu = approxhess(x, u, storedb, key)
        
        Hu  = getHessian(problem, x, u,  storedb, key);
        HHu = getHessian(problem, x, Hu, storedb, key);
        
    end

end
