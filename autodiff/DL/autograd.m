function autogradfunc = autograd(problem)

    costfunction = problem.cost;
    
    function [y egrad] = autogradfuncinternel(costfunction,x)
       
        y = costfunction(x);
        if isstruct(y) && isfield(y,'real')
            y = creal(y);
        end
        egrad = dlgradient(y,x,'RetainData',false,'EnableHigherDerivatives',false);
        
        if (contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors'))
        A = problem.M.A;
        egrad(:, :, A) = 0;
        end
    end
    
    func = @(x) autogradfuncinternel(costfunction,x);
    autogradfunc = dlaccelerate(func);
    clearCache(autogradfunc);
    
end