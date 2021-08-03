function problem = preprocessautodiff(problem)
%%    
    if (isfield(problem,'egrad') && isfield(problem,'ehess'))..., 
            || (isfield(problem,'egrad') && isfield(problem,'hess'))...,
            || (isfield(problem,'grad') && isfield(problem,'ehess'))...,
            || (isfield(problem,'grad') && isfield(problem,'hess'))...,
            || (isfield(problem,'costgrad') && isfield(problem,'ehess'))...,
            || (isfield(problem,'costgrad') && isfield(problem,'hess'))
        return 
    elseif contains(problem.M.name(),'sparsity')
         warning(['Auto differentiation currently does not support '...
                    'sparse matrices']);
        return
    elseif ~(exist('dlarray', 'file') == 2)
        sprintf(['It seems the Deep learning tool box is not installed.', ...
         '\nIt is needed for automatic differentiation \n']);
        return
    else 
        complexflag = false;
        x = problem.M.rand();
        try
            dlx = mat2dl(x);
            cost_testdlx = problem.cost(dlx);
        catch ME
            if (strcmp(ME.identifier,'deep:dlarray:ComplexNotSupported'))
                try
                    dlx = mat2dl_complex(x);
                    cost_testx = problem.cost(x);
                    cost_testdlx = problem.cost(dlx);
                catch
                    warning(['Auto differentiation failed. '...
                    'Cost function contains complex numbers. Check if '...
                    'it works for both numerical arrays and structures with'...
                    ' fields real and imag']);
                    return
                end
                complexflag = true;
            else
                 warning(['Auto differentiation failed. '...
                    'Problem defining cost function. Check the list of '...
                    'functions with AD support on the following website.'...
                    'https://ww2.mathworks.cn/help/deeplearning/ug/'...
                    'list-of-functions-with-dlarray-support.html']);
                return   
            end
        end                   
    end
%%
    problem.autogradfunc = autograd(problem);
    problem.egrad = @(x) egradcompute(problem.autogradfunc,complexflag,x);
    problem.ehess = @(x,xdot,store) ehesscompute(problem,complexflag,x,xdot,store);
      
    try 
        egrad = problem.egrad(x);
    catch
        warning(['Auto differentiation failed. '...
                    'Problem defining cost function. Check the list of '...
                    'functions with AD support on the following website.'...
                    'https://ww2.mathworks.cn/help/deeplearning/ug/'...
                    'list-of-functions-with-dlarray-support.html']);
        problem = rmfield(problem,'egrad');
        problem = rmfield(problem,'ehess');
        return
    end
    
    warning(['It seems no gradient was provided. '...
                    'Auto differentiation is used to compute egrad and ehess']);
    
end