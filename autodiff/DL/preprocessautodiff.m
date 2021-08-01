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
    else 
        x = problem.M.rand();
        if isfield(problem,'complex')  %should be replaced by another indicator
            try
                dlx = mat2dl_complex(x);
                cost_test1 = problem.cost(x);
                cost_test2 = problem.cost(dlx);
            catch 
                warning(['Auto differentiation failed. '...
                    'Problem defining cost function.']);
                return
            end
        else
            try
                dlx = mat2dl(x);
                cost_test = problem.cost(dlx);
            catch 
                warning(['Auto differentiation failed. '...
                    'Problem defining cost function.']);
                return
            end
        end
    end
%%
    problem.autogradfunc = autograd(problem);
   
    if startsWith(problem.M.name(),'Rotations manifold SO')

        problem.egrad = @(x) egradcompute(problem.autogradfunc,x);
        problem.ehess = @(x,xdot,store) ehesscall_ambient(problem,x,xdot,store);
        
    elseif contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors')
        
        problem.egrad = @(x) egradcall_anchor(problem,x);
        problem.ehess = @(x,xdot,store) ehesscall_anchor(problem,x,xdot,store);
        
    elseif isfield(problem,'Xmat') && (problem.Xmat == true)
        
        problem.egrad = @(x) egradcompute(problem.autogradfunc,x);
        problem.ehess = @(x,H,store) ehesscallfixedrank(problem,x,H,store);
        
    elseif isfield(problem,'complex') %should be replaced by another indicator
        
        if startsWith(problem.M.name(),'Unitary manifold')
            
            problem.egrad = @(x) egradcompute_complex(problem.autogradfunc,x);
            problem.ehess = @(x,xdot,store) ehesscomputeambient_complex(problem,x,xdot,store);
        
        else
        
            problem.egrad = @(x) egradcompute_complex(problem.autogradfunc,x);
            problem.ehess = @(x,xdot,store) ehesscompute_complex(problem,x,xdot,store);
            
        end
        
    else
        
        problem.egrad = @(x) egradcompute(problem.autogradfunc,x);
        problem.ehess = @(x,xdot,store) ehesscompute(problem,x,xdot,store);
    
    end
    
    try 
        egrad = problem.egrad(x);
    catch
        warning(['Auto differentiation failed.'...
                    'Problem defining cost function.']);
        problem = rmfield(problem,'egrad');
        problem = rmfield(problem,'ehess');
        return
    end
    
    warning(['It seems no gradient was provided. '...
                    'Auto differentiation is used to compute egrad and ehess']);
    %%
        function [ehess,store] = ehesscall_ambient(problem,x,xdot,store)
            u = problem.M.tangent2ambient(x, xdot); 
            [ehess,store] = ehesscompute(problem,x,u,store);
        end
        
        function [ehess,store] = ehesscomputeambient_complex(problem,x,xdot,store)
            u = problem.M.tangent2ambient(x, xdot); 
            [ehess,store] = ehesscompute_complex(problem,x,u,store);
        end
        
        function [egrad,store] = egradcall_anchor(problem,x)
            A = problem.M.A;
            egrad = egradcompute(problem.autogradfunc,x);
            egrad(:, :, A) = 0;
        end
    
        function [ehess,store] = ehesscall_anchor(problem,x,xdot,store)
            A = problem.M.A;
            u = problem.M.tangent2ambient(x, xdot);  
            [ehess,store] = ehesscompute(problem,x,u,store);
            ehess(:, :, A) = 0;
        end
    
        function [ehess,store] = ehesscallfixedrank(problem,x,H,store) 
            ambient_H = problem.M.tangent2ambient(x, H);
            xdot = ambient_H.U*ambient_H.S*ambient_H.V';
            [ehess,store] = ehesscompute(problem,x,xdot,store);
        end
end