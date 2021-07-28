function preprocessautodiff(problem)
    
    problem.autogradfunc = autograd(problem);
   
    if startsWith(problem.M.name(),'Rotations manifold SO')
        
        problem.egrad = @(x) egradcompute(problem.autogradfunc,x);
        problem.ehess = @(x,xdot,store) ehesscall_ambient(problem,x,xdot,store);
        
        function [ehess,store] = ehesscall_ambient(problem,x,xdot,store)
            u = problem.M.tangent2ambient(x, xdot); 
            [ehess,store] = ehesscompute_new(problem,x,u,store);
        end
        
    elseif contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors')
        
        problem.egrad = @(x) egradcall_anchor(problem,x);
        problem.ehess = @(x,xdot,store) ehesscall_anchor(problem,x,xdot,store);
        
        function [egrad,store] = egradcall_anchor(problem,x)
            A = problem.M.A;
            egrad = egradcompute(problem.autogradfunc,x);
            egrad(:, :, A) = 0;
        end

        function [ehess,store] = ehesscall_anchor(problem,x,xdot,store)
            A = problem.M.A;
            u = problem.M.tangent2ambient(x, xdot);  
            [ehess,store] = ehesscompute_new(problem,x,u,store);
            ehess(:, :, A) = 0;
        end
        
    elseif isfield(problem,'Xmat') && (problem.Xmat == true)
        
        problem.egrad = @(x) egradcompute(problem.autogradfunc,x);
        problem.ehess = @(x,H,store) ehesscallfixedrank(problem,x,H,store);
        
        function [ehess,store] = ehesscallfixedrank(problem,x,H,store) 
            ambient_H = problem.M.tangent2ambient(x, H);
            xdot = ambient_H.U*ambient_H.S*ambient_H.V';
            [ehess,store] = ehesscompute_new(problem,x,xdot,store);
        end
        
    else
        
        problem.egrad = @(x) egradcompute(problem.autogradfunc,x);
        problem.ehess = @(x,xdot,store) ehesscompute_new(problem,x,xdot,store);
    
    end



end