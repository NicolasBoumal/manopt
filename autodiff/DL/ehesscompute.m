function [ehess,store] = ehesscompute(problem,complexflag,x,xdot,store)
    
    costfunction = problem.cost;
    if ~isfield(store,'dlegrad') 
        
        tm = deep.internal.recording.TapeManager();
        record = deep.internal.startTracingAndSetupCleanup(tm);
        
        [dlx,dlegrad] = subautograd(costfunction,complexflag,x);
        store.dlegrad = dlegrad;
        store.dlx = dlx;
        store.tm = tm;
        store.record = record;
       
    end
    
    tm = store.tm;
    record = store.record;
    dlegrad = store.dlegrad;
    dlx = store.dlx;
    
    if startsWith(problem.M.name(),'Rotations manifold SO')..., 
            ||  startsWith(problem.M.name(),'Unitary manifold')...,
            || (contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors'))
        xdot = problem.M.tangent2ambient(x, xdot);
    end 
    
    if complexflag == true
        z = cinnerprodgeneral(dlegrad,xdot);
    else
        z = innerprodgeneral(dlegrad,xdot);
    end
    
    ehess = dlgradient(z,dlx,'RetainData',false,'EnableHigherDerivatives',false);
    
    if complexflag == true
        ehess = dl2mat_complex(ehess);
    else
        ehess = dl2mat(ehess);
    end
    
    if (contains(problem.M.name(),'Product rotations manifold') &&..., 
            contains(problem.M.name(),'anchors'))
        A = problem.M.A;
        ehess(:, :, A) = 0;
    end
    
    
    function [dlx,dlegrad] = subautograd(costfunction,complexflag,x)
        
        if complexflag == true
            dlx = mat2dl_complex(x);
        else
            dlx = mat2dl(x);
        end
        
        dlx = deep.internal.recording.recordContainer(dlx);
        
        y = costfunction(dlx);
        if isstruct(y) && isfield(y,'real')
            y = creal(y);
        end
        dlegrad = dlgradient(y,dlx,'RetainData',true,'EnableHigherDerivatives',true);
    end
    
end