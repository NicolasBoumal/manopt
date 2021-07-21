function [ehess,store] = ehesscompute_new(problem,x,xdot,store)
    
    % xdot = mat2dl(xdot);
    mycostfunction = problem.cost;
    
    if ~isfield(store,'dlegrad') 
        
        tm = deep.internal.recording.TapeManager();
        record = deep.internal.startTracingAndSetupCleanup(tm);
        
        [dlx,dlegrad] = subautograd(mycostfunction,x);
        store.dlegrad = dlegrad;
        store.dlx = dlx;
        store.tm = tm;
        store.record = record;
        
    end
    
    tm = store.tm;
    record = store.record;
    dlegrad = store.dlegrad;
    dlx = store.dlx;
    z = innerprodgeneral(dlegrad,xdot);
    ehess = dlgradient(z,dlx,'RetainData',false,'EnableHigherDerivatives',false);
    ehess = dl2mat(ehess);
    
    
    function [dlx,dlegrad] = subautograd(mycostfunction,x)

        dlx = mat2dl(x);
        dlx = deep.internal.recording.recordContainer(dlx);
        y = mycostfunction(dlx);
        dlegrad = dlgradient(y,dlx,'RetainData',true,'EnableHigherDerivatives',true);
   
    end
    
end