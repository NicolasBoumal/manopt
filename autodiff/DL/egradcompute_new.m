function [egrad,store] = egradcompute_new(problem,x,store)

    persistent tm;
    if isempty(tm)
        tm = deep.internal.recording.TapeManager();
    end
    
    mycostfunction = problem.cost;
    if ~isfield(store,'dlegrad') 
        record = deep.internal.startTracingAndSetupCleanup(tm);
        %startTracing(tm);
        [dlx,dlegrad] = subautograd(mycostfunction,x);
        store.dlegrad = dlegrad;
        store.dlx = dlx;
    end
    dlegrad = store.dlegrad;
    egrad = dl2mat(dlegrad);
    
    function [dlx,dlegrad] = subautograd(mycostfunction,x)

        dlx = mat2dl(x);
        dlx = deep.internal.recording.recordContainer(dlx);
        y = mycostfunction(dlx);
        dlegrad = dlgradient(y,dlx,'RetainData',true,'EnableHigherDerivatives',true);

    end

    function x = stopRecording(x)
    if isa(x, 'dlarray')
        x = stop(x);
    end
    end
    
end