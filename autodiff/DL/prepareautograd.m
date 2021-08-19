function store = prepareautograd(problem,x,store)

    % contruct a recording Tape
    persistent tm;
    if isempty(tm)
        tm = deep.internal.recording.TapeManager();
    end
    
    if ~isfield(store,'dlegrad')
        
        cost = problem.cost; 
        record = deep.internal.startTracingAndSetupCleanup(tm);
        % compute egrad and keep recorded dlarrays
        [dlx,dlegrad] = autograd(cost,x);
        store.dlegrad = dlegrad;
        store.dlx = dlx;
        store.record = record;
        store.tm = tm;
     
    end

    function [dlx,dlegrad] = autograd(cost,x)
        
        dlx = mat2dl(x);
        % start recording
        dlx = deep.internal.recording.recordContainer(dlx);
        y = cost(dlx);
        dlegrad = dlgradient(y,dlx,'RetainData',false,'EnableHigherDerivatives',false);
        
    end

end