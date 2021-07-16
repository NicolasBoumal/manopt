function g_auto = auto_grad(X,mycostfunction)
    import casadi.*
    
    if isempty(which('DM'))
        error(['You should first add casadi to the Matlab path.\n' ...
		       'Please run import casadi.*']);
    end
    
    if ~isstruct(X)
        
        X_cas = MX.sym('X',size(X));
        f_cas = mycostfunction(X_cas);
        f_cas = Function('f',{X_cas},{f_cas},'X_cas','F_cas');
    
        grad_cas = gradient(f_cas(X_cas),X_cas);
        g_auto = Function('grad',{X_cas},{grad_cas},'X_cas','Grad_cas');
        
    else
        
        elems = fieldnames(X);
        nelems = numel(elems);
        
        for ii = 1 : nelems
            X_cas.(elems{ii}) = MX.sym(sprintf('X%d',ii),size(X.(elems{ii})));
        end
        myfunc = @(X_cas) mycostfunction(X_cas);
        f_cas = Function('f',struct2cell(X_cas),{myfunc(X_cas)},fieldnames(X_cas),{'cost'});
        
        for ii = 1 : nelems
            g_auto.(elems{ii}) = jacobian_old(f_cas,ii-1,0);
        end
    end
end

