function h_auto = auto_hess(X,V,mycostfunction)
    import casadi.*
    
    if isempty(which('DM'))
        error(['You should first add casadi to the Matlab path.\n' ...
		       'Please run import casadi.*']);
    end
    
    X_cas = MX.sym('X',size(X));
    V_cas = MX.sym('V',size(V));
    f_cas = mycostfunction(X_cas);
    f_cas = Function('f',{X_cas},{f_cas},'X_cas','F_cas');
    
    grad_cas = gradient(f_cas(X_cas),X_cas);
    h_auto = Function('hess_v',{X_cas,V_cas},{jtimes(grad_cas,X_cas,V_cas)});

end