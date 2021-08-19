function autohessfunc = autohess()

    function ehess = autohessfunc0(dlx,dlegrad,xdot)
        
        % inner product (assume commutative differential)
        z = innerprodgeneral(dlegrad,xdot);
        ehess = dlgradient(z,dlx,'RetainData',false, 'EnableHigherDerivatives',false); 
        
    end

    %autohessfunc = @(x,xdot) autohessfunc0(mycostfunction,x,xdot)
    func = @(dlx,dlegrad,xdot) autohessfunc0(dlx,dlegrad,xdot);
    autohessfunc = dlaccelerate(func);
    clearCache(autohessfunc);
    
end