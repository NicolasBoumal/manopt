function autohessfunc = autohess(problem)

    mycostfunction = problem.cost;
    
    function ehess = autohessfunc0(mycostfunction,x,xdot)
        y = mycostfunction(x);
        egrad = dlgradient(y,x,'RetainData',true, 'EnableHigherDerivatives',true);
        % inner product (assume commutative differential)
        z = innerprodgeneral(egrad,xdot);
        ehess = dlgradient(z,x,'RetainData',false, 'EnableHigherDerivatives',false); 
    end

    %autohessfunc = @(x,xdot) autohessfunc0(mycostfunction,x,xdot);
    func = @(x,xdot) autohessfunc0(mycostfunction,x,xdot);
    autohessfunc = dlaccelerate(func);
    clearCache(autohessfunc)
    %x = problem.M.rand();
    %x = mat2dl(x);
    %xdot = problem.M.rand();
    %xdot = mat2dl(xdot)
    %ehess = dlfeval(autohessfunc,x,xdot);
    
end
    