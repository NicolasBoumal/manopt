function autohessfunc = autohess(problem)

    mycostfunction = problem.cost;
    x = problem.M.rand();
    xdot = problem.M.rand();
    
    function ehess = autohessfunc0(mycostfunction,x,xdot)
        y = mycostfunction(x);
        egrad = dlgradient(y,x,'RetainData',true, 'EnableHigherDerivatives',true);
        % inner product (assume commutative differential)
        z = innerprodgeneral(egrad,xdot);
        ehess = dlgradient(z,x,'RetainData',false, 'EnableHigherDerivatives',false); 
    end

    autohessfunc = @(x,xdot) autohessfunc0(mycostfunction,x,xdot);
    
end