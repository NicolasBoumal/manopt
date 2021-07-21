function autogradfunc = autograd(problem)

    mycostfunction = problem.cost;
    
    function [y egrad] = autogradfunc0(mycostfunction,x)
        y = mycostfunction(x);
        egrad = dlgradient(y,x,'RetainData',false);
    end

    %autogradfunc = @(x) autogradfunc0(mycostfunction,x);
    func = @(x) autogradfunc0(mycostfunction,x);
    autogradfunc = dlaccelerate(func);
    clearCache(autogradfunc)
end