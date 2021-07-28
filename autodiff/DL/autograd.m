function autogradfunc = autograd(problem)

    mycostfunction = problem.cost;
    
    function [y egrad] = autogradfunc0(mycostfunction,x)
        
        if isfield(problem,'Xmat') && (problem.Xmat == true)
            [y,Xmat] = mycostfunction(x);
            egrad = dlgradient(y,Xmat,'RetainData',false,'EnableHigherDerivatives',false);
        else
            y = mycostfunction(x);
            egrad = dlgradient(y,x,'RetainData',false,'EnableHigherDerivatives',false);
        end
    end
    
    %autogradfunc = @(x) autogradfunc0(mycostfunction,x);
    func = @(x) autogradfunc0(mycostfunction,x);
    autogradfunc = dlaccelerate(func);
    %clearCache(autogradfunc)
end