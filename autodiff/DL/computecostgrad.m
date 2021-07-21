function [f, g] = computecostgrad(autogradfunc, problem, x)
    dlx = mat2dl(x);
    [f,egrad] = dlfeval(autogradfunc,dlx);
    f = dl2mat(f);
    egrad = dl2mat(egrad);
    g = problem.M.egrad2rgrad(x,egrad);
end