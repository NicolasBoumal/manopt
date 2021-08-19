function grad = gradcomputefixedrankembedded(problem,x)

    assert(isfield(problem,'autogradfunc'),['the problem structure must'...,
        ' contain the field autogradfunc, see autograd.'])
    
    % convert A,B into dlarrays to prepare for AD
    A = mat2dl(x.U*x.S); B = mat2dl(x.V*x.S);
    
    % compute egrad according to autogradfunc
    [~,egrad] = dlfeval(problem.autogradfunc,x,A,B);
    
    % compute grad
    Udelta = dl2mat(egrad.A); Vdelta = dl2mat(egrad.B);
    grad.M = x.U'*Udelta;
    grad.Up = Udelta - x.U*((x.U)'*Udelta);
    grad.Vp = Vdelta - x.V*((x.V)'*Vdelta);

end