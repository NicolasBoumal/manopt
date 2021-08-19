function [y,grad] = costgradcomputefixedrankembedded(problem,x)

    assert(isfield(problem,'autogradfunc'),['the problem structure must'...,
        ' contain the field autogradfunc, see autograd.'])
    
    % convert A,B into dlarrays to prepare for AD
    A = mat2dl(x.U*x.S); B = mat2dl(x.V*x.S);
    
    % compute cost and egrad according to autogradfunc
    [g1,egrad] = dlfeval(problem.autogradfunc,x,A,B);
    y = dl2mat(g1);
    
    % compute grad
    Udelta = dl2mat(egrad.A); Vdelta = dl2mat(egrad.B);
    grad.M = x.U'*Udelta;
    grad.Up = Udelta - x.U*((x.U)'*Udelta);
    grad.Vp = Vdelta - x.V*((x.V)'*Vdelta);

end