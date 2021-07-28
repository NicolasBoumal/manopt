function egrad = egradcompute(autogradfunc,x)
    % check 
    dlx = mat2dl(x);
    [f,egrad] = dlfeval(autogradfunc,dlx);
    % f = dl2mat(f);
    egrad = dl2mat(egrad);

end