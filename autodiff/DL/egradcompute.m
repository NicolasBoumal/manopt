function egrad = egradcompute(autogradfunc,x)
    
    dlx = mat2dl(x);
    [y,egrad] = dlfeval(autogradfunc,dlx);
    y = dl2mat(y);
    egrad = dl2mat(egrad);

end