function egrad = egradcompute_complex(autogradfunc,x)
    % check 
    dlx = mat2dl_complex(x);
    [f,egrad] = dlfeval(autogradfunc,dlx);
    % f = dl2mat(f);
    egrad = dl2mat_complex(egrad);
end