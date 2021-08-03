function egrad = egradcompute(autogradfunc,complexflag,x)
    
    if complexflag == true
        dlx = mat2dl_complex(x);
    else
        dlx = mat2dl(x);
    end
    
    [~,egrad] = dlfeval(autogradfunc,dlx);
    
    if complexflag == true
        egrad = dl2mat_complex(egrad);
    else
        egrad = dl2mat(egrad);
    end
 
end