function ehess = ehesscompute(autohessfunc,x,xdot)
    
    dlx = mat2dl(x);
    ehess = dlfeval(autohessfunc,dlx,xdot);
    ehess = dl2mat(ehess);

end