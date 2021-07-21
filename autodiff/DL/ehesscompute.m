function ehess = ehesscompute(autohessfunc,x,xdot)
    
    dlx = mat2dl(x);
    dlxdot = mat2dl(xdot);
    ehess = dlfeval(autohessfunc,dlx,dlxdot);
    ehess = dl2mat(ehess);

end