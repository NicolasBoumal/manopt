function h = auto_hess_call(X,Xdot,h_auto)

    if ~isstruct(X)
        h = full(h_auto(X,Xdot));
    end
end
