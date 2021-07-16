function g = auto_grad_call(X,g_auto)

    if ~isstruct(X)
        g = full(g_auto(X));
        
    else
        elems = fieldnames(X);
        nelems = numel(elems);
        for ii = 1 : nelems
            c = g_auto.(elems{ii}).call(X);
            g.(elems{ii}) = c.(['jac_cost_',elems{ii}]).reshape(size(X.(elems{ii})));
            g.(elems{ii}) = full(g.(elems{ii}));
        end
    end
end
