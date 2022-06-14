function options = standarddelta(M, ~, options)
% if: Delta_bar is provided but not Delta0, let Delta0 automatically
% be some fraction of the provided Delta_bar.
    if ~isfield(options, 'Delta_bar')
        if isfield(M, 'typicaldist')
            options.Delta_bar = M.typicaldist();
        else
            options.Delta_bar = sqrt(M.dim());
        end 
    end
    if ~isfield(options,'Delta0')
        options.Delta0 = options.Delta_bar / 8;
    end
end
