function dlx = mat2dl(x)

    if ~isstruct(x) && ~iscell(x) && ~isnumeric(x)
        up = MException('manopt:autodiff:mat2dl', ...
                    'mat2dl should only accept structs, cells or arrays.');
        throw(up);
    end
    
    if isstruct(x)
        dlx = mat2dl_struct(x);
    elseif iscell(x)
        dlx = mat2dl_cell(x);
    else
        dlx = dlarray(x);
    end
    
    function dlx = mat2dl_struct(x)
        elems = fieldnames(x);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elems{ii}))
                dlx.(elems{ii}) = mat2dl_struct(x.(elems{ii}));
            elseif iscell(x.(elems{ii}))
                dlx.(elems{ii}) = mat2dl_cell(x.(elems{ii}));
            else
                dlx.(elems{ii}) = dlarray(x.(elems{ii}));
            end
        end
    end
    function dlx = mat2dl_cell(x)
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii})
                dlx{ii} = mat2dl_struct(x{ii});
            elseif iscell(x{ii})
                dlx{ii} = mat2dl_cell(x{ii});
            else
                dlx{ii} = dlarray(x{ii});
            end
        end
    end
end

