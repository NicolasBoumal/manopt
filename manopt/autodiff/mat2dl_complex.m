function dlx = mat2dl_complex(x)

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
        xreal = real(x);
        ximag = imag(x);
        dlx.real = dlarray(xreal);
        dlx.imag = dlarray(ximag);
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
                dlx.(elems{ii}) = struct();
                xreal = real(x.(elems{ii}));
                ximag = imag(x.(elems{ii}));
                dlx.(elems{ii}).real = dlarray(xreal);
                dlx.(elems{ii}).imag = dlarray(ximag);
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
                xreal = real(x{ii});
                ximag = imag(x{ii});
                dlx{ii} = struct();
                dlx{ii}.real = dlarray(xreal);
                dlx{ii}.imag = dlarray(ximag);
            end
        end
    end
end

