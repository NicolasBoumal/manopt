function x = dl2mat_complex(dlx)
    
    if ~isstruct(dlx) && ~iscell(dlx) && ~isnumeric(dlx)
        up = MException('manopt:autodiff:dl2mat', ...
                    'dl2mat should only accept structs, cells or arrays.');
        throw(up);
    end

    if isstruct(dlx) && (~isfield(dlx,'real'))
        x = dl2mat_struct(dlx);
    elseif iscell(dlx)
        x = dl2mat_cell(dlx);
    else
        x.real = extractdata(dlx.real);
        x.imag = extractdata(dlx.imag);
        x = x.real + i*x.imag;
    end
    
    function x = dl2mat_struct(dlx)
        elems = fieldnames(dlx);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(dlx.(elems{ii})) && (~isfield(dlx.(elems{ii}),'real'))
                x.(elems{ii}) = dl2mat_struct(dlx.(elems{ii}));
            elseif iscell(dlx.(elems{ii})) 
                x.(elems{ii}) = dl2mat_cell(dlx.(elems{ii}));
            else
                x.(elems{ii}) = extractdata(dlx.(elems{ii}).real) + i*extractdata(dlx.(elems{ii}).imag);
            end
        end
    end
    function x = dl2mat_cell(dlx)
        ncell = length(dlx);
        for ii = 1:ncell
            if isstruct(dlx{ii}) && (~isfield(dlx{ii},'real'))
                x{ii} = dl2mat_struct(dlx{ii});
            elseif iscell(dlx{ii})
                x{ii} = dl2mat_cell(dlx{ii});
            else
                x{ii} = extractdata(dlx{ii}.real) + i*extractdata(dlx{ii}.imag);
            end
        end
    end
end

