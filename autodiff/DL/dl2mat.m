function x = dl2mat(dlx)
    
    if ~isstruct(dlx) && ~iscell(dlx) && ~isnumeric(dlx)
        up = MException('manopt:autodiff:dl2mat', ...
                    'dl2mat should only accept structs, cells or arrays.');
        throw(up);
    end

    if isstruct(dlx)
        x = dl2mat_stuct(dlx);
    elseif iscell(dlx)
        x = dl2mat_cell(dlx);
    else
        x = extractdata(dlx);
    end
    
    function x = dl2mat_stuct(dlx)
        elems = fieldnames(dlx);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(dlx.(elems{ii}))
                x.(elems{ii}) = dl2mat_stuct(dlx.(elems{ii}));
            elseif iscell(dlx.(elems{ii}))
                x.(elems{ii}) = dl2mat_cell(dlx.(elems{ii}));
            else
                x.(elems{ii}) = extractdata(dlx.(elems{ii}));
            end
        end
    end
    function x = dl2mat_cell(dlx)
        ncell = length(dlx);
        for ii = 1:ncell
            if isstruct(dlx{ii})
                x{ii} = dl2mat_stuct(dlx{ii});
            elseif iscell(dlx{ii})
                x{ii} = dl2mat_cell(dlx{ii});
            else
                x{ii} = extractdata(dlx{ii});
            end
        end
    end
end



