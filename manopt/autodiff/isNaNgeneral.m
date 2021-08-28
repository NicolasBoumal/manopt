function result = isNaNgeneral(x)

    if ~isstruct(x) && ~iscell(x) && ~isnumeric(x)
        up = MException('manopt:isNaNgeneral', ...
                    'isNaNgeneral should only accept structs, cells or arrays.');
        throw(up);
    end
    
    if isstruct(x)
        result = isNaN_struct(x);
    elseif iscell(x)
        result = isNaN_cell(x);
    else
        result = any(isnan(x(:)));
    end
    
    function result = isNaN_struct(x)
        elems = fieldnames(x);
        nelems = numel(elems);
        result = false;
        for ii = 1:nelems
            if isstruct(x.(elems{ii}))
                result = result + isNaN_struct(x.(elems{ii}));
            elseif iscell(x.(elems{ii}))
                result = result + isNaN_cell(x.(elems{ii}));
            else
                result = result + any(isnan(x.(elems{ii})(:)));
            end
        end
    end
    function result = isNaN_cell(x)
        ncell = length(x);
        result = false;
        for ii = 1:ncell
            if isstruct(x{ii})
                result = result + isNaN_struct(x{ii});
            elseif iscell(x{ii})
                result = result + isNaN_cell(x{ii});
            else
                result = result + any(isnan(x{ii}(:)));
            end
        end
    end
end

