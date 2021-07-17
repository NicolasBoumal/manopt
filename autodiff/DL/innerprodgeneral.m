function innerpro = innerprodgeneral(x,y)
    
    if ~((isstruct(x) && isstruct(y)) || (iscell(x) && iscell(y))...,
            || (isnumeric(x) && isnumeric(y)))
        
        up = MException('manopt:autodiff:innerprodgeneral' ,...
            'innerprodgeneral should only accept structs, cells or arrays.');
        throw(up);
        
    end
    
    if isstruct(x) && isstruct(y)
        innerpro  = innerprodgeneral_struct(x,y);
    elseif iscell(x) && iscell(y)
        innerpro = innerprodgeneral_cell(x,y);
    else
        innerpro = sum(x.*y,'all');
    end
    
    function innerpro = innerprodgeneral_struct(x,y)
        innerpro = 0;
        elems = fieldnames(x);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elems{ii}))
                innerpro = innerpro + innerprodgeneral_struct(...,
                    x.(elems{ii}),y.(elems{ii}));
            elseif iscell(x.(elems{ii}))
                innerpro = innerpro + innerprodgeneral_cell(...,
                    x.(elems{ii}),y.(elems{ii}));
            else
                innerpro = innerpro + sum(x.(elems{ii}).* y.(elems{ii}),'all');
            end
        end
    end

    function innerpro = innerprodgeneral_cell(x,y)
        innerpro = 0;
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii})
                innerpro = innerpro + innerprodgeneral_struct(x{ii},y{ii});
            elseif iscell(x{ii})
                innerpro = innerpro + innerprodgeneral_cell(x{ii},y{ii});
            else
                innerpro = innerpro + sum(x{ii}.* y{ii},'all');
            end
        end
    end

end
    