function innerpro = cinnerprodgeneral(x,y)
    % Compute inner product between x and y. In this case, x is supposed to 
    % be a general dlarray structure with fields real and imag

    if ~((isstruct(x) && isstruct(y)) || (iscell(x) && iscell(y))...,
            || (isnumeric(x) && isnumeric(y)))
        
        up = MException('manopt:autodiff:innerprodgeneral' ,...
            'innerprodgeneral should only accept structs, cells or arrays.');
        throw(up);
        
    end
    
    if isstruct(x) && isstruct(y)
        innerpro  = cinnerprodgeneral_struct(x,y);
    elseif iscell(x) && iscell(y)
        innerpro = cinnerprodgeneral_cell(x,y);
    else
        dotproduct = cdotprod(x,y);
        innerpro.real = sum(dotproduct.real,'all');
        innerpro.imag = sum(dotproduct.imag,'all');
    end
    
    function innerpro = cinnerprodgeneral_struct(x,y)
        innerpro.real = 0;
        innerpro.imag = 0;
        elems = fieldnames(x);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elems{ii})) && (~isfield(x.(elems{ii}),'real'))
                innerpro.real = innerpro.real + cinnerprodgeneral_struct(...,
                    x.(elems{ii}),y.(elems{ii})).real;
                innerpro.imag = innerpro.imag + cinnerprodgeneral_struct(...,
                    x.(elems{ii}),y.(elems{ii})).imag;
            elseif iscell(x.(elems{ii}))
                innerpro.real = innerpro.real + cinnerprodgeneral_cell(...,
                    x.(elems{ii}),y.(elems{ii})).real;
                innerpro.imag = innerpro.imag + cinnerprodgeneral_cell(...,
                    x.(elems{ii}),y.(elems{ii})).imag;
            else
                dotproduct = cdotprod(x.(elems{ii}),y.(elems{ii}));
                innerprodreal = sum(dotproduct.real,'all');
                innerprodimag = sum(dotproduct.imag,'all');
                innerpro.real = innerpro.real + innerprodreal;
                innerpro.imag = innerpro.imag + innerprodimag;
            end
        end
    end

    function innerpro = cinnerprodgeneral_cell(x,y)
        innerpro.real = 0;
        innerpro.imag = 0;
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii}) && (~isfield(x.(elems{ii}),'real'))
                innerpro.real = innerpro.real + cinnerprodgeneral_struct(...,
                    x{ii},y{ii}).real;
                innerpro.imag = innerpro.imag + cinnerprodgeneral_struct(...,
                    x{ii},y{ii}).imag;
            elseif iscell(x{ii})
                innerpro.real = innerpro.real + cinnerprodgeneral_cell(...,
                    x{ii},y{ii}).real;
                innerpro.imag = innerpro.imag + cinnerprodgeneral_cell(...,
                    x{ii},y{ii}).imag;
            else
                dotproduct = cdotprod(x{ii},y{ii});
                innerprodreal = sum(dotproduct.real,'all');
                innerprodimag = sum(dotproduct.imag,'all');
                innerpro.real = innerpro.real + innerprodreal;
                innerpro.imag = innerpro.imag + innerprodimag;
            end
        end
    end

end
    