function innerpro = cinnerprodgeneral(x,y)
    % Compute inner product between x and y(complex case).

    if ~((isstruct(x) && isstruct(y)) || (iscell(x) && iscell(y))...,
            || (isnumeric(x) && isnumeric(y)) || (isstruct(x) && isnumeric(y)))
        
        up = MException('manopt:autodiff:cinnerprodgeneral' ,...
            'cinnerprodgeneral should only accept structs, cells or arrays.');
        throw(up);
        
    end
    
    if isstruct(x) && isstruct(y) && (~isfield(x,'real')) && (~isfield(y,'real'))
        innerpro  = cinnerprodgeneral_struct(x,y);
    elseif iscell(x) && iscell(y)
        innerpro = cinnerprodgeneral_cell(x,y);
    else
        xconj = cconj(x);
        product = cdotprod(xconj,y);
        innerpro = sum(creal(product),'all');
    end
    
    function innerpro = cinnerprodgeneral_struct(x,y)
        innerpro.real = 0;
        innerpro.imag = 0;
        elems = fieldnames(x);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elems{ii})) && (~isfield(x.(elems{ii}),'real')) && (~isfield(y,'real'))
                innerpro = innerpro + cinnerprodgeneral_struct(...,
                    x.(elems{ii}),y.(elems{ii}));
            elseif iscell(x.(elems{ii}))
                innerpro = innerpro + cinnerprodgeneral_cell(...,
                    x.(elems{ii}),y.(elems{ii}));
            else
                xconj = cconj(x.(elems{ii}));
                product = cdotprod(xconj,y.(elems{ii}));
                innerpro = sum(creal(product),'all');
            end
        end
    end

    function innerpro = cinnerprodgeneral_cell(x,y)
        innerpro.real = 0;
        innerpro.imag = 0;
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii}) && (~isfield(x{ii},'real')) && (~isfield(y,'real'))
                innerpro = innerpro + cinnerprodgeneral_struct(...,
                    x{ii},y{ii});
            elseif iscell(x{ii})
                innerpro = innerpro + cinnerprodgeneral_cell(...,
                    x{ii},y{ii});
            else
                xconj = cconj(x{ii});
                product = cdotprod(xconj,y{ii});
                innerpro = sum(creal(product),'all');
            end
        end
    end

end
    