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
        % slower
        % xcol = cmat2col(x);
        % innerpro = creal(cprod(ctransp(xcol),xcol));
    end
    
    function innerpro = cinnerprodgeneral_struct(x,y)
        innerpro = 0;
        elemsx = fieldnames(x);
        elemsy = fieldnames(y);
        [elems,ix,iy] = intersect(elemsx,elemsy, 'stable');
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elemsx{ix(ii)})) && (~isfield(x.(elemsx{ix(ii)}),'real'))...,
                    && (~isfield(y.(elemsy{iy(ii)}),'real'))
                innerpro = innerpro + cinnerprodgeneral_struct(...,
                    x.(elemsx{ix(ii)}),y.(elemsy{iy(ii)}));
            elseif iscell(x.(elemsx{ix(ii)}))
                innerpro = innerpro + cinnerprodgeneral_cell(...,
                    x.(elemsx{ix(ii)}),y.(elemsy{iy(ii)}));
            else
                xconj = cconj(x.(elemsx{ix(ii)}));
                product = cdotprod(xconj,y.(elemsy{iy(ii)}));
                innerpro = innerpro + sum(creal(product),'all');
            end
        end
    end

    function innerpro = cinnerprodgeneral_cell(x,y)
        innerpro = 0;
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii}) && (~isfield(x{ii},'real')) && (~isfield(y{ii},'real'))
                innerpro = innerpro + cinnerprodgeneral_struct(...,
                    x{ii},y{ii});
            elseif iscell(x{ii})
                innerpro = innerpro + cinnerprodgeneral_cell(...,
                    x{ii},y{ii});
            else
                xconj = cconj(x{ii});
                product = cdotprod(xconj,y{ii});
                innerpro = innerpro + sum(creal(product),'all');
            end
        end
    end

end
    