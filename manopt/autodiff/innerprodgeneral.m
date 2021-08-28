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
        if isreal(x) && isreal(y)
            innerpro = x(:)'*y(:);
        else
            innerpro = sum(real(conj(x(:)).*y(:)));
        end
    end
    
    function innerpro = innerprodgeneral_struct(x,y)
        innerpro = 0;
        elemsx = fieldnames(x);
        elemsy = fieldnames(y);
        [elems,ix,iy] = intersect(elemsx,elemsy, 'stable');
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elemsx{ix(ii)}))
                innerpro = innerpro + innerprodgeneral_struct(...,
                    x.(elemsx{ix(ii)}),y.(elemsy{iy(ii)}));
            elseif iscell(x.(elemsx{ix(ii)}))
                innerpro = innerpro + innerprodgeneral_cell(...,
                    x.(elemsx{ix(ii)}),y.(elemsy{iy(ii)}));
            else
                xelem = x.(elemsx{ix(ii)});
                yelem = y.(elemsy{iy(ii)});
                if isreal(xelem) && isreal(yelem)
                    innerpro = innerpro + xelem(:)'*yelem(:);
                else
                    innerpro = innerpro + sum(real(conj(xelem(:)).*yelem(:)));
                end
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
                xii = x{ii};
                yii = y{ii};
                if isreal(xii) && isreal(yii)
                    innerpro = innerpro + xii(:)'*yii(:);
                else
                    innerpro = innerpro + sum(real(conj(xii(:)).*yii(:)));
                end
            end
        end
    end

end
    