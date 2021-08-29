function innerpro = innerprodgeneral(x,y)
% Compute the Euclidean inner product between x and y
%
% function innerpro = innerprodgeneral(x,y)
%
% The input x and y are numeric data structures which can be defined  
% recursively by arrays, structs and cells. For the real case, the 
% inner product is defined as the sum of the hadamard product. For the
% complex case, the inner product between x and y is defined as 
% sum(real(conj(x(:)).*y(:))). The return is the sum of the inner products
% over each part of x and y. In case that x and y are structs with
% different fields, the inner product are computed only for the common fields.
%
% Note: Operations between dlarrays containing complex numbers are only
% introduced in Matlab R2021b or later. For Matlab R2021a or earlier, try 
% cinnerprodgeneral as an alternative way to deal with complex numbers
% stored in dlarrays. 
%
% See also: cinnerprodgeneral 

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if ~((isstruct(x) && isstruct(y)) || (iscell(x) && iscell(y))...,
            || (isnumeric(x) && isnumeric(y)))
        up = MException('manopt:autodiff:innerprodgeneral' ,...
            'innerprodgeneral should only accept structs, cells or arrays.');
        throw(up);
    end
    % recursively compute the inner product 
    if isstruct(x) && isstruct(y)
        innerpro  = innerprodgeneral_struct(x,y);
    elseif iscell(x) && iscell(y)
        innerpro = innerprodgeneral_cell(x,y);
    else
        % real case
        if isreal(x) && isreal(y)
            innerpro = x(:)'*y(:);
        else
        % complex case
            innerpro = sum(real(conj(x(:)).*y(:)));
        end
    end
    
    % struct case
    function innerpro = innerprodgeneral_struct(x,y)
        innerpro = 0;
        elemsx = fieldnames(x);
        elemsy = fieldnames(y);
        % find the common fields
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

    % cell case
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
    