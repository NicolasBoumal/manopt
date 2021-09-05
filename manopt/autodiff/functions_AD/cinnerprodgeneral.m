function innerpro = cinnerprodgeneral(x, y)
% Computes the Euclidean inner product between x and y in the complex case
%
% function innerpro = cinnerprodgeneral(x, y)
%
% The input x and y are numeric data structures which can be defined  
% recursively by arrays, structs and cells. Each part of x and y should 
% be a struct which contains the fields real and iamg which indicate
% the real and imaginary part of the stored complex numbers. The inner
% product between x and y is defined as sum(real(conj(x(:)).*y(:))).
% The return is the sum of the inner products over each part of x and y.
% In case that x and y are structs with different fields, the inner products
% are computed only for the common fields.
%
% Note: Operations between dlarrays containing complex numbers have been
% introduced in Matlab R2021b or later. This file is only useful for Matlab
% R2021a or earlier. It will be discarded when Matlab R2021b is stable. 
% 
% See also: innerprodgeneral, manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if ~((isstruct(x) && isstruct(y)) || (iscell(x) && iscell(y))...,
            || (isnumeric(x) && isnumeric(y)) || (isstruct(x) && isnumeric(y)))
        
        up = MException('manopt:autodiff:cinnerprodgeneral' ,...
            'cinnerprodgeneral should only accept structs, cells or arrays.');
        throw(up);
        
    end
    % recursively compute the inner product 
    if isstruct(x) && isstruct(y) && (~isfield(x,'real')) && (~isfield(y,'real'))
        innerpro  = cinnerprodgeneral_struct(x,y);
    elseif iscell(x) && iscell(y)
        innerpro = cinnerprodgeneral_cell(x,y);
    else
        xconj = cconj(x);
        product = cdottimes(xconj,y);
        innerpro = sum(creal(product),'all');
        % slower
        % xcol = cmat2col(x);
        % innerpro = creal(cprod(ctransp(xcol),xcol));
    end
    
    % struct case
    function innerpro = cinnerprodgeneral_struct(x,y)
        innerpro = 0;
        elemsx = fieldnames(x);
        elemsy = fieldnames(y);
        % find the common fields
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
                product = cdottimes(xconj, y.(elemsy{iy(ii)}));
                innerpro = innerpro + sum(creal(product), 'all');
            end
        end
    end
    
    % cell case
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
                product = cdottimes(xconj, y{ii});
                innerpro = innerpro + sum(creal(product), 'all');
            end
        end
    end

end
    