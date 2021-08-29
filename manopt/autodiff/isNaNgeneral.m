function result = isNaNgeneral(x)
% Determine if x contains a NaN value
%
% function result = isNaNgeneral(x)
%
% Returns a logical value which indicates whether or not the input x 
% contains a NaN value. The input x can be defined recursively by arrays, 
% structs and cells.
%
% See also:

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, Aug. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if ~isstruct(x) && ~iscell(x) && ~isnumeric(x)
        up = MException('manopt:isNaNgeneral', ...
                    'isNaNgeneral should only accept structs, cells or arrays.');
        throw(up);
    end
    
    % recursively find NaN for each part of x
    if isstruct(x)
        result = isNaN_struct(x);
        if result > 0
            result = true;
        end
    elseif iscell(x)
        result = isNaN_cell(x);
        if result > 0
            result = true;
        end
    else
        result = any(isnan(x(:)));
    end

    % when x is a struct
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

    % when x is a cell
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

