function dlx = mat2dl(x)
% Convert the data type of x from numeric into dlarray 
%
% function dlx = mat2dl(x)
% 
% The iput x is a numeric data structure which can be defined recursively 
% by arrays, structs and cells. The output is of dlarray data type with the 
% same data structure.
%
% See also: mat2dl_complex, dl2mat, dl2mat_complex

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if ~isstruct(x) && ~iscell(x) && ~isnumeric(x)
        up = MException('manopt:autodiff:mat2dl', ...
                    'mat2dl should only accept structs, cells or arrays.');
        throw(up);
    end

    % recursively convert each part of x into dlarrays
    if isstruct(x)
        dlx = mat2dl_struct(x);
    elseif iscell(x)
        dlx = mat2dl_cell(x);
    else
        dlx = dlarray(x);
    end

    % convert x into dlarray if x is a struct
    function dlx = mat2dl_struct(x)
        elems = fieldnames(x);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elems{ii}))
                dlx.(elems{ii}) = mat2dl_struct(x.(elems{ii}));
            elseif iscell(x.(elems{ii}))
                dlx.(elems{ii}) = mat2dl_cell(x.(elems{ii}));
            else
                dlx.(elems{ii}) = dlarray(x.(elems{ii}));
            end
        end
    end

    % convert x into dlarray if x is a cell
    function dlx = mat2dl_cell(x)
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii})
                dlx{ii} = mat2dl_struct(x{ii});
            elseif iscell(x{ii})
                dlx{ii} = mat2dl_cell(x{ii});
            else
                dlx{ii} = dlarray(x{ii});
            end
        end
    end
end

