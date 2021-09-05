function x = dl2mat(dlx)
% Convert the data type of x from dlarray into double 
%
% function dlx = dl2mat(x)
% 
% The iput dlx can be defined recursively by arrays, structs and cells. 
% Each part of dlx is a dlarray. The output x has the same data structure 
% as x but each part of x is converted into double data type.
%
% See also: mat2dl, mat2dl_complex, dl2mat_complex

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log: 

    if ~isstruct(dlx) && ~iscell(dlx) && ~isnumeric(dlx)
        up = MException('manopt:autodiff:dl2mat', ...
                    'dl2mat should only accept structs, cells or arrays.');
        throw(up);
    end

    % recursively convert each part of dlx into double
    if isstruct(dlx)
        x = dl2mat_struct(dlx);
    elseif iscell(dlx)
        x = dl2mat_cell(dlx);
    else
        x = extractdata(dlx);
    end
    
    % convert dlx into double if dlx is a struct
    function x = dl2mat_struct(dlx)
        elems = fieldnames(dlx);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(dlx.(elems{ii}))
                x.(elems{ii}) = dl2mat_struct(dlx.(elems{ii}));
            elseif iscell(dlx.(elems{ii}))
                x.(elems{ii}) = dl2mat_cell(dlx.(elems{ii}));
            else
                x.(elems{ii}) = extractdata(dlx.(elems{ii}));
            end
        end
    end

    % convert dlx into double if dlx is a cell
    function x = dl2mat_cell(dlx)
        ncell = length(dlx);
        for ii = 1:ncell
            if isstruct(dlx{ii})
                x{ii} = dl2mat_struct(dlx{ii});
            elseif iscell(dlx{ii})
                x{ii} = dl2mat_cell(dlx{ii});
            else
                x{ii} = extractdata(dlx{ii});
            end
        end
    end
end



