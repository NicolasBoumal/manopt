function x = dl2mat_complex(dlx)
% Convert dlx which stores complex numbers in a structure into double
%
% function dlx = dl2mat_complex(x)
% 
% The iput dlx can be defined recursively by arrays, structs and cells. 
% Each part of dlx is a struct containing dlarrays with fields real and imag 
% which indicate the real and imaginary part of the stored complex numbers. 
% The function converts the struct of each part back to complex numbers. 
%
% See also: mat2dl_complex, manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:     

    if ~isstruct(dlx) && ~iscell(dlx) 
        up = MException('manopt:autodiff:dl2mat_complex', ...
                    'dl2mat_complex should only accept a struct or a cell.');
        throw(up);
    end

    % recursively convert each part of dlx into double
    if isstruct(dlx) && (~isfield(dlx,'real'))
        x = dl2mat_struct(dlx);
    elseif iscell(dlx)
        x = dl2mat_cell(dlx);
    else
        x.real = extractdata(dlx.real);
        x.imag = extractdata(dlx.imag);
        % recover complex numbers
        x = x.real + 1i*x.imag;
    end
    % convert dlx into double if dlx is a struct
    function x = dl2mat_struct(dlx)
        elems = fieldnames(dlx);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(dlx.(elems{ii})) && (~isfield(dlx.(elems{ii}),'real'))
                x.(elems{ii}) = dl2mat_struct(dlx.(elems{ii}));
            elseif iscell(dlx.(elems{ii})) 
                x.(elems{ii}) = dl2mat_cell(dlx.(elems{ii}));
            else
                % recover complex numbers
                x.(elems{ii}) = extractdata(dlx.(elems{ii}).real) + 1i*extractdata(dlx.(elems{ii}).imag);
            end
        end
    end
    % convert dlx into double if dlx is a cell
    function x = dl2mat_cell(dlx)
        ncell = length(dlx);
        for ii = 1:ncell
            if isstruct(dlx{ii}) && (~isfield(dlx{ii},'real'))
                x{ii} = dl2mat_struct(dlx{ii});
            elseif iscell(dlx{ii})
                x{ii} = dl2mat_cell(dlx{ii});
            else
                % recover complex numbers
                x{ii} = extractdata(dlx{ii}.real) + 1i*extractdata(dlx{ii}.imag);
            end
        end
    end
end

