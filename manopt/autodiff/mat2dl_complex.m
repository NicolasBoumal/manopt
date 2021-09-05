function dlx = mat2dl_complex(x)
% Convert x into a particular data structure to store complex numbers 
%
% function dlx = mat2dl_complex(x)
% 
% The iput x can be defined recursively by arrays, structs and cells. Each
% part of x should contain complex numbers. The function converts each 
% part of x into a struct containing dlarrays with fields real and imag 
% which indicate the real and imaginary part of the stored complex numbers. 
%
% See also: dl2mat_complex, manoptADhelp

% This file is part of Manopt: www.manopt.org.
% Original author: Xiaowen Jiang, July. 31, 2021.
% Contributors: Nicolas Boumal
% Change log:     

    if ~isstruct(x) && ~iscell(x) && ~isnumeric(x)
        up = MException('manopt:autodiff:mat2dl_complex', ...
                    'mat2dl_complex should only accept a struct, a cell or a numeric array');
        throw(up);
    end

    % recursively convert each part of x into a particular struct
    if isstruct(x)
        dlx = mat2dl_struct(x);
    elseif iscell(x)
        dlx = mat2dl_cell(x);
    else
        xreal = real(x);
        ximag = imag(x);
        dlx.real = dlarray(xreal);
        dlx.imag = dlarray(ximag);
    end

    % convert x into a particular dlarray struct if x is a struct
    function dlx = mat2dl_struct(x)
        elems = fieldnames(x);
        nelems = numel(elems);
        for ii = 1:nelems
            if isstruct(x.(elems{ii}))
                dlx.(elems{ii}) = mat2dl_struct(x.(elems{ii}));
            elseif iscell(x.(elems{ii}))
                dlx.(elems{ii}) = mat2dl_cell(x.(elems{ii}));
            else
                dlx.(elems{ii}) = struct();
                xreal = real(x.(elems{ii}));
                ximag = imag(x.(elems{ii}));
                dlx.(elems{ii}).real = dlarray(xreal);
                dlx.(elems{ii}).imag = dlarray(ximag);
            end
        end
    end

    % convert x into a particular dlarray struct if x is a cell
    function dlx = mat2dl_cell(x)
        ncell = length(x);
        for ii = 1:ncell
            if isstruct(x{ii})
                dlx{ii} = mat2dl_struct(x{ii});
            elseif iscell(x{ii})
                dlx{ii} = mat2dl_cell(x{ii});
            else
                xreal = real(x{ii});
                ximag = imag(x{ii});
                dlx{ii} = struct();
                dlx{ii}.real = dlarray(xreal);
                dlx{ii}.imag = dlarray(ximag);
            end
        end
    end
end

