function flag = iscstruct(A)
    flag = ( isstruct(A) && isfield(A, 'real') && isfield(A, 'imag') );
end
