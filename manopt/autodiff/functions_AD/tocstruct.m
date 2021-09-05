function B = tocstruct(A)
    if iscstruct(A)
        B = A;
    elseif isnumeric(A)
        B.real = real(A);
        B.imag = imag(A);
    else
        error('Input does not have the expected format.');
    end
end
