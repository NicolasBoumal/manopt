function A = cmultiscale(scale, A)

    [~, ~, N] = size(A);
    %A = A.*reshape(scale,1,1,N);
    A = bsxfun(@times,A,reshape(scale,1,1,N));
end