function Xt = multitransp_ez(X)

    [n1, n2, n3] = size(X);
    Xt = zeros(n2, n1, n3);
    for k = 1 : n3
        Xt(:, :, k) = X(:, :, k)';
    end

end
