function Z = multiprod_ez(X, Y)

    Z = zeros(size(X, 1), size(Y, 2), size(X, 3));
    
    for k = 1 : size(Z, 3)
        Z(:, :, k) = X(:, :, k) * Y(:, :, k);
    end

end
