function I = rotidentity(n, k)

    if nargin < 2
        k = 1;
    end

    if n == 1
        I = ones(1, 1, k);
        return;
    end

    I = zeros(n,n,k);
    In = eye(n);
    for i = 1 : k
        I(:,:,i) = In;
    end

end