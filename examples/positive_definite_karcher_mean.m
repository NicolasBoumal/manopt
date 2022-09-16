function X = positive_definite_karcher_mean(A)

    warning(['positive_definite_karcher_mean is now named ' ...
             'positive_definite_intrinsic_mean.']);

    if exist('A', 'var')
        X = positive_definite_intrinsic_mean(A);
    end

end
