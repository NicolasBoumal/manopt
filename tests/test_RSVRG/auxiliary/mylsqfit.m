function W = mylsqfit(U, x)
    W = zeros(length(x), size(U, 2));
    for k = 1 : length(x)
        % Pull out the relevant indices and revealed entries for this column
        idx = x(k).indicator;
        values_Omega = x(k).values;
        U_Omega = U(idx,:);
        
        % Solve a simple least squares problem to populate U
        W(k,:) = (U_Omega\values_Omega)';
    end
end

