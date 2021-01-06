function P = constr_precond( A, k )

    %   TTeMPS Toolbox. 
    %   Michael Steinlechner, 2013-2016
    %   Questions and contact: michael.steinlechner@epfl.ch
    %   BSD 2-clause license, see LICENSE.txt

    d = A.order;
    ev = eig(A.L0);

    lmin = d*min(ev);
    lmax = d*max(ev);

    R = lmax/lmin
    
    if k == 3
        [omega, alpha] = load_coefficients( R );

    elseif k == 7
        omega = [0.0133615547183825570028305575534521842940 0.0429728469424360175410925952177443321034 0.1143029399081515586560726591147663100401,...
                 0.2838881266934189482611071431161775535656 0.6622322841999484042811198458711174907876 1.4847175320092703810050463464342840325116,...
                 3.4859753729916252771962870138366952232900];
        alpha = [0.0050213411684266507485648978019454613531 0.0312546410994290844202411500801774835168 0.1045970270084145620410366606112262388706,...
                 0.2920522758702768403556507270657505159761 0.7407504784499061527671195936939341208927 1.7609744335543204401530945069076494746696,...
                 4.0759036969145123916954953635638503328664];
    else
        error('Unknown rank specified. Choose either k=3 or k=7');
    end

    omega = omega/lmin;
    alpha = alpha/lmin;

    E = reshape( expm( -alpha(1) * A.L0), [1, A.size_row(1), A.size_col(1), 1]);
    P = omega(1)*TTeMPS_op( repmat({E},1,d) );
    for i = 2:k
        E = reshape( expm( -alpha(i) * A.L0), [1, A.size_row(1), A.size_col(1), 1]);
        P = P + omega(i)*TTeMPS_op( repmat({E},1,d) );
    end

end
