%Return the norm of each column of the input
%function nv=cnorm(v)
function nv=cnorm(v)
    nv=sqrt(sum(v.^2));
end

