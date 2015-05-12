%Extracts the non-redundant coefficients of the skew-symmetric part of a matrix
%function v=vee3(V)
%v is a [3xN] array of vectors containing a non-redundant representation of
%the skew symmetric part of each [3x3] block in V.
%It is the reverse operation of hat3.
function v=vee3(V)
    v=squeeze([V(3,2,:)-V(2,3,:); V(1,3,:)-V(3,1,:); V(2,1,:)-V(1,2,:)])/2;
end
