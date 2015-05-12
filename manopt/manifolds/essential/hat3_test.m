function hat3_test
v=randn(3,5);
disp('Check that hat3 and vee3 are each inverse of the other')
disp(v-vee3(hat3(v)))
