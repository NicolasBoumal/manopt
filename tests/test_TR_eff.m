clear; close all; clc;
tic;
for i=1:25
    sparse_pca();
    elliptope_SDP_complex();
    shapefit_smoothed();
    generalized_procrustes();
    doubly_stochastic_denoising();
    truncated_svd();
    essential_svd();
    dominant_invariant_subspace();
    dominant_invariant_subspace_complex();
    low_rank_tensor_completion();
    low_rank_tensor_completion_embedded();
end
toc;

close all;
