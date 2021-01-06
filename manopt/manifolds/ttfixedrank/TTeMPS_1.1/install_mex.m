% Install helper for mex functions

%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

% MAC AND LINUX:
if strcmp(filesep,'/') 
    disp('Compiling for Mac or Linux system ...')
    mex -lmwlapack -lmwblas -largeArrayDims @TTeMPS/subsref_mex.c -outdir @TTeMPS
    mex -lmwlapack -lmwblas -largeArrayDims @TTeMPS_tangent/TTeMPS_tangent_omega.c -outdir @TTeMPS_tangent
    mex -lmwlapack -lmwblas -largeArrayDims @TTeMPS_tangent_orth/TTeMPS_tangent_orth_omega.c -outdir @TTeMPS_tangent_orth
    mex -lmwlapack -lmwblas -largeArrayDims algorithms/completion/als_solve_mex.c -outdir algorithms/completion/

% WINDOWS:
elseif strcmp(filesep, '\')
    disp('Compiling for Windows system ...')
    mex -lmwlapack -lmwblas -largeArrayDims @TTeMPS\subsref_mex.c -outdir @TTeMPS
    mex -lmwlapack -lmwblas -largeArrayDims @TTeMPS_tangent\TTeMPS_tangent_omega.c -outdir @TTeMPS_tangent
    mex -lmwlapack -lmwblas -largeArrayDims @TTeMPS_tangent_orth\TTeMPS_tangent_orth_omega.c -outdir @TTeMPS_tangent_orth
    mex -lmwlapack -lmwblas -largeArrayDims algorithms\completion\als_solve_mex.c -outdir algorithms\completion
    mex -lmwlapack -lmwblas -largeArrayDims ..\weingarten_omega.c -outdir ..

else
    disp('Unknown filesep. Compile manually. Aborting.')
end
