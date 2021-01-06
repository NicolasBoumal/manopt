%   TTeMPS Toolbox. 
%   Michael Steinlechner, 2013-2016
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt

disp('__________________________________________________________________________') 
disp('                                                                          ') 
disp('  ______                 |    TTeMPS: A TT/MPS tensor toolbox for MATLAB  ')
disp('   |  |   |\/||_)(_`     |                                                ') 
disp('   |  | E |  ||  __)     |    Michael Steinlechner                        ')
disp('                         |    Ecole Polytechnique Federale de Lausanne    ')
disp('                         |                                                ')
disp('                         |    Version 1.1, June 2016                      ')
disp('__________________________________________________________________________')
disp('   ')
disp('         This toolbox is designed to simplify algorithmic development in the ')
disp('         TT/MPS format, making use of the object oriented programming ')
disp('         programming techniques introduced in current MATLAB versions. ')
disp('   ')
disp('WARNING: TTeMPS is experimental and not a finished product. ')
disp('         Many routines do not have sanity checks for the inputs. Use with care. ')
disp('         For questions and contact: michael.steinlechner@epfl.ch                ')
disp(' ')
disp('         In this toolbox, you will also find conversion routines betweens TTeMPS ')
disp('         and the TT Toolbox. If these are needed, the TT toolbox has to be loaded')
disp('         into the current path, too.')
disp(' ')
disp('         The algorithms are described in the following publications')
disp(' ')
disp('            D. Kressner, M. Steinlechner, and A. Uschmajew. ')
disp('            Low-rank tensor methods with subspace correction for symmetric eigenvalue problems. ')
disp('            SIAM J. Sci. Comput., 36(5):A2346-A2368, 2014.')
disp(' ')
disp('            Michael Steinlechner.                                                                                  ')
disp('            Riemannian optimization for high-dimensional tensor completion, ')
disp('            Technical report, March 2015, revised December 2015. To appear in SIAM J. Sci. Comput.')
disp('            ')
disp('            D. Kressner, M. Steinlechner, and B. Vandereycken. ')
disp('            Preconditioned low-rank Riemannian optimization for linear systems with tensor product structure. ')
disp('            Technical report, July 2015. Revised February 2016. To appear in SIAM J. Sci. Comput.')
disp('            ')
disp('            Michael Steinlechner.')
disp('            Riemannian Optimization for Solving High-Dimensional Problems with Low-Rank Tensor Structure. ')
disp('            EPFL PhD Thesis No. 6958, 2016. http://dx.doi.org/10.5075/epfl-thesis-6958 ')
disp(' ')
disp(' ')
disp('TTeMPS is licensed under a BSD 2-clause license, see LICENSE.txt')
disp('   ')
                                   
disp('------------------------')
disp('IMPORTANT: The tensor completion routines need auxiliary MEX functions')
disp('To install them, call install_mex.')
disp('Linear system and eigenvalue solvers are independent from this functionality.')
disp('------------------------')

addpath( cd )
disp('Adding algorithms ...')
addpath( [cd, filesep, 'algorithms'] )
disp('... eigenvalue solvers')
addpath( [cd, filesep, 'algorithms', filesep, 'eigenvalue'] )
disp('... linear system solvers')
addpath( [cd, filesep, 'algorithms', filesep, 'linearsystem'] )
disp('... tensor completion')
addpath( [cd, filesep, 'algorithms', filesep, 'completion'] )
disp('Adding operators')
addpath( [cd, filesep, 'operators'] )

%addpath( [cd, filesep, 'experiments'] )

disp('Adding example code')
addpath( [cd, filesep, 'examples'] )
disp('Finished. Try out the example code example.m')







