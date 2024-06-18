/*=================================================================
% function X = spmaskmult(A, B, I, J)
% Computes A*B at entries (I(k), J(k)) and returns the result in
% a k1-by-k2 real double matrix X the same size as I and J.
% I and J must be UINT32 matrices of size k1-by-k2.
%
% A: m-by-r, real, double
% B: r-by-n, real, double
% I: k1-by-k2 row indices, uint32
% J: k1-by-k2 column indices, uint32
%
% Complexity: O(k1k2r)
%
% Warning: no check of data consistency is performed. Matlab will
% most likely crash if I or J go out of bounds.
%
% Compile with: mex spmaskmult.c -largeArrayDims
%
% May 19, 2011, Nicolas Boumal, UCLouvain
 *=================================================================*/

/* #include <math.h> */
#include "mex.h"
#include "matrix.h"

/* Input Arguments */

#define	pA	prhs[0]
#define	pB	prhs[1]
#define	pI	prhs[2]
#define	pJ	prhs[3]

/* Output Arguments */

#define	pX	plhs[0]

void mexFunction(
          int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray* prhs[] )
{
    uint32_T *I, *J;
    double *A, *B, *X;
    mwSize m, n, r, k1, k2, numit;
    mwIndex k, it;
    
    /* Check for proper number of arguments */
    if (nrhs != 4) { 
        mexErrMsgTxt("Four input arguments are required."); 
    } else if (nlhs != 1) {
        mexErrMsgTxt("A single output argument is required."); 
    } 
    
    /* Check argument classes */
    if(!mxIsUint32(pI) || !mxIsUint32(pJ)) {
        mexErrMsgTxt("I and J must be of class UINT32."); 
    }
    if(!mxIsDouble(pA) || !mxIsDouble(pB)) {
        mexErrMsgTxt("A and B must be of class DOUBLE."); 
    }
    if(mxIsComplex(pA) || mxIsComplex(pB)) {
        mexErrMsgTxt("A and B must be REAL."); 
    }
    
    /* Check the dimensions of input arguments */ 
    m = mxGetM(pA);
    r = mxGetN(pA);
    n = mxGetN(pB);
    k1 = mxGetM(pI);
    k2 = mxGetN(pI);
    
    if(mxGetM(pB) != r)
        mexErrMsgTxt("Matrix dimensions mismatch for A and B.");
    
    if(mxGetM(pJ) != k1 || mxGetN(pJ) != k2)
        mexErrMsgTxt("Matrix dimensions mismatch for I and J.");
    
    
    /* Get pointers to the data in A, B, I, J */
    A = mxGetPr(pA);
    B = mxGetPr(pB);
    I = (uint32_T*) mxGetData(pI);
    J = (uint32_T*) mxGetData(pJ);
    
    /* Create a matrix for the ouput argument */ 
    pX = mxCreateDoubleMatrix(k1, k2, mxREAL);
    if(pX == NULL)
        mexErrMsgTxt("SPMASKMULT: Could not allocate X. Out of memory?");
    X = mxGetPr(pX);
    
    
    /* Compute */
    numit = k1*k2;
    for(it = 0; it < numit; ++it)
    {
        /* Multiply row I(it) of A with col J(it) of B */
        X[it] = A[ I[it]-1 ] * B[ r*(J[it]-1) ];
        for(k = 1; k < r; ++k)
        {
            X[it] += A[ I[it]-1 + m*k ]*B[ k + r*(J[it]-1) ];
        }
    }
    
    return;
}


