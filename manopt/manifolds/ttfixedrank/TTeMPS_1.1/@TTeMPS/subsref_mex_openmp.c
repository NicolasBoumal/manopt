/* DOES NOT WORK PROPERLY, AS GETTING OPENMP SUPPORT IN MATLAB IS DIFFICULT.
 *
 * Compile using:
 *      mex -lmwlapack -lmwblas -largeArrayDims subsref_mex_openmp.c 
 * calling (do NOT call directly. Only meant to be called through TTeMPS.subsref 
 *      subsref_mex( n, r, transpose(ind), Cores)
 */

/*
 *   TTeMPS Toolbox. 
 *   Michael Steinlechner, 2013-2014
 *   Questions and contact: michael.steinlechner@epfl.ch
 *   BSD 2-clause license, see LICENSE.txt
 */


#include "mex.h"
#include "blas.h"
#include <omp.h>

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] ) {

	/* input variables */
	double* n_raw;
	double* r_raw;
	double* ind_raw;
	double** C;
	
	/* output variables */
	double* result;
	
	/* internal variables */
	double* P;
	double* current;
	
	mwSignedIndex* n;
	mwSignedIndex* r;
	mwSignedIndex* ind;

	mwSignedIndex numSubsref;
	mwSignedIndex d;
	mwSignedIndex i;
	mwSignedIndex j;
	mwSignedIndex k;
	mwSignedIndex maxrank = 1;
					

	/* get sizes */
	n_raw = mxGetPr( prhs[0] );
	/* get ranks */
	r_raw = mxGetPr( prhs[1] );
	/* get indices */
	ind_raw = mxGetPr( prhs[2] );
	d = mxGetM( prhs[2] );
	numSubsref = mxGetN( prhs[2] );
	
	n = mxMalloc( d*sizeof(mwSignedIndex) );
	r = mxMalloc( (d+1)*sizeof(mwSignedIndex) );
	ind = mxMalloc( d*numSubsref*sizeof(mwSignedIndex) );
	
	/* Convert index arrays to integer arrays as they get converted
	 * to double arrays when passing to mex.
	 * Converting beforehand allows to avoid multiple typecasts inside the inner loop */
	for( i = 0; i < d; ++i ) {
		n[i] = (mwSignedIndex) n_raw[i];
		r[i] = (mwSignedIndex) r_raw[i];
		if( r[i] > maxrank )
			maxrank = r[i];
	}
	r[d] = (mwSize) r_raw[d];
	
	for( i = 0; i < numSubsref*d; ++i ) {
		ind[i] = (mwSignedIndex) ind_raw[i];
	}
	

	/* Get pointers to the matrices within the cell array */
	C = mxMalloc( d*sizeof(double*) );
	
	for( i = 0; i<d; ++i ) {
		C[i] = mxGetPr( mxGetCell( prhs[3], i ) );
	}
	
	/* Allocate space for output */
	plhs[0] = mxCreateDoubleMatrix( numSubsref, 1, mxREAL);
	result = mxGetPr( plhs[0] );
	
	/* helper variables for dgemv call */
	char transa = 'T';
	mwSignedIndex ONE_i = 1;
	double ONE_d = 1.0;
	double ZERO_d = 0.0;
	
    #pragma omp parallel shared(n,r,ind,C,result) private(i,j,k,P,current)
    {
    	/* Allocate enough space for internal intermediate results */
    	P = malloc( maxrank*sizeof(double) );
    	current = malloc( maxrank*sizeof(double) );
    
        #pragma omp for
    	for( j = 0; j < numSubsref; ++j ) {
    		/* first two cores */
    		dgemv( &transa, &r[1], &r[2], &ONE_d, 
    				&C[1][ (ind[d*j+1]-1)*r[1]*r[2] ], 
    				&r[1],   
    			 	&C[0][ (ind[d*j]-1)*r[0]*r[1] ], 
    				&ONE_i, &ZERO_d, P, &ONE_i);
		
    		/* loop over remaining cores */
    		for( i = 2; i < d; ++i ) {
    			/* copy over the previous result to free space at P 
    			 * (necessary because dgemv does not work in-place */
    			for( k = 0; k < r[i]; ++k )
    				current[k] = P[k];
			
    			dgemv( &transa, &r[i], &r[i+1], &ONE_d, 
    					&C[i][ (ind[d*j+i]-1)*r[i]*r[i+1] ], 
    					&r[i],   
    				 	current, 
    					&ONE_i, &ZERO_d, P, &ONE_i);
			
    		}
    		result[j] = P[0];		
    	}
        free( P );
        free( current );
    }

	mxFree( n );
	mxFree( r );
	mxFree( ind );
	mxFree( C ); 
}
