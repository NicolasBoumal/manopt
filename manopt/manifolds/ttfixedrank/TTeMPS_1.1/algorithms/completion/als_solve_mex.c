/*mex -lmwlapack -lmwblas -largeArrayDims als_solve_mex.c

TTeMPS Toolbox. 
Michael Steinlechner, 2013-2016
Questions and contact: michael.steinlechner@epfl.ch
BSD 2-clause license, see LICENSE.txt 
*/

#define C_SLICE(i,j) &C[i][(ind[d*j+i]-1)*r[i]*r[i+1]]

#include "mex.h"
#include "blas.h"

/* calling: 
	TTeMPS_tangent_omega( n, r, Cores, ind, mu )
*/
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] ) {

	/* input variables */
	double* n_raw;
	double* r_raw;
	double** C;
	double* ind_raw;
	double* mu_raw;
	
	/* output variables */
	double* result;
	
	/* internal variables */
	double* tmp;
	double* L;
	double* R;
	
	mwSignedIndex* n;
	mwSignedIndex* r;
	mwSignedIndex* ind;

	mwSignedIndex numSubsref;
	mwSignedIndex mu;
	mwSignedIndex d;
	mwSignedIndex i;
	mwSignedIndex j;
	mwSignedIndex LD_result;
	mwSignedIndex maxrank = 1;
					

	/* get sizes */
	n_raw = mxGetPr( prhs[0] );
	/* get ranks */
	r_raw = mxGetPr( prhs[1] );
	/* get indices */
	ind_raw = mxGetPr( prhs[3] );
	d = mxGetM( prhs[3] );
	numSubsref = mxGetN( prhs[3] );
	mu = *mxGetPr( prhs[4] );
	
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
	r[d] = (mwSignedIndex) r_raw[d];
	
	for( i = 0; i < numSubsref*d; ++i ) {
		ind[i] = (mwSignedIndex) ind_raw[i];
	}
	

	/* Get pointers to the matrices within the cell array */
	C = mxMalloc( d*sizeof(double*) );
	
	for( i = 0; i < d; ++i ) {
		C[i] = mxGetPr( mxGetCell( prhs[2], i ) );
	}
	
	
	/* Allocate space for output */
    LD_result = r[mu-1]*r[mu];
	plhs[0] = mxCreateDoubleMatrix( LD_result, numSubsref, mxREAL);
	result = mxGetPr( plhs[0] );
	
	/* Allocate enough space for internal intermediate results */
	tmp = mxMalloc( maxrank*sizeof(double) );
	L = mxMalloc( maxrank*sizeof(double) );
	R = mxMalloc( maxrank*sizeof(double) );
	
	/* helper variables for dgemv call */
	char transa = 'T';
	char no_transa = 'N';
	mwSignedIndex ONE_i = 1;
	double ONE_d = 1.0;
	double ZERO_d = 0.0;

	for( j = 0; j < numSubsref; ++j ) {
	/*for( j = 0; j < 1; ++j ) {*/
		
		/* left side first */
		/* ... update L by multiplying with the other cores up to mu-1: */
        L[0] = 1.0;
		for( i = 0; i < mu-1; ++i ) {
			dgemv( &transa, &r[i], &r[i+1], &ONE_d, 
					C_SLICE(i,j), 
					&r[i],   
				 	L, 
					&ONE_i, &ZERO_d, tmp, &ONE_i);
			/* copy over the previous result from tmp array to L
			 * (necessary because dgemv does not work in-place */
            dcopy( &r[i+1], tmp, &ONE_i, L, &ONE_i );  
		}
		
		/* right side */
        R[0] = 1.0;
		for( i = d-1; i >= mu; --i ) {
			dgemv( &no_transa, &r[i], &r[i+1], &ONE_d, 
					C_SLICE(i,j), 
					&r[i],
				 	R,
					&ONE_i, &ZERO_d, tmp, &ONE_i);
			/* copy over the previous result from tmp array to R
			 * (necessary because dgemv does not work in-place */
            dcopy( &r[i], tmp, &ONE_i, R, &ONE_i );  
        }

        dger( &r[mu-1], &r[mu], &ONE_d, 
              L, &ONE_i, 
              R, &ONE_i, 
              &result[ j*LD_result ], &r[mu-1] );
	}

	mxFree( n );
	mxFree( r );
	mxFree( ind );
	mxFree( C ); 
	mxFree( tmp );
	mxFree( L );
	mxFree( R );
}
