/*
 * Compile with:
 *
 * mex -lmwlapack -lmwblas -largeArrayDims weingarten_omega.c 
 *
 * or simply run install_mex.m

Framework for code: TTeMPS_tangent_orth_omega.c
from Michael Steinlechner's TTeMPS Toolbox
BSD 2-clause license, see LICENSE.txt

WEINGARTEN_OMEGA

Efficiently calculates the heavier part of the Weingarten map
when assuming sparsity (the set of non-zero points is denoted
by omega in Steinlechner's literature on tensor trains).

Comments for all 7 inputs are listed beside each variable.

There are three outputs. ZZ is the original orthogonal
projection of the sparse tensor. vZ is (I \tens V')ZX'
and Zv is (I \tens X')ZV' (see Psenka & Boumal paper for details).

This file is part of Manopt: www.manopt.org.
Original author: Michael Psenka, Nov. 24, 2020.
Contributors: Nicolas Boumal
Change log:

*/
    

// TODO: try to make more efficient (not use LdRtmp?)
    
#define U_SLICE(i, j) &U[i][(ind[d * j + i] - 1) * r[i] * r[i + 1]]
#define V_SLICE(i, j) &V[i][(ind[d * j + i] - 1) * r[i] * r[i + 1]]
#define dU_SLICE(i, j) &dUR[i][(ind[d * j + i] - 1) * r[i] * r[i + 1]]
#define ZZ_SLICE(i, j) &ZZ[i][(ind[d * j + i] - 1) * r[i] * r[i + 1]]
#define Zv_SLICE(i, j) &Zv[i][(ind[d * j + i] - 1) * r[i] * r[i + 1]]
#define vZ_SLICE(i, j) &vZ[i][(ind[d * j + i] - 1) * r[i] * r[i + 1]]
#define printfFnc(...)             \
	{                              \
		mexPrintf(__VA_ARGS__);    \
		mexEvalString("drawnow;"); \
	}

#include "mex.h"
#include "blas.h"

/* calling: 
	weingarten_omega( n, r, CU, CV, ind, vals)
*/
void mexFunction(int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[])
{

	if (nlhs != 3)
	{
		mexErrMsgTxt("Must have three ouputs");
	}

	if (nrhs != 7)
	{
		mexErrMsgTxt("Must have seven inputs");
	}
	/* input variables */
	//NOTE: all cores have dimensions permuted [1 3 2]
	double *n_raw;   // tensor dimension
	double *r_raw;   // TT-rank
	double **U;		 // left-orthogonalized cores of base point
	double **V;		 // right-orthogonalized cores of base point
	double **dUR;	// dU cores of tangent vec (U1..dUR{k}...Ud)
	double *ind_raw; // list on indeces
	double *vals;	// values at list of indeces

	/* output variables */
	double **ZZ;
	mxArray **ZZ_cells;
	double **vZ;
	mxArray **vZ_cells;
	double **Zv;
	mxArray **Zv_cells;

	/* internal variables */
	double *left;
	double *dL;  // for calculating variational part
	double *LR;  // for standard projection
	double *dLR; // left side variational part
	double *LdR; // right side variational part
	double *LdRtmp; // tmp var for right side calc
	double *tmp;

	mwSignedIndex *n;
	mwSignedIndex *r;
	mwSignedIndex *ind;

	mwSignedIndex numSubsref;
	mwSignedIndex d;
	mwSignedIndex i;
	mwSignedIndex j;
	mwSignedIndex maxrank = 1;

	/* get sizes */
	n_raw = mxGetPr(prhs[0]);
	/* get ranks */
	r_raw = mxGetPr(prhs[1]);
	/* get indices */
	ind_raw = mxGetPr(prhs[5]);
	d = mxGetM(prhs[5]);
	numSubsref = mxGetN(prhs[5]);
	vals = mxGetPr(prhs[6]);

	n = mxMalloc(d * sizeof(mwSignedIndex));
	r = mxMalloc((d + 1) * sizeof(mwSignedIndex));
	ind = mxMalloc(d * numSubsref * sizeof(mwSignedIndex));

	/* Convert index arrays to integer arrays as they get converted
	 * to double arrays when passing to mex.
	 * Converting beforehand allows to avoid multiple typecasts inside the inner loop */
	for (i = 0; i < d; ++i)
	{
		n[i] = (mwSignedIndex)n_raw[i];
		r[i] = (mwSignedIndex)r_raw[i];
		if (r[i] > maxrank)
			maxrank = r[i];
	}
	r[d] = (mwSize)r_raw[d];

	for (i = 0; i < numSubsref * d; ++i)
	{
		ind[i] = (mwSignedIndex)ind_raw[i];
	}

	/* Get pointers to the matrices within the cell array */
	U = mxMalloc(d * sizeof(double *));
	V = mxMalloc(d * sizeof(double *));
	dUR = mxMalloc(d * sizeof(double *));

	for (i = 0; i < d; ++i)
	{
		U[i] = mxGetPr(mxGetCell(prhs[2], i));
		V[i] = mxGetPr(mxGetCell(prhs[3], i));
		dUR[i] = mxGetPr(mxGetCell(prhs[4], i));
	}

	/* Allocate space for output */
	plhs[0] = mxCreateCellMatrix(1, d);
	ZZ_cells = mxMalloc(d * sizeof(mxArray *));
	ZZ = mxMalloc(d * sizeof(double *));

	plhs[1] = mxCreateCellMatrix(1, d);
	vZ_cells = mxMalloc(d * sizeof(mxArray *));
	vZ = mxMalloc(d * sizeof(double *));

	plhs[2] = mxCreateCellMatrix(1, d);
	Zv_cells = mxMalloc(d * sizeof(mxArray *));
	Zv = mxMalloc(d * sizeof(double *));

	for (i = 0; i < d; i++)
	{
		ZZ_cells[i] = mxCreateDoubleMatrix(r[i] * r[i + 1] * n[i], 1, mxREAL);
		ZZ[i] = mxGetPr(ZZ_cells[i]);
		mxSetCell(plhs[0], i, ZZ_cells[i]);

		vZ_cells[i] = mxCreateDoubleMatrix(r[i] * r[i + 1] * n[i], 1, mxREAL);
		vZ[i] = mxGetPr(vZ_cells[i]);
		mxSetCell(plhs[1], i, vZ_cells[i]);

		Zv_cells[i] = mxCreateDoubleMatrix(r[i] * r[i + 1] * n[i], 1, mxREAL);
		Zv[i] = mxGetPr(Zv_cells[i]);
		mxSetCell(plhs[2], i, Zv_cells[i]);
	}

	/* Allocate enough space for internal intermediate results */
	left = mxMalloc(maxrank * (d - 1) * sizeof(double));
	dL = mxMalloc(maxrank * (d - 1) * sizeof(double));
	LR = mxMalloc(maxrank * sizeof(double));
	dLR = mxMalloc(maxrank * sizeof(double));
	LdR = mxMalloc(maxrank * sizeof(double));
	LdRtmp = mxMalloc(maxrank * sizeof(double));
	tmp = mxMalloc(maxrank * sizeof(double));

	/* helper variables for dgemv call */
	char transa = 'T';
	char no_transa = 'N';
	mwSignedIndex ONE_i = 1;
	double ONE_d = 1.0;
	double ZERO_d = 0.0;

	// vals[j] represents the gradient at the jth element of the sparse index set
	for (j = 0; j < numSubsref; ++j)
	{

		/* LEFT TO RIGHT FIRST (PRECOMPUTE)*/
		/* ... copy first core to left: */
		dcopy(&r[1], dU_SLICE(0, j), &ONE_i, &dL[0], &ONE_i);
		dcopy(&r[1], U_SLICE(0, j), &ONE_i, &left[0], &ONE_i);
		/* ... and then multiply with the other cores and store results in columns of left: */

		for (i = 1; i < d - 1; ++i)
		{
			// dL[i] = U(i)^T * dL[i-1]
			dgemv(&transa, &r[i], &r[i + 1], &ONE_d,
				  U_SLICE(i, j),
				  &r[i],
				  &dL[maxrank * (i - 1)],
				  &ONE_i, &ZERO_d, &dL[maxrank * i], &ONE_i);

			// dL[i] = dL[i] + dU(i)^T * left[i-1]
			dgemv(&transa, &r[i], &r[i + 1], &ONE_d,
				  dU_SLICE(i, j),
				  &r[i],
				  &left[maxrank * (i - 1)],
				  &ONE_i, &ONE_d, &dL[maxrank * i], &ONE_i);

			// left[i] = U(i)^T * left[i-1]
			dgemv(&transa, &r[i], &r[i + 1], &ONE_d,
				  U_SLICE(i, j),
				  &r[i],
				  &left[maxrank * (i - 1)],
				  &ONE_i, &ZERO_d, &left[maxrank * i], &ONE_i);
		}

		/* RIGHT TO LEFT PRODUCTS NOW -- USING PRECOMPUTED LEFT SIDES FROM ABOVE */
		/* last dU is without any contributions from the right */
		daxpy(&r[d - 1], &vals[j], &left[maxrank * (d - 2)], &ONE_i, ZZ_SLICE(d - 1, j), &ONE_i);
		daxpy(&r[d - 1], &vals[j], &left[maxrank * (d - 2)], &ONE_i, Zv_SLICE(d - 1, j), &ONE_i);
		daxpy(&r[d - 1], &vals[j], &dL[maxrank * (d - 2)], &ONE_i, vZ_SLICE(d - 1, j), &ONE_i);

		/* copy rightmost slice to LR variable */
		dcopy(&r[d - 1], V_SLICE(d - 1, j), &ONE_i, LR, &ONE_i);
		dcopy(&r[d - 1], V_SLICE(d - 1, j), &ONE_i, dLR, &ONE_i);
		dcopy(&r[d - 1], dU_SLICE(d - 1, j), &ONE_i, LdR, &ONE_i);
		dcopy(&r[d - 1], U_SLICE(d - 1, j), &ONE_i, LdRtmp, &ONE_i);

		/* sweep right-left to form dU{i-1} to dU{1} */
		for (i = d - 2; i > 0; --i)
		{
			/* Outer product update: 
			 * result(:,:,idx) = result(:,:,idx) + left(1:r(i), i-1)*LR' */
			dger(&r[i], &r[i + 1], &vals[j],
				 &left[maxrank * (i - 1)], &ONE_i,
				 LdR, &ONE_i,
				 Zv_SLICE(i, j), &r[i]);

			/* Outer product update: 
			 * result(:,:,idx) = result(:,:,idx) + left(1:r(i), i-1)*LR' */
			dger(&r[i], &r[i + 1], &vals[j],
				 &left[maxrank * (i - 1)], &ONE_i,
				 LR, &ONE_i,
				 ZZ_SLICE(i, j), &r[i]);

			/* Outer product update: 
			 * result(:,:,idx) = result(:,:,idx) + left(1:r(i), i-1)*LR' */
			dger(&r[i], &r[i + 1], &vals[j],
				 &dL[maxrank * (i - 1)], &ONE_i,
				 dLR, &ONE_i,
				 vZ_SLICE(i, j), &r[i]);


			/* update LR */
			dgemv(&no_transa, &r[i], &r[i + 1], &ONE_d,
				  V_SLICE(i, j),
				  &r[i],
				  LR,
				  &ONE_i, &ZERO_d, tmp, &ONE_i);
			/* ... and copy result back to LR */
			dcopy(&r[i], tmp, &ONE_i, LR, &ONE_i);


			/* update dLR */
			dgemv(&no_transa, &r[i], &r[i + 1], &ONE_d,
				  V_SLICE(i, j),
				  &r[i],
				  dLR,
				  &ONE_i, &ZERO_d, tmp, &ONE_i);
			/* ... and copy result back to dLR */
			dcopy(&r[i], tmp, &ONE_i, dLR, &ONE_i);


			/* update LdR */
			dgemv(&no_transa, &r[i], &r[i + 1], &ONE_d,
				  U_SLICE(i, j),
				  &r[i],
				  LdR,
				  &ONE_i, &ZERO_d, tmp, &ONE_i);

			dgemv(&no_transa, &r[i], &r[i + 1], &ONE_d,
				  dU_SLICE(i, j),
				  &r[i],
				  LdRtmp,
				  &ONE_i, &ONE_d, tmp, &ONE_i);

			/* ... and copy result back to LdR */
			dcopy(&r[i], tmp, &ONE_i, LdR, &ONE_i);

			/* update LdRtmp */
			dgemv(&no_transa, &r[i], &r[i + 1], &ONE_d,
				  U_SLICE(i, j),
				  &r[i],
				  LdRtmp,
				  &ONE_i, &ZERO_d, tmp, &ONE_i);
			/* ... and copy result back to LdRtmp */
			dcopy(&r[i], tmp, &ONE_i, LdRtmp, &ONE_i);
		}

		/* last core */
		daxpy(&r[1], &vals[j], LR, &ONE_i, ZZ_SLICE(0, j), &ONE_i);
		daxpy(&r[1], &vals[j], dLR, &ONE_i, vZ_SLICE(0, j), &ONE_i);
		daxpy(&r[1], &vals[j], LdR, &ONE_i, Zv_SLICE(0, j), &ONE_i);
	}
	mxFree(n);
	mxFree(r);
	mxFree(ind);
	mxFree(U);
	mxFree(V);
	mxFree(dUR);
	mxFree(left);
	mxFree(dL);
	mxFree(LR);
	mxFree(dLR);
	mxFree(LdR);
	mxFree(LdRtmp);
	mxFree(tmp);
}
