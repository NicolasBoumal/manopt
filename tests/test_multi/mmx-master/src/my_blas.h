extern "C"{

void dpotrf(const char *UPLO, const ptrdiff_t *M, double A[], const ptrdiff_t *LDA,
      ptrdiff_t *INFO);

void dposv(const char *UPLO, const ptrdiff_t *N,  const ptrdiff_t *NRHS, double A[], const ptrdiff_t *LDA,
      double B[], const ptrdiff_t *LDB, const ptrdiff_t *INFO);

void dgesv(const ptrdiff_t *N,  const ptrdiff_t *NRHS, double A[], const ptrdiff_t *LDA,
      const ptrdiff_t *IPIV, double B[], const ptrdiff_t *LDB, const ptrdiff_t *INFO);   

void dgelsy(const ptrdiff_t *M, const ptrdiff_t *N,  const ptrdiff_t *NRHS, double A[], const ptrdiff_t *LDA,
      double B[], const ptrdiff_t *LDB, const ptrdiff_t *JPVT, const double *rcond, const ptrdiff_t *rank, double work[],
      const ptrdiff_t *lwork, const ptrdiff_t *INFO);

extern void dgemm(
      char   *transa,
      char   *transb,
      ptrdiff_t *m,
      ptrdiff_t *n,
      ptrdiff_t *k,
      double *alpha,
      double *a,
      ptrdiff_t *lda,
      double *b,
      ptrdiff_t *ldb,
      double *beta,
      double *c,
      ptrdiff_t *ldc
      );

extern void dsymm(
      char   *side,
      char   *uplo,
      ptrdiff_t *m,
      ptrdiff_t *n,
      double *alpha,
      double *a,
      ptrdiff_t *lda,
      double *b,
      ptrdiff_t *ldb,
      double *beta,
      double *c,
      ptrdiff_t *ldc
      );

extern void dsyrk(
      char   *uplo,
      char   *trans,
      ptrdiff_t *n,
      ptrdiff_t *k,
      double *alpha,
      double *a,
      ptrdiff_t *lda,
      double *beta,
      double *c,
      ptrdiff_t *ldc
      );

extern void dsyr2k(
      char   *uplo,
      char   *trans,
      ptrdiff_t *n,
      ptrdiff_t *k,
      double *alpha,
      double *a,
      ptrdiff_t *lda,
      double *b,
      ptrdiff_t *ldb,
      double *beta,
      double *c,
      ptrdiff_t *ldc
      );


extern void dtrsm(
      char   *side,
      char   *uplo,
      char   *transa,
      char   *diag,
      ptrdiff_t *m,
      ptrdiff_t *n,
      double *alpha,
      double *a,
      ptrdiff_t *lda,
      double *b,
      ptrdiff_t *ldb
      );

}
