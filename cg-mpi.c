// The MPI version of CG

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "util.h"

#define id(i,j) (i)*(lN+2)+j

/* sparse matrix-vector multiplication */
double * SpMV(double *lu, int lN, double h, int mpirank, int p, int sp)
{
    int i, j;
    MPI_Status status1, status2, status3, status4;
    
    double * lu_new = calloc((lN+2)*(lN+2), sizeof(double));
    double * ghost_left = calloc(lN, sizeof(double));
    double * ghost_right = calloc(lN, sizeof(double));
    double * ghost_up = calloc(lN, sizeof(double));
    double * ghost_down = calloc(lN, sizeof(double));
    
    /* communicate ghost values */
    if (mpirank % sp < sp - 1) {
        for (i = 1; i <= lN; i++)
            ghost_left[i-1] = lu[id(i,lN)];
        MPI_Send(ghost_left, lN, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
        MPI_Recv(ghost_right, lN, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status1);
        for (i = 1; i <= lN; i++)
            lu[id(i,lN+1)] = ghost_right[i-1];
    }
    if (mpirank % sp > 0) {
        for (i = 1; i <= lN; i++)
            ghost_right[i-1] = lu[id(i,1)];
        MPI_Send(ghost_right, lN, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
        MPI_Recv(ghost_left, lN, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status2);
        for (i = 1; i <= lN; i++)
            lu[id(i,0)] = ghost_left[i-1];
    }
    if (mpirank < p - sp) {
        for (j = 1; j <= lN; j++)
            ghost_down[j-1] = lu[id(lN,j)];
        MPI_Send(ghost_down, lN, MPI_DOUBLE, mpirank+sp, 126, MPI_COMM_WORLD);
        MPI_Recv(ghost_up, lN, MPI_DOUBLE, mpirank+sp, 125, MPI_COMM_WORLD, &status3);
        for (j = 1; j <= lN; j++)
            lu[id(lN+1,j)] = ghost_up[j-1];
    }
    if (mpirank > sp - 1) {
        for (j = 1; j <= lN; j++)
            ghost_up[j-1] = lu[id(1,j)];
        MPI_Send(ghost_up, lN, MPI_DOUBLE, mpirank-sp, 125, MPI_COMM_WORLD);
        MPI_Recv(ghost_down, lN, MPI_DOUBLE, mpirank-sp, 126, MPI_COMM_WORLD, &status4);
        for (j = 1; j <= lN; j++)
            lu[id(0,j)] = ghost_down[j-1];
    }
    
    /* sparse matrix-vector */
    for (i = 1; i <= lN; i++)
        for (j = 1; j <= lN; j++)
            lu_new[id(i,j)] = (4 * lu[id(i,j)] - lu[id(i-1,j)] - lu[id(i,j-1)] - lu[id(i+1,j)] - lu[id(i,j+1)]) / (h*h);
    
    return lu_new;
    
    free(lu_new);
    free(ghost_left);
    free(ghost_right);
    free(ghost_up);
    free(ghost_down);
}

/* vector-vector inner product */
double inner_prod(double *lu, double *lv, int lN)
{
    int i, j;
    double gprod = 0.0, lprod = 0.0;
    for (i = 1; i <= lN; i++)
        for (j = 1; j <= lN; j++)
            lprod += lu[id(i,j)] * lv[id(i,j)];
    MPI_Allreduce(&lprod, &gprod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return gprod;
}


int main(int argc, char *argv[])
{
    int i, j, k, mpirank, p, sp, N, lN, max_iters;
    double h, alpha, beta, r_inner, p_inner, gres, gres0, tol = 1e-5;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iters);
    
    h = 1./(N + 1);
    sp = floor(sqrt(p + 0.5));
    
    /* compute number of unknowns handled by each process */
    lN = N / sp;
    if ((N % sp != 0) && mpirank == 0) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of sqrt(p)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (mpirank == 0) {
        printf("MPI Parallel CG method for -u'' = f on (0,1)*(0,1)\n");
        printf("Number of unknowns = %d, number of processors = %d\n", N, p);
    }
    
    /* allocation of vectors, including boundary ghost points */
    double * lx = calloc((lN+2)*(lN+2), sizeof(double));
    double * lr = calloc((lN+2)*(lN+2), sizeof(double));
    double * lp = calloc((lN+2)*(lN+2), sizeof(double));
    for (i = 1; i <= lN; i++)
        for (j = 1; j <= lN; j++) {
            lr[id(i,j)] = 1;
            lp[id(i,j)] = 1;
        }
    double * Alp = calloc((lN+2)*(lN+2), sizeof(double));
    
    /* compute initial residual */
    gres0 = sqrt(inner_prod(lr, lr, lN));
    gres = gres0;
    
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
    r_inner = inner_prod(lr, lr, lN);
    
    for (k = 0; k < max_iters && gres/gres0 > tol; k++) {
        
        Alp = SpMV(lp, lN, h, mpirank, p, sp);
        p_inner = inner_prod(lp, Alp, lN);
        
        alpha = r_inner / p_inner;
        beta = r_inner;
        for (i = 0; i < (lN+2)*(lN+2); i++) {
            lx[i] += alpha * lp[i];
            lr[i] -= alpha * Alp[i];
        }
        r_inner = inner_prod(lr, lr, lN);
        beta = r_inner / beta;
        
        for (i = 0; i < (lN+2)*(lN+2); i++)
            lp[i] = lr[i] + beta * lp[i];
        
        /* compute residual for each iteration */
        gres = sqrt(inner_prod(lr, lr, lN));
        if (mpirank == 0)
            printf("Iter: %d;\t Residual: %f.\n", k, gres);
    }
    
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1, time2);
    if (mpirank == 0)
        printf("Time elapsed is %f seconds.\n", elapsed);
    
    /* clean up */
    free(lx);
    free(lr);
    free(lp);
    free(Alp);
    
    MPI_Finalize();
    return 0;
}
