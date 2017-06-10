// The MPI version of Communication-avoiding CG

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "util.h"

#define id(i,j,size) (i)*(size)+j

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
            ghost_left[i-1] = lu[id(i,lN,lN+2)];
        MPI_Send(ghost_left, lN, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
        MPI_Recv(ghost_right, lN, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status1);
        for (i = 1; i <= lN; i++)
            lu[id(i,lN+1,lN+2)] = ghost_right[i-1];
    }
    if (mpirank % sp > 0) {
        for (i = 1; i <= lN; i++)
            ghost_right[i-1] = lu[id(i,1,lN+2)];
        MPI_Send(ghost_right, lN, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
        MPI_Recv(ghost_left, lN, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status2);
        for (i = 1; i <= lN; i++)
            lu[id(i,0,lN+2)] = ghost_left[i-1];
    }
    if (mpirank < p - sp) {
        for (j = 1; j <= lN; j++)
            ghost_down[j-1] = lu[id(lN,j,lN+2)];
        MPI_Send(ghost_down, lN, MPI_DOUBLE, mpirank+sp, 126, MPI_COMM_WORLD);
        MPI_Recv(ghost_up, lN, MPI_DOUBLE, mpirank+sp, 125, MPI_COMM_WORLD, &status3);
        for (j = 1; j <= lN; j++)
            lu[id(lN+1,j,lN+2)] = ghost_up[j-1];
    }
    if (mpirank > sp - 1) {
        for (j = 1; j <= lN; j++)
            ghost_up[j-1] = lu[id(1,j,lN+2)];
        MPI_Send(ghost_up, lN, MPI_DOUBLE, mpirank-sp, 125, MPI_COMM_WORLD);
        MPI_Recv(ghost_down, lN, MPI_DOUBLE, mpirank-sp, 126, MPI_COMM_WORLD, &status4);
        for (j = 1; j <= lN; j++)
            lu[id(0,j,lN+2)] = ghost_down[j-1];
    }
    
    /* sparse matrix-vector */
    for (i = 1; i <= lN; i++)
        for (j = 1; j <= lN; j++)
            lu_new[id(i,j,lN+2)] = (4 * lu[id(i,j,lN+2)] - lu[id(i-1,j,lN+2)] - lu[id(i,j-1,lN+2)] - lu[id(i+1,j,lN+2)] - lu[id(i,j+1,lN+2)]) / (h*h);
    
    return lu_new;
    
    free(lu_new);
    free(ghost_left);
    free(ghost_right);
    free(ghost_up);
    free(ghost_down);
}

/* matrix-vector multiplication */
double * mat_vec(double *A, double *v, int row, int col)
{
    int i, j;
    double * prod = calloc(row, sizeof(double));
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++) {
            if (row == col)
                prod[i] += A[id(i,j,col)] * v[j];
            else
                prod[i] += A[id(j,i,row)] * v[j];
        }
    return prod;
}

/* vector-vector inner product */
double vec_vec(double *u, double *v, int len)
{
    int i;
    double prod = 0;
    for (i = 0; i < len; i++)
        prod += u[i] * v[i];
    return prod;
}

/* compute norm of residual */
double residual(double *lr, int lN)
{
    int i, j;
    double gres = 0.0, lres = 0.0;
    for (i = 1; i <= lN; i++)
        for (j = 1; j <= lN; j++)
            lres += lr[id(i,j,lN+2)] * lr[id(i,j,lN+2)];
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}


int main(int argc, char *argv[])
{
    int i, j, k, m, mpirank, p, sp, N, lN, max_iters, s = 5;
    double h, lg, alpha, beta, rGr, gres, gres0, tol = 1e-5;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iters);
    if (argc > 3)
        sscanf(argv[3], "%d", &s);
    
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
        printf("MPI Parallel CA-CG method for -u'' = f on (0,1)*(0,1)\n");
        printf("Number of unknowns = %d, number of processors = %d, s-steps = %d\n", N, p, s);
    }
    
    /* allocation of vectors, including boundary ghost points */
    double * lx = calloc((lN+2)*(lN+2), sizeof(double));
    double * lr = calloc((lN+2)*(lN+2), sizeof(double));
    double * lp = calloc((lN+2)*(lN+2), sizeof(double));
    for (i = 1; i <= lN; i++)
        for (j = 1; j <= lN; j++) {
            lr[id(i,j,lN+2)] = 1;
            lp[id(i,j,lN+2)] = 1;
        }
    double * prod = calloc((lN+2)*(lN+2), sizeof(double));
    double * Bpco = calloc(2*s+1, sizeof(double));
    double * Vxco = calloc((lN+2)*(lN+2), sizeof(double));
    
    /* allocation of matrices */
    //lV: (lN*lN) * (2*s+1)
    double *lV[2*s+1];
    for (m = 0; m < 2*s+1; m++)
        lV[m] = calloc(lN*lN, sizeof(double));
    //lV_tmp: ((lN+2)*(lN+2)) * (2*s+1)
    double * lV_tmp = calloc((lN+2)*(lN+2)*(2*s+1), sizeof(double));
    //B: (2*s+1) * (2*s+1)
    double * B = calloc((2*s+1)*(2*s+1), sizeof(double));
    for (j = 0; j < 2*s; j++) {
        if (j != s)
            B[id(j+1,j,2*s+1)] = 1;
    }
    
    /* compute initial residual */
    gres0 = residual(lr, lN);
    gres = gres0;
    
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
    for (k = 0; k < max_iters && gres/gres0 > tol; k++) {
        
        /* compute basis Vk */
        for (m = 0; m < 2*s+1; m++) {
            if (m == 0)
                for (i = 0; i < (lN+2)*(lN+2); i++)
                    lV_tmp[id(m,i,(lN+2)*(lN+2))] = lp[i];
            else if (m == s+1)
                for (i = 0; i < (lN+2)*(lN+2); i++)
                    lV_tmp[id(m,i,(lN+2)*(lN+2))] = lr[i];
            else {
                prod = SpMV(&lV_tmp[id(m-1,0,(lN+2)*(lN+2))], lN, h, mpirank, p, sp);
                for (i = 0; i < (lN+2)*(lN+2); i++)
                    lV_tmp[id(m,i,(lN+2)*(lN+2))] = prod[i];
            }
            for (i = 0; i < lN; i++)
                for (j = 0; j < lN; j++)
                    lV[m][i*lN+j] = lV_tmp[id(m,id(i+1,j+1,lN+2),(lN+2)*(lN+2))];
        }
        
        /* compute Gram matrix Gk */
        //G: (2*s+1) * (2*s+1)
        double * G = calloc((2*s+1)*(2*s+1), sizeof(double));
        for (i = 0; i < 2*s+1; i++)
            for (j = 0; j < 2*s+1; j++) {
                lg = vec_vec(lV[i], lV[j], lN*lN);
                MPI_Allreduce(&lg, &G[id(i,j,2*s+1)], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
        
        /* initialize coordinates */
        double * xco = calloc(2*s+1, sizeof(double));
        double * rco = calloc(2*s+1, sizeof(double)); rco[s+1] = 1;
        double * pco = calloc(2*s+1, sizeof(double)); pco[0] = 1;
        
        rGr = vec_vec(rco, mat_vec(G, rco, 2*s+1, 2*s+1), 2*s+1);
        
        /* s-step inner loops of updating coordinates, no communication */
        for (j = 0; j < s; j++) {
            Bpco = mat_vec(B, pco, 2*s+1, 2*s+1);
            alpha = rGr / vec_vec(pco, mat_vec(G, Bpco, 2*s+1, 2*s+1), 2*s+1);
            beta = rGr;
            for (i = 0; i < 2*s+1; i++) {
                xco[i] += alpha * pco[i];
                rco[i] -= alpha * Bpco[i];
            }
            rGr = vec_vec(rco, mat_vec(G, rco, 2*s+1, 2*s+1), 2*s+1);
            beta = rGr / beta;
            for (i = 0; i < 2*s+1; i++)
                pco[i] = rco[i] + beta * pco[i];
        }
        
        /* update approximate solution, residual, search direction */
        Vxco = mat_vec(lV_tmp, xco, (lN+2)*(lN+2), 2*s+1);
        for (i = 1; i <= lN; i++)
            for (j = 1; j <= lN; j++)
                lx[id(i,j,lN+2)] += Vxco[id(i,j,lN+2)];
        lr = mat_vec(lV_tmp, rco, (lN+2)*(lN+2), 2*s+1);
        lp = mat_vec(lV_tmp, rco, (lN+2)*(lN+2), 2*s+1);
        
        /* compute residual for each iteration */
        gres = residual(lr, lN);
        if (mpirank == 0)
            printf("Iter: %d;\t Residual: %f.\n", k, gres);
        
        free(xco);
        free(rco);
        free(pco);
        free(G);
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
    free(B);
    free(lV_tmp);
    for (m = 0; m < 2*s+1; m++)
        free(lV[m]);
    
    MPI_Finalize();
    return 0;
}
