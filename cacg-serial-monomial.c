// The serial version of Communication-avoiding CG

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"

#define index(i,j,size) (i)*(size)+j

double * matrix_vec(double *A, double *v, int row, int col);
double vec_vec(double *u, double *v, int len);

int main(int argc, char *argv[])
{
    int i, k, j, len, slen, max_iters, s = 5;
    double h, alpha, beta, rGr, res0, res, tol = 1e-5;
    double *b, *x, *r, *p, *xco, *rco, *pco, *prod, *Bpco;
    double *A, *V, *G, *B;
    
    sscanf(argv[1], "%d", &slen);
    sscanf(argv[2], "%d", &max_iters);
    if (argc > 3)
        sscanf(argv[3], "%d", &s);
    
    len = slen * slen;
    h = 1./(slen + 1);
    
    b = (double *)calloc(len, sizeof(double));
    for (i = 0; i < len; i++)
        b[i] = 1;
    x = (double *)calloc(len, sizeof(double));
    r = (double *)calloc(len, sizeof(double));
    p = (double *)calloc(len, sizeof(double));
    prod = (double *)calloc(len, sizeof(double));
    Bpco = (double *)calloc(2*s+1, sizeof(double));
    
    /* timing */
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
    //A: len * len
    A = (double *)calloc(len*len, sizeof(double));
    for (i = 0; i < len; i++) {
        A[index(i,i,len)] = 4./(h * h);
    }
    for (k = 0; k < slen; k++) {
        for (i = 0; i < slen-1; i++) {
            A[index(k*slen+i,k*slen+i+1,len)] = -1./(h * h);
            A[index(k*slen+i+1,k*slen+i,len)] = A[index(k*slen+i,k*slen+i+1,len)];
        }
    }
    for (k = 0; k < slen-1; k++) {
        for (i = 0; i < slen; i++) {
            A[index(k*slen+i,(k+1)*slen+i,len)] = -1./(h * h);
            A[index((k+1)*slen+i,k*slen+i,len)] = A[index(k*slen+i,(k+1)*slen+i,len)];
        }
    }
    
    //V: len * (2*s+1)
    V = (double *)calloc(len*(2*s+1), sizeof(double));
    
    //G: (2*s+1) * (2*s+1)
    G = (double *)calloc((2*s+1)*(2*s+1), sizeof(double));
    
    //B: (2*s+1) * (2*s+1)
    B = (double *)calloc((2*s+1)*(2*s+1), sizeof(double));
    for (j = 0; j < 2*s; j++) {
        if (j != s)
            B[index(j+1,j,2*s+1)] = 1;
    }
    
    prod = matrix_vec(A, x, len, len);
    for (i = 0; i < len; i++) {
        r[i] = b[i] - prod[i];
        p[i] = r[i];
        res0 = sqrt(vec_vec(r, r, len));
        res = res0;
    }
    
    for (k = 0; k < max_iters && res/res0 > tol; k++) {
        for (j = 0; j < 2*s+1; j++) {
            if (j == 0)
                for (i = 0; i < len; i++)
                    V[index(j,i,len)] = p[i];
            else if (j == s+1)
                for (i = 0; i < len; i++)
                    V[index(j,i,len)] = r[i];
            else {
                prod = matrix_vec(A, &V[index(j-1,0,len)], len, len);
                for (i = 0; i < len; i++)
                    V[index(j,i,len)] = prod[i];
            }
        }
    
        for (i = 0; i < 2*s+1; i++)
            for (j = 0; j < 2*s+1; j++)
                G[index(i,j,2*s+1)] = vec_vec(&V[index(i,0,len)], &V[index(j,0,len)], len);
        
        xco = (double *)calloc(2*s+1, sizeof(double));
        rco = (double *)calloc(2*s+1, sizeof(double)); rco[s+1] = 1;
        pco = (double *)calloc(2*s+1, sizeof(double)); pco[0] = 1;
        
        rGr = vec_vec(rco, matrix_vec(G, rco, 2*s+1, 2*s+1), 2*s+1);
        
        for (j = 0; j < s; j++) {
            Bpco = matrix_vec(B, pco, 2*s+1, 2*s+1);
            alpha = rGr / vec_vec(pco, matrix_vec(G, Bpco, 2*s+1, 2*s+1), 2*s+1);
            beta = rGr;
            for (i = 0; i < 2*s+1; i++) {
                xco[i] += alpha * pco[i];
                rco[i] -= alpha * Bpco[i];
            }
            rGr = vec_vec(rco, matrix_vec(G, rco, 2*s+1, 2*s+1), 2*s+1);
            beta = rGr / beta;
            for (i = 0; i < 2*s+1; i++)
                pco[i] = rco[i] + beta * pco[i];
        }
        
        prod = matrix_vec(V, xco, len, 2*s+1);
        for (i = 0; i < len; i++)
            x[i] += prod[i];
            
        r = matrix_vec(V, rco, len, 2*s+1);
        p = matrix_vec(V, pco, len, 2*s+1);
        
        res = sqrt(vec_vec(r, r, len));
        
        //print
        printf("Iter: %d; residual: %f\n", k, res);
        
        free(xco);
        free(rco);
        free(pco);
    }
    
    for (i = 0; i < len; i++)
        printf("%.5f\t", x[i]);
    printf("\n");
    
    /* timing */
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1, time2);
    printf("Time elapsed is %f seconds.\n", elapsed);
    
    /* clean up */
    free(b);
    free(x);
    free(r);
    free(p);
    free(prod);
    free(Bpco);
    free(A);
    free(V);
    free(G);
    free(B);
    
    return 0;
}


double * matrix_vec(double *A, double *v, int row, int col)
{
    int i, j;
    double *prod = (double *)calloc(row, sizeof(double));
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++) {
            if (row == col)
                prod[i] += A[index(i,j,col)] * v[j];
            else
                prod[i] += A[index(j,i,row)] * v[j];
        }
    return prod;
}

double vec_vec(double *u, double *v, int len)
{
    int i;
    double prod = 0;
    for (i = 0; i < len; i++)
        prod += u[i] * v[i];
    return prod;
}
