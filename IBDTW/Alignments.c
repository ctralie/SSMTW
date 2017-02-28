/*Programmer: Chris Tralie
*Purpose: To implement dtw on a cross-similarity
*matrix*/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "Alignments.h"

double tripleMin(double a, double b, double c) {
    double min = a;
    if (b < a) min = b;
    if (c < min) min = c;
    return min;
}

double DTW(double* S, int M, int N, int i1, int j1, int i2, int j2) {
    double* D;
    int i, j;
    double d, dul, dl, du;
    int A = i2 - i1 + 2;
    int B = j2 - j1 + 2;
    D = malloc(A*B*sizeof(double));//Dynamic programming matrix
    for (j = 1; j < B; j++) {
        //Fill first row with infinity
        D[j] = FLT_MAX;
    }
    for (i = 1; i < A; i++) {
        //Fill first column with infinity
        D[i*B] = FLT_MAX;
    }
    D[0] = 0.0;
    for (i = 1; i < A; i++) {
        for (j = 1; j < B; j++) {
            d = S[(i1+i-1)*N + (j1+j-1)];
            dul = d + D[(i-1)*B + (j-1)];
            dl = d + D[i*B + (j-1)];
            du = d + D[(i-1)*B + j];
            D[i*B + j] = tripleMin(dul, dl, du);
        }
    }
    d = D[A*B-1];
    free(D);
    return d;
}

double constrainedDTW(double* S, int M, int N, int ci, int cj) {
    double ret = DTW(S, M, N, 0, 0, ci, cj);
    return ret + DTW(S, M, N, ci, cj, M-1, N-1) - S[ci*N+cj];
}
