/*Programmer: Chris Tralie
*Purpose: To implement dtw on a cross-similarity
*matrix*/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "Alignments.h"

double tripleMin(double a, double b, double c) {
    /* Return the minimum of [a, b, c] */
    double min = a;
    if (b < a) min = b;
    if (c < min) min = c;
    return min;
}

double tripleMaxBounded(double a, double b, double c) {
    /* Return the maximum of [a, b, c, 0.0] */
    double max = 0.0;
    if (a > max) max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}

float sign(int a) {
    if (a > 0) return 1;
    if (a < 0) return -1;
    return 0;
}

double DTW(double* S, int M, int N, int i1, int j1, int i2, int j2) {
    double* D;
    int i, j;
    double d, dul, dl, du;
    int A = i2 - i1 + 2;
    int B = j2 - j1 + 2;
    D = (double*)malloc(A*B*sizeof(double));//Dynamic programming matrix
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

double DTWConstrained(double* S, int M, int N, int ci, int cj) {
    double ret = DTW(S, M, N, 0, 0, ci, cj);
    return ret + DTW(S, M, N, ci, cj, M-1, N-1) - S[ci*N+cj];
}

double SMWat(double* S, int M, int N, int i1, int j1, int i2, int j2,
            double hvPenalty) {
    /*
    Allow for i1 > i2 and/or j1 > j2, so that the direction is reversed
    */
    double* D;
    int i, j;
    double d, dul, dl, du;
    int diri = sign(i2 - i1);
    int dirj = sign(j2 - j1);
    int A = diri*(i2 - i1) + 2;
    int B = dirj*(j2 - j1) + 2;
    D = (double*)malloc(A*B*sizeof(double));//Dynamic programming matrix
    for (j = 0; j < B; j++) {
        //Fill first row with zero
        D[j] = 0;
    }
    for (i = 0; i < A; i++) {
        //Fill first column with zero
        D[i*B] = 0;
    }
    for (i = 1; i < A; i++) {
        for (j = 1; j < B; j++) {
            d = S[(i1+diri*(i-1))*N + (j1+dirj*(j-1))];
            dul = d + D[(i-1)*B + (j-1)];
            dl = d + D[i*B + (j-1)] + hvPenalty;
            du = d + D[(i-1)*B + j] + hvPenalty;
            D[i*B + j] = tripleMaxBounded(dul, dl, du);
        }
    }
    d = D[A*B-1];
    free(D);
    return d;
}

double SMWatConstrained(double* S, int M, int N, int ci, int cj, double hvPenalty) {
    double ret = SMWat(S, M, N, 0, 0, ci, cj, hvPenalty);
    return ret + SMWat(S, M, N, M-1, N-1, ci, cj, hvPenalty);
}
