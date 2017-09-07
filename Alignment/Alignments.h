double DTW(double* S, int M, int N, int i1, int j1, int i2, int j2);

double DTWConstrained(double* S, int M, int N, int ci, int cj);

double SMWat(double* S, int M, int N, int i1, int j1, int i2, int j2, double hvPenalty);

double SMWatConstrained(double* S, int M, int N, int ci, int cj, double hvPenalty);
