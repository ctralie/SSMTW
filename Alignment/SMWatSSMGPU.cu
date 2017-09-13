__global__ void SMWatSSM(float* SSMA, float* SSMB, float* CSM, int M, int N, int diagLen, int diagLenPow2, float hvPenalty, int flip) {
    //Have circularly rotating system of 3 buffers
    extern __shared__ float x[]; //Circular buffer
    int off = 0;
    int upoff = 0;

    //Other local variables
    int i, k;
    int i1, i2, j1, j2;
    int thisi, thisj;
    int idx;
    float val, score;
    int ci = blockIdx.x;
    int cj = blockIdx.y;
    int finished = 0;


    //Figure out K (number of batches)
    int K = diagLenPow2 >> 9;
    if (K == 0) {
        K = 1;
    }

    //Initialize all buffer elements to -1
    for (k = 0; k < K; k++) {
        for (off = 0; off < 3; off++) {
            if (512*k + threadIdx.x < diagLen) {
                x[512*k + threadIdx.x + off*diagLen] = -1;
            }
        }
    }
    off = 0;

    //Process each diagonal
    for (i = 0; i < N + M - 1; i++) {
        if (finished == 1) {
            break;
        }
        //Figure out the bounds of this diagonal
        i1 = i;
        j1 = 0;
        upoff = -1;
        if (i1 >= M) {
            i1 = M-1;
            j1 = i - (M-1);
            upoff = 0;
        }
        j2 = i;
        i2 = 0;
        if (j2 >= N) {
            j2 = N-1;
            i2 = i - (N-1);
        }
        //Update each batch
        for (k = 0; k < K; k++) {
            idx = k*512 + threadIdx.x;
            if (idx >= diagLen) {
                break;
            }
            thisi = i1 - idx;
            thisj = j1 + idx;
            if (thisi < i2 || thisj > j2) {
                x[off*diagLen + idx] = -1;
                continue;
            }
            if (flip) {
                val = SSMA[(M-ci-1)*M + (M-thisi-1)] - SSMB[(N-cj-1)*N + N-thisj-1];
            }
            else {
                val = SSMA[ci*M + thisi] - SSMB[cj*N + thisj];
            }
            if (val < 0) {
                val = val*-1.0f;
            }
            val = expf(-val/0.09f)-0.6f;
            score = 0.0;
            //Above
            if (idx + upoff + 1 < N + M - 1 && thisi > 0) {
                if (x[((off+1)%3)*diagLen + idx + upoff + 1] > -1) {
                    if (val + x[((off+1)%3)*diagLen + idx + upoff + 1] + hvPenalty > score)
                    score = val + x[((off+1)%3)*diagLen + idx + upoff + 1] + hvPenalty;
                }
                else if (val + hvPenalty > score) {
                    score = val + hvPenalty;
                }
                //U[thisi*N + thisj] = x[((off+1)%3)*diagLen + idx + upoff + 1];
            }
            else if (val + hvPenalty > score) {
                score = val + hvPenalty;
            }


            if (idx + upoff >= 0 && thisj > 0) {
                //Left
                if (x[((off+1)%3)*diagLen + idx + upoff] > -1) {
                    if (x[((off+1)%3)*diagLen + idx + upoff] + val + hvPenalty > score) {
                        score = x[((off+1)%3)*diagLen + idx + upoff] + val + hvPenalty;
                    }
                    else if (val + hvPenalty > score) {
                        score = val + hvPenalty;
                    }
                    //L[thisi*N + thisj] = x[((off+1)%3)*diagLen + idx + upoff];
                }
            }
            else if (val + hvPenalty > score) {
                score = val + hvPenalty;
            }


            if (i1 == M-1 && j1 > 1) {
                upoff = 1;
            }
            if (idx + upoff >= 0 && thisi > 0) {
                //Diagonal
                if (x[((off+2)%3)*diagLen + idx + upoff] > -1) {
                    if (x[((off+2)%3)*diagLen + idx + upoff] + val > score) {
                        score = x[((off+2)%3)*diagLen + idx + upoff] + val;
                    }
                }
                else if (val > score) {
                    score = val;
                }
                //UL[thisi*N + thisj] = x[((off+2)%3)*diagLen + idx + upoff];
            }
            else if (val > score) {
                score = val;
            }
            x[off*diagLen + idx] = score;
            if (thisi == ci && thisj == cj) {
                CSM[ci*N + cj] = score;
                finished = 1;
            }
        }
        off = (off + 2) % 3; //Cycle buffers
        __syncthreads();
    }
}
