/*
How much of a difference would it make if I stored the buffers in global memory?
*/

__global__ void DTW(float* CSM, float* D, int M, int N, int diagLen, int diagLenPow2, float* res) {
    //Have circularly rotating system of 3 buffers
    extern __shared__ float x[]; //Circular buffer
    int off = 0;

    //Other local variables
    int i, k;
    int i1, i2, j1, j2;
    int thisi, thisj;
    int idx;
    float val, score;

    //Figure out K (number of batches)
    int K = diagLenPow2 >> 9;
    if (K == 0) {
        K = 1;
    }

    //Initialize all buffer elements to -1
    for (k = 0; k < K; k++) {
        for (off = 0; off < 3; off++) {
            x[512*k + threadIdx.x + off*diagLen] = -1;
        }
    }
    off = 0;

    //Process each diagonal
    for (i = 0; i < N + M - 1; i++) {
        //Figure out the bounds of this diagonal
        i1 = i;
        j1 = 0;
        if (i1 >= M) {
            i1 = M-1;
            j1 = i - (M-1);
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
            val = CSM[thisi*N + thisj];
            score = -1;
            //Above
            if (x[((off+1)%3)*diagLen + idx] > -1) {
                score = val + x[((off+1)%3)*diagLen + threadIdx.x];
            }
            if (idx > 0) {
                //Left
                if (x[((off+1)%3)*diagLen + idx - 1] > -1) {
                    if (score == -1 || x[((off+1)%3)*diagLen + idx - 1] + val < score) {
                        score = x[((off+1)%3)*diagLen + idx - 1] + val;
                    }
                }
                //Diagonal
                if (x[((off+2)%3)*diagLen + idx - 1] > -1) {
                    if (score == -1 || x[((off+2)%3)*diagLen + idx - 1] + val < score) {
                        score = x[((off+2)%3)*diagLen + idx - 1] + val;
                    }
                }
            }
            if (score == -1) {
                score = val;
            }
            D[thisi*N + thisj] = score;
            x[off*diagLen + threadIdx.x] = score;
            if (i == N + M - 2) {
                res[0] = score;
            }
        }
        off = (off + 4) % 3; //Cycle buffers
        __syncthreads();
    }
}
