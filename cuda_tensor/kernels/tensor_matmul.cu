


template <int block_size>
__device__ void tensor_matmul(float *Z, const float *X, const float *Y,
			      unsigned int m, unsigned int l, unsigned int n){
    unsigned int z_i = blockIdx.y*block_size+threadIdx.y;
    unsigned int z_j = blockIdx.x*block_size+threadIdx.x;
    unsigned int ch = blockIdx.z;
    unsigned int k = 0;
    float z_ij = 0.0;
    unsigned int x_start_pos = ch * m * l;
    unsigned int y_start_pos = ch * l * n;
    unsigned int z_start_pos = ch * m * n;

    const float *Xstart = X + x_start_pos;
    const float *Ystart = Y + y_start_pos;
    float *Zstart = Z + z_start_pos;

#if 0
    printf("block (%d,%d,%d), thread (%d,%d,%d)\n",
	   blockIdx.x,blockIdx.y,blockIdx.z,
	   threadIdx.x,threadIdx.y,threadIdx.z);
#endif

    // SMがblockに対応する.
    // SMの中にSPがあり,SPがthreadに対応します.
    // SP間でshared memoryを共有する.

    //block間で共有する値を読み込む.
    //thread毎に対応する値を読み込む
    //Xの部分行列は右方向に動く
    //Yの部分行列は下方向に動く

    unsigned int num_of_block = l/block_size + 1;
#if 0
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
	printf("x start %f. %d %d %d\n", Xstart[0], x_start_pos, m, l);
	printf("y start %f. %d\n", Ystart[0], y_start_pos);
    }
#endif
    for(k=0;k<num_of_block;k++){
	__shared__ float Xsub[block_size][block_size];
	__shared__ float Ysub[block_size][block_size];
	unsigned int x_i = z_i;
	unsigned int x_j = k*block_size + threadIdx.x;
	unsigned int y_i = k*block_size + threadIdx.y;
	unsigned int y_j = z_j;
	//各ブロック内のthreadが対応する部分行列の要素を読み込む.
	if( x_i < m && x_j < l ){
	    Xsub[threadIdx.y][threadIdx.x] = Xstart[x_i*l+x_j];
	}
	else {
	    Xsub[threadIdx.y][threadIdx.x] = 0;
	}
	if( y_i < l && y_j < n ) {
	    //printf("(%d,%d) <- (%u,%u)\n", threadIdx.y, threadIdx.x, y_i, y_j);
	    Ysub[threadIdx.y][threadIdx.x] = Ystart[y_i*n+y_j];
	}
	else {
	    Ysub[threadIdx.y][threadIdx.x] = 0.0;
	}

	__syncthreads();
#if 0
	if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
	    unsigned int i;
	    for(i=0;i<block_size;i++){
		printf("x (%d,%d,%d) (%d,%d,%d) %d [ %f, %f, %f, %f, %f, %f, %f, %f] \n",
		       blockIdx.x,blockIdx.y,blockIdx.z,
		       threadIdx.x, threadIdx.y, threadIdx.z, i,
		       Xsub[i][0],Xsub[i][1],Xsub[i][2],Xsub[i][3],Xsub[i][4],Xsub[i][5],Xsub[i][6],Xsub[i][7]);
	    }
	    for(i=0;i<block_size;i++){
		printf("y (%d,%d,%d), (%d,%d,%d) %d [ %f, %f, %f, %f, %f, %f, %f, %f]\n",
		       blockIdx.x,blockIdx.y,blockIdx.z,
		       threadIdx.x, threadIdx.y, threadIdx.z, i,
		       Ysub[i][0],Ysub[i][1],Ysub[i][2],Ysub[i][3],Ysub[i][4],Ysub[i][5],Ysub[i][6],Ysub[i][7]);
	    }
	}
#endif
	{
	    unsigned int i;
#pragma unroll
	    for(i=0;i<block_size;i++){
		z_ij += Xsub[threadIdx.y][i]*Ysub[i][threadIdx.x];
	    }
	}
	__syncthreads();
    }

    //printf("z(%d,%d) = %f\n", z_i, z_j,  z_ij);
    if ( z_i < m && z_j < n ){
	//printf("z(%d, %d, %d) = %f\n", ch, z_i, z_j,  z_ij);
	Zstart[z_i*n + z_j] = z_ij;
    }

    return;
}

// C wrappers around our template kernel
extern "C" __global__ void tensor_matmul_bs8(float *C, float *A, float *B,
					     unsigned int m, unsigned int l, unsigned int n) {
    tensor_matmul<8>(C, A, B, m, l, n);
}
extern "C" __global__ void tensor_matmul_bs16(float *C, float *A, float *B,
					      unsigned int m, unsigned int l, unsigned int n) {
    tensor_matmul<16>(C, A, B, m, l, n);
}
extern "C" __global__ void tensor_matmul_bs32(float *C, float *A, float *B,
					      unsigned int m, unsigned int l, unsigned int n) {
    tensor_matmul<32>(C, A, B, m, l, n);
}
