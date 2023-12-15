////////////////////////////////////////////////////////////////////////////////
// CUDA Kernel
////////////////////////////////////////////////////////////////////////////////

__constant__ int const_d[CT_MEM_SIZE];

__global__ void foo(int *gid_d)
{
    
    //----------------------------PRIMERA PARTE----------------
    
    extern __shared__ int shared_mem[];

	// size of the block
    int blockSize = blockDim.x * blockDim.y * blockDim.z;

    // Para el calculo suponemos que la dimension z es como añadir bloques adicionales detras del original.
    int threads_in_2d_block = blockDim.x * blockDim.y;
    // global thread ID in thread block
    int tidb = (threadIdx.x) + (blockDim.x * threadIdx.y) + (threads_in_2d_block * threadIdx.z);

    int threads_in_2d_grid = blockSize * gridDim.x * gridDim.y;
    // global thread ID in grid
    int tidg = (blockIdx.x * blockSize + tidb + gridDim.x * blockIdx.y * blockSize + threads_in_2d_grid * blockIdx.z);
    
    //----------------------------SEGUNDA PARTE----------------

    shared_mem[tidb] = gid_d[tidg];
    
    __syncthreads();

	/* shared memory */
    shared_mem[tidb] = (tidg + const_d[tidg % CT_MEM_SIZE]);

    __syncthreads();

    gid_d[tidg] = shared_mem[tidb];
}
