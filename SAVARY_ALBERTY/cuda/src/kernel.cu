#include "cuda.h"
#include "particule.h"

__global__ void kernel_acceleration( int n, particule_t * in, particule_t * out ) {
	float sumX, sumY, sumZ ,dX, dY, dZ, distance, masse_invDist3;
	int i;
	float g_t = 0.1f;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < n ) { 
		sumX = 0;
		sumY = 0;
		sumZ = 0;
		for (i = 0 ; i < NB_PARTICULE ; i++){
			if (i != index){
				dX = in[i].PosX - in[index].PosX;
				dY = in[i].PosY - in[index].PosY;
				dZ = in[i].PosZ - in[index].PosZ;

				distance = sqrtf( Pow2(dX) + Pow2(dY) + Pow2(dZ) );
				if ( distance < 1.0 ) distance = 1.0;

				masse_invDist3 = in[i].Masse * (1/Pow3(distance)) * ME;

				sumX += dX * masse_invDist3;
				sumY += dY * masse_invDist3;
				sumZ += dZ * masse_invDist3;
			}
		}

		out[index].VelX += sumX;
		out[index].VelY += sumY;
		out[index].VelZ += sumZ;

		out[index].PosX += out[index].VelX * g_t;
		out[index].PosY += out[index].VelY * g_t;
		out[index].PosZ += out[index].VelZ * g_t;
	}
}

void cuda_calcul_acceleration( int nblocks, int nthreads, int n, particule_t * in, particule_t * out ) {
	kernel_acceleration<<<nblocks, nthreads>>>( n, in, out );
}


