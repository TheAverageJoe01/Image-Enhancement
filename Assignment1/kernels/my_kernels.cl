
kernel void hist_simple(global const unsigned short* A, global int* B, global const float* C) 
{
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index]
	// divide bin index by binsize
	int index = (int)(bin_index / *C);

	atomic_inc(&B[index]);
}

// double buffered hillis-steele pattern  
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//made for buffer swap
	//copying input to local memory 
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	//hillis-steele algorithm
	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copying output data to global memory 
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
kernel void scan_bl(global int* A, global int* B) 
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
	// copying output data 
	B[id] = A[id];
}

kernel void lookupTable(global int* A, global int* B, const int maxIntensity, int bin_num) 
{
    int globalID = get_global_id(0);

	// normalising it according to the bit depth and bin number 
    B[globalID] = A[globalID] * (double)maxIntensity / A[bin_num -1];
}


kernel void backprojection(global ushort* A, global int* LUT, global ushort* B, float binSize) 
{
    int globalID = get_global_id(0);
	int index = A[globalID] / binSize;

    // Set the value for the output using the value from the look-up table
    B[globalID] = LUT[index];
}