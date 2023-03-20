//a simple OpenCL kernel which copies all pixels from A to B
kernel void Image1(global const unsigned char* A, global unsigned char* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}


kernel void hist_simple(global const unsigned char* A, global int* B, global const float* C) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index]
	int index = (int)(bin_index / *C);

	atomic_inc(&B[index]);
}

kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

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

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

// Store the normalised cumulative histogram to a look-up table for mapping the original intensities onto the output image
kernel void lookupTable(global int* A, global int* B, const int maxIntensity, int bin_num) {
    // Get the global ID of the current item and store it in a variable
    int globalID = get_global_id(0);

    // Calculate the value for the output
    B[globalID] = A[globalID] * (double)maxIntensity / A[bin_num -1];
}

// Back-project each output pixel by indexing the look-up table with the original intensity level
kernel void backprojection(global uchar* A, global int* LUT, global uchar* B, float binSize) 
{
    // Get the global ID of the current item and store it in a variable
    int globalID = get_global_id(0);
	int index = A[globalID] / binSize;

    // Set the value for the output using the value from the look-up table
    B[globalID] = LUT[index];
}