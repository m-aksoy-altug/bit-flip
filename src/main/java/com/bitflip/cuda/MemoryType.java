package com.bitflip.cuda;

public enum MemoryType {
	// Device memory 
	DEVICE,
	//Pinned host memory, allocated with cudaHostAlloc
	 HOST_PINNED,
	// Pageable memory in form of a Pointer.to(array) 
	 HOST_PAGEABLE_ARRAY,
	// Pageable memory in form of a Pointer.to(directBuffer)
	 HOST_PAGEABLE_DIRECT
}
