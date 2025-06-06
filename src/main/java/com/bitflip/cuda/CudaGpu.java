package com.bitflip.cuda;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;

public class CudaGpu {
	
	// dummy test
	public static void start() {
		 // Enable JCuda exceptions for easier debugging
		JCuda.setExceptionsEnabled(true);
		
        int dataSize = 10_000_000;  // 10 million integers (~40 MB)

        int[] hostData = new int[dataSize];
        for (int i = 0; i < dataSize; i++) {
            hostData[i] = i;
        }

        Pointer deviceData = new Pointer();

        try {
            // Allocate device memory
            JCuda.cudaMalloc(deviceData, dataSize * Sizeof.INT);

            // Time host-to-device copy
            long startTime = System.nanoTime();
            JCuda.cudaMemcpy(deviceData, Pointer.to(hostData), dataSize * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            long h2dTime = System.nanoTime() - startTime;

            // Prepare array for device-to-host copy
            int[] result = new int[dataSize];

            // Time device-to-host copy
            startTime = System.nanoTime();
            JCuda.cudaMemcpy(Pointer.to(result), deviceData, dataSize * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            long d2hTime = System.nanoTime() - startTime;

            // Print results in milliseconds
            System.out.printf("Host to Device copy time: %.3f ms%n", h2dTime / 1e6);
            System.out.printf("Device to Host copy time: %.3f ms%n", d2hTime / 1e6);

            // Optionally verify correctness for the first 5 elements
            System.out.println("First 5 elements copied back:");
            for (int i = 0; i < 5; i++) {
                System.out.printf("result[%d] = %d%n", i, result[i]);
            }
        } finally {
            JCuda.cudaFree(deviceData);
        }
     
	}
	

	
}


/*
 * OS:Linux - Architecture: x86_64 - Distribution: Ubuntu - Version: 24.04
 *  lspci | grep -i nvidia
 *	01:00.0 VGA compatible controller: NVIDIA Corporation GP107M [GeForce GTX 1050 Mobile] (rev a1)
 *	01:00.1 Audio device: NVIDIA Corporation GP107GL High Definition Audio Controller (rev a1)
 *  Step1: Cuda toolkit
 *  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get -y install cuda-toolkit-12-9
 *	Step2: Cuda Driver
 *  sudo apt-get install -y cuda-drivers
 *	Step3: Confirm
 *	nvidia-smi
 *	Step4: 
 *  sudo systemctl status nvidia-persistenced
 *
*/
