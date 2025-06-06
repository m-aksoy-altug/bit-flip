package com.bitflip.cuda;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Locale;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaEvent_t;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStreamCallback;
import jcuda.runtime.cudaStream_t;

 // https://github.com/jcuda/jcuda-samples
public class CudaTest {
	private static final Logger log = LoggerFactory.getLogger(CudaTest.class);
	private static final int NUM_FLOATS = (1 << 22);
	private static final int COPY_RUNS = 10;
	@BeforeEach
	public void setCuda() {
	  JCuda.setExceptionsEnabled(true);
	  JCublas2.setExceptionsEnabled(true);	
	}
	
		
	@Test
	public void JCudaRuntimeAsyncCopies(){
		  // Create the host input data
        float data[] = new float[NUM_FLOATS];
        for (int i = 0; i < NUM_FLOATS; i++){
            data[i] = i;
        }
        
        // Run tests for all combinations of target- and source memory types:
        System.out.println("Timing " + COPY_RUNS + " copy operations of " + NUM_FLOATS + " float values");
        System.out.println("Synchronous memory copies");
        System.out.printf("%22s    %22s   %10s    %10s     %s\n", "TARGET", "SOURCE", "", "TOTAL", "PASSED");
        System.out.println(String.format("%90s", " ").replace(' ', '-'));
        
        boolean passed = true;
        
        for (MemoryType targetType : MemoryType.values()){
            for (MemoryType sourceType : MemoryType.values()){
                passed &= testSync(targetType, sourceType, data);
            }
        }
        
        System.out.println("\n Asynchronous memory copies");
        System.out.printf("%22s    %22s   %10s    %10s     %s\n", "TARGET", "SOURCE", "CALL", "WAIT", "PASSED");
        System.out.println(String.format("%90s", " ").replace(' ', '-'));

        for (MemoryType targetType : MemoryType.values()){
            for (MemoryType sourceType : MemoryType.values()){
                passed &= testAsync(targetType, sourceType, data);
            }
        }
        System.out.println("DONE, result: " + (passed ? "PASSED" : "FAILED"));
   }
	
	/**Test a synchronous (blocking) copy of the given data between the given memory types
     * @param targetType The target {@link MemoryType}
     * @param sourceType The source {@link MemoryType}
     * @param data The data
     * @return Whether the test passed
     */
	private static boolean testSync(MemoryType targetType, MemoryType sourceType, float data[]){
	        // Allocate source- and target memory, and fill the source memory with the given data
	        int numBytes = data.length * Sizeof.FLOAT;
	        int kind = Memory.getCudaMemcpyKind(targetType, sourceType);
	        Memory target = new Memory(targetType, numBytes);
	        Memory source = new Memory(sourceType, numBytes);
	        source.put(data);
	        Pointer t = target.getPointer();
	        Pointer s = source.getPointer();

	        // Perform the copying operations
	        long before = System.nanoTime();
	        for (int i = 0; i < COPY_RUNS; i++){
	        	// Pointer dst,Pointer src,long count, int cudaMemcpyKind_kind
	            JCuda.cudaMemcpy(t, s, numBytes, kind);
	        }
	        long after = System.nanoTime();
	        double durationCopyMS = (after - before) / 1e6;

	        // Verify the result and clean up
	        boolean passed = verify(target, data);
	        target.release();
	        source.release();

	        // Print the timing information
	        String dcs = String.format(Locale.ENGLISH, "%10.3f", durationCopyMS);
	        System.out.printf("%22s <- %22s : %10s ms %10s ms  %s\n", 
	                targetType, sourceType, "", dcs, passed);
	        
	        return passed;
	  }

	/** Test an asynchronous (non-blocking) copy of the given data between the given memory types
     * @param targetType The target {@link MemoryType}
     * @param sourceType The source {@link MemoryType}
     * @param data The data
     * @return Whether the test passed
     */
    private static boolean testAsync(MemoryType targetType, MemoryType sourceType, float data[]){
        // Allocate source- and target memory, and fill the source
        // memory with the given data
        int numBytes = data.length * Sizeof.FLOAT;
        int kind = Memory.getCudaMemcpyKind(targetType, sourceType);
        Memory target = new Memory(targetType, numBytes);
        Memory source = new Memory(sourceType, numBytes);
        source.put(data);
        Pointer t = target.getPointer();
        Pointer s = source.getPointer();

        // Create a stream
        cudaStream_t stream = new cudaStream_t();
        JCuda.cudaStreamCreate(stream);

        // Issue the asynchronous copies on the stream
        long beforeCall = System.nanoTime();
        for (int i = 0; i < COPY_RUNS; i++){
        	JCuda.cudaMemcpyAsync(t, s, numBytes, kind, stream);
        }
        long afterCall = System.nanoTime();
        double durationCallMS = (afterCall - beforeCall) / 1e6;

        // Wait for the stream to be finished
        long beforeWait = System.nanoTime();
        JCuda.cudaStreamSynchronize(stream);
        long afterWait = System.nanoTime();
        double durationWaitMS = (afterWait - beforeWait) / 1e6;

        // Verify the result and clean up
        boolean passed = verify(target, data);
        target.release();
        source.release();

        // Print the timing information
        String dcs = String.format(Locale.ENGLISH, "%10.3f", durationCallMS);
        String dws = String.format(Locale.ENGLISH, "%10.3f", durationWaitMS);
        System.out.printf("%22s <- %22s : %10s ms %10s ms  %s\n",
            targetType, sourceType, dcs, dws, passed);
        return passed;
    }
	
    
    /**
     * Verify that the data that is stored in the given memory is equal to the data in the given array
     * @param target The memory
     * @param data The data that is expected in the memory
     * @return Whether the data was equal
     */
    private static boolean verify(Memory target, float data[]){
        float result[] = new float[data.length];
        target.get(result);
        boolean passed = true;
        for (int i = 0; i < data.length; i++){
            float f0 = data[i];
            float f1 = result[i];
            if (f0 != f1){
                System.out.println("ERROR: At index " + i + " expected " + f0 + " but found " + f1);
                passed = false;
                break;
            }
        }
        return passed;
    }
    
    
    /**
     * A comparison of the bandwidth of memory copy operations, depending on
     * the memory type
     * This test computes the bandwidth of the data transfer from the host to 
     * the device for different host memory types:
     * - Host data is once allocated as pinned memory  (using cudaHostAlloc)
     * - Host data that is stored in pageable memory (comparable to malloc in C), 
     *  in a Java array
     *  a direct buffer
     */
    @Test
	public void JCudaRuntimeMemoryBandwidths() {
    	int device = 0;
        JCuda.cudaSetDevice(device);
        
        int hostAllocFlags = JCuda.cudaHostAllocWriteCombined;
        memoryBandwidths(MemoryType.HOST_PINNED, hostAllocFlags);
        memoryBandwidths(MemoryType.HOST_PAGEABLE_ARRAY, hostAllocFlags);
        memoryBandwidths(MemoryType.HOST_PAGEABLE_DIRECT, hostAllocFlags);
        System.out.println("JCudaRuntimeMemoryBandwidths is Done");
    	
    }
    
    static void memoryBandwidths(MemoryType hostMemoryMode, int hostAllocFlags){
        int minExponent = 10;
        int maxExponent = 28;
        int count = maxExponent - minExponent;
        int memorySizes[] = new int[count];
        float bandwidths[] = new float[memorySizes.length];
        
        System.out.print("Running with " + hostMemoryMode);
        for (int i = 0; i < count; i++){
            System.out.print(".");
            memorySizes[i] = (1 << minExponent + i);
            float bandwidth = computeBandwidth(
                hostMemoryMode, hostAllocFlags, memorySizes[i]);
            bandwidths[i] = bandwidth;
        }
       
        System.out.println("\n Bandwidths for " + hostMemoryMode);
        for (int i = 0; i < memorySizes.length; i++){
            String s = String.format("%10d", memorySizes[i]);
            String b = String.format(Locale.ENGLISH, "%5.3f", bandwidths[i]);
            System.out.println(s + " bytes : " + b + " MB/s");
        }
        System.out.println("\n");
    }
    
    
    static void computeBandwidths(MemoryType hostMemoryMode, int hostAllocFlags,
            								int memorySizes[], float bandwidths[]){
            for (int i = 0; i < memorySizes.length; i++){
                int memorySize = memorySizes[i];
                float bandwidth = computeBandwidth(
                    hostMemoryMode, hostAllocFlags, memorySize);
                bandwidths[i] = bandwidth;
            }
    }
    
    
    static float computeBandwidth(MemoryType hostMemoryMode, int hostAllocFlags, int memorySize){
            // Initialize the host memory
            Pointer hostData = null;
            ByteBuffer hostDataBuffer = null;
            if (hostMemoryMode == MemoryType.HOST_PINNED){
                // Allocate pinned (page-locked) host memory
                hostData = new Pointer();
                JCuda.cudaHostAlloc(hostData, memorySize, hostAllocFlags);
                hostDataBuffer = hostData.getByteBuffer(0, memorySize);
            }
            else if (hostMemoryMode == MemoryType.HOST_PAGEABLE_ARRAY){
                // The host memory is pageable and stored in a Java array
                byte array[] = new byte[memorySize];
                hostDataBuffer = ByteBuffer.wrap(array);
                hostData = Pointer.to(array);
            }else{
                // The host memory is pageable and stored in a direct byte buffer
                hostDataBuffer = ByteBuffer.allocateDirect(memorySize);
                hostData = Pointer.to(hostDataBuffer);
            }

            // Fill the memory with arbitrary data
            for (int i = 0; i < memorySize; i++){
                hostDataBuffer.put(i, (byte) i);
            }

            // Allocate device memory
            Pointer deviceData = new Pointer();
            JCuda.cudaMalloc(deviceData, memorySize);

            final int runs = 10;
            float bandwidth = computeBandwidth(
                deviceData, hostData, cudaMemcpyKind.cudaMemcpyHostToDevice, memorySize, runs);

            // Clean up
            if (hostMemoryMode == MemoryType.HOST_PINNED){
                JCuda.cudaFreeHost(hostData);
            }
            JCuda.cudaFree(deviceData);
            return bandwidth;
        }
    
    
    static float computeBandwidth(Pointer dstData, Pointer srcData, int memcopyKind, int memSize, int runs)
        {
            // Initialize the events for the time measure
            cudaEvent_t start = new cudaEvent_t();
            cudaEvent_t stop = new cudaEvent_t();
            JCuda.cudaEventCreate(start);
            JCuda.cudaEventCreate(stop);

            // Perform the specified number of copying operations
            JCuda.cudaEventRecord(start, null);
            for (int i = 0; i < runs; i++)
            {
                JCuda.cudaMemcpyAsync(dstData, srcData, memSize, memcopyKind, null);
            }
            JCuda.cudaEventRecord(stop, null);
            JCuda.cudaDeviceSynchronize();

            // Compute the elapsed time and bandwidth
            // in MB per second
            float elapsedTimeMsArray[] = { Float.NaN };
            JCuda.cudaEventElapsedTime(elapsedTimeMsArray, start, stop);
            float elapsedTimeMs = elapsedTimeMsArray[0];
            float bandwidthInBytesPerMs = ((float) memSize * runs) / elapsedTimeMs;
            float bandwidth = bandwidthInBytesPerMs / 1024;

            // Clean up
            JCuda.cudaEventDestroy(stop);
            JCuda.cudaEventDestroy(start);
            return bandwidth;
        }
    
    /**
     *  how to use mapped memory in JCuda. Host memory is allocated and mapped to the device.
     *  There, it is modified with a  runtime library function (CUBLAS, for example), 
     *  which then effectively writes to host memory.
     */
    @Test
	public void JCudaRuntimeMappedMemory() {
    	// Check if the device supports mapped host memory
        cudaDeviceProp deviceProperties = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(deviceProperties, 0);
        if (deviceProperties.canMapHostMemory == 0){
            System.err.println("This device can not map host memory");
            System.err.println(deviceProperties.toFormattedString());
            return;
        }
        // Set the flag indicating that mapped memory will be used
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceMapHost);

        // Allocate mappable host memory
        int n = 5;
        Pointer hostPointer = new Pointer();
        JCuda.cudaHostAlloc(hostPointer, n * Sizeof.FLOAT, JCuda.cudaHostAllocMapped);

        // Create a device pointer mapping the host memory
        Pointer devicePointer = new Pointer();
        JCuda.cudaHostGetDevicePointer(devicePointer, hostPointer, 0);

        // Obtain a ByteBuffer for accessing the data in the host
        // pointer. Modifications in this ByteBuffer will be
        // visible in the device memory.
        ByteBuffer byteBuffer = hostPointer.getByteBuffer(0, n * Sizeof.FLOAT);

        // Set the byte order of the ByteBuffer
        byteBuffer.order(ByteOrder.nativeOrder());

        // For convenience, view the ByteBuffer as a FloatBuffer
        // and fill it with some sample data
        FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
        System.out.print("Input : ");
        for (int i = 0; i < n; i++){
            floatBuffer.put(i, (float) i);
            System.out.print(floatBuffer.get(i) + ", ");
        }
        System.out.println();

        // Apply a CUBLAS routine to the device pointer. This will
        // modify the host data, which was mapped to the device.
        cublasHandle handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        Pointer two = Pointer.to(new float[] { 3.0f });
        JCublas2.cublasSscal(handle, n, two, devicePointer, 1);
        JCublas2.cublasDestroy(handle);
        JCuda.cudaDeviceSynchronize();

        // Print the contents of the host memory after the
        // modification via the mapped pointer.
        System.out.print("Output: ");
        for (int i = 0; i < n; i++){
            System.out.print(floatBuffer.get(i) + ", ");
        }
        System.out.println();

        // Clean up
        JCuda.cudaFreeHost(hostPointer);
    }
    
    @Test
	public void JCudaRuntimeBasicStreamCallback() {
		// The stream on which the callbacks will be registered. 
		// When this is "null", then it is the default stream.
        cudaStream_t stream = null;
        boolean useDefaultStream = true;
        useDefaultStream = false;
        if (!useDefaultStream){
            stream = new cudaStream_t();
            JCuda.cudaStreamCreate(stream);
        }
        System.out.println("Using stream " + stream);
        // Define the callback
        cudaStreamCallback callback = new cudaStreamCallback(){
            @Override
            public void call(cudaStream_t stream, int status, Object userData){
                System.out.println("Callback called");
                System.out.println("    stream  : " + stream);
                System.out.println("    status  : " + status);
                System.out.println("    userData: " + userData);
                System.out.println("    thread  : " + Thread.currentThread());
            }
        };

        // Create some dummy data on the host, and copy it to the device asynchronously
        int n = 100000;
        float hostData[] = new float[n];
        Pointer deviceData = new Pointer();
        JCuda.cudaMalloc(deviceData, n * Sizeof.FLOAT);
        JCuda.cudaMemcpyAsync(deviceData, Pointer.to(hostData), 
            n * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice, stream);

        // Add the callback to the stream that carries the copy operation
        Object userData = "Example user data";
        JCuda.cudaStreamAddCallback(stream, callback, userData, 0);

        // Wait until the stream is finished
        JCuda.cudaStreamSynchronize(stream);
        JCuda.cudaFree(deviceData);
        System.out.println("Done");
    }

    
	@Test
	public void getCudaDetails() throws Exception {
		// CUDA Cores = Number of SMs × Cores per SM
		int[] deviceCount = new int[1];
		JCuda.cudaGetDeviceCount(deviceCount);
		for (int i = 0; i < deviceCount[0]; i++) {
			cudaDeviceProp prop = new cudaDeviceProp();
			JCuda.cudaGetDeviceProperties(prop, i);
			log.info("=== Device " + i + " ===");
//             NVIDIA GeForce GTX 1050
			log.info("Name: " + new String(prop.name).trim());
//            Compute capability: 
//           5.0	Maxwell	GTX 750, 750 Ti	128
//           6.1	Pascal	GTX 1050, 1060	128
//           7.5	Turing	GTX 1650, 1660	64
//           8.6	Ampere	RTX 30xx Laptop GPUs	128 or more
//           9.0+	Hopper/Blackwell	Future server GPUs	Varies
			log.info("Compute capability: " + prop.major + "." + prop.minor);
			log.info("MultiProcessor Count: " + prop.multiProcessorCount);
			log.info("CUDA Cores = " + prop.multiProcessorCount + " SMs × " + 128 + " cores/SM = **"
					+ prop.multiProcessorCount * 128 + " CUDA cores**");
			
		}
	}

}
