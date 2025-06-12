package com.bitflip.cuda;

import static java.math.BigInteger.ONE;

import java.io.File;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.ProviderException;
import java.security.PublicKey;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.bitflip.executors.ParallelCpu;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;

public class CudaGpu {
	private static final Logger log = LoggerFactory.getLogger(CudaGpu.class);

	static {
		LoadNatives.loadLibrariesFromLib("lib");
	}

	public static void start(PublicKey publicKey) {
		JCuda.setExceptionsEnabled(true);
		java.security.interfaces.RSAPublicKey rsaPublicKey = (java.security.interfaces.RSAPublicKey) publicKey;
		// Derived from the product of two prime numbers p * q, both prime numbers are
		BigInteger e = rsaPublicKey.getPublicExponent();
		log.info("PublicExponent::" + e);
		BigInteger n = rsaPublicKey.getModulus();
		log.info("Modulus::" + n);
		int keySize = n.bitLength();
		log.info("KeySize::" + keySize);
		boolean useNew = (keySize >= 2048 && ((keySize & 1) == 0));
		log.info("UseNew::" + useNew);

		BigInteger minValue = ParallelCpu.getSqrt(keySize);
		int lp = (keySize + 1) >> 1;
		int lq = keySize - lp;
		log.info("lp::" + lp);
		log.info("lq::" + lq);
		int pqDiffSize = lp - 100;
		log.info("pqDiffSize::" + pqDiffSize);

		startCuda(keySize);
	}

	private static void startCuda(int keySize) {

		BigInteger bigA = new BigInteger("12345678901234567890");
		BigInteger bigB = new BigInteger("12345678901234567890");
		// not correct !!!
		byte[] product = multiply(bigA.toByteArray(), bigB.toByteArray()); //  returns big-endian bytes
		System.out.println("Result: " + Arrays.toString(product));
		System.out.println("Result: " + new BigInteger(1, reverse(product)).toString());
		System.out.println("Expected: " + bigA.multiply(bigB));
		
		
		byte[] aBytes = stripLeadingZero(bigA.toByteArray());
		byte[] bBytes = stripLeadingZero(bigB.toByteArray());
		
		System.out.println("bigA length: " + bigA.toByteArray().length); 
		System.out.println("bigA bytes: " + Arrays.toString(aBytes));
		System.out.println("a length: " + aBytes.length); 
		System.out.println("a bytes: " + Arrays.toString(aBytes));
		
		
		byte[] productX = xmpMultiply(aBytes, bBytes);
		System.out.println("ResultX: " + Arrays.toString(productX));
		byte[] productXReverse= reverse(productX);
		System.out.println("ResultX Reverse: " + Arrays.toString(productXReverse));
		System.out.println("ResultX: " + new BigInteger(1,productXReverse ).toString());
				
	}

	private static native byte[] xmpMultiply(byte[] a, byte[] b);
	
	private static native void bigintMultiply(ByteBuffer a, int aLen, ByteBuffer b, int bLen, ByteBuffer result,
			int resultLen, int numBlocks, int threadsPerBlock);

	// Removing sign byte when BigInteger.toByteArray()
	private static byte[] stripLeadingZero(byte[] arr) {
	    if (arr.length > 1 && arr[0] == 0) {
	        return Arrays.copyOfRange(arr, 1, arr.length);
	    }
	    return arr;
	}
	
	public static byte[] multiply(byte[] a, byte[] b) {
		byte[] aLE = reverse(a);
		byte[] bLE = reverse(b);
		
	    int aLen = aLE.length;
	    int bLen = bLE.length;
	    int resultLen = aLen + bLen;

	    ByteBuffer aBuf = ByteBuffer.allocateDirect(aLen).order(ByteOrder.nativeOrder());
	    ByteBuffer bBuf = ByteBuffer.allocateDirect(bLen).order(ByteOrder.nativeOrder());
	    ByteBuffer resultBuf = ByteBuffer.allocateDirect(resultLen).order(ByteOrder.nativeOrder());
	    
		// Copy input into direct buffers
		aBuf.put(aLE);
		bBuf.put(bLE);
		aBuf.rewind();
		bBuf.rewind();
		resultBuf.rewind();

		// Configure CUDA kernel dimensions
		int threadsPerBlock = 256;
		int numBlocks = (aLen + threadsPerBlock - 1) / threadsPerBlock;

		// Call native method (mapped via JNI)
		bigintMultiply(aBuf, aLen, bBuf, bLen, resultBuf, resultLen, numBlocks, threadsPerBlock);

		// Extract result back to byte[]
		byte[] result = new byte[resultLen];
		resultBuf.rewind();
		resultBuf.get(result);

		return result;
	}
	
	public static BigInteger bytesToBigInteger(byte[] bytes, boolean littleEndian) {
		if (littleEndian) {
			// Reverse the array for little-endian interpretation
			bytes = reverse(bytes);
		}
		return new BigInteger(1, bytes); // 1 means positive number
	}

	private static byte[] reverse(byte[] array) {
		byte[] reversed = new byte[array.length];
		for (int i = 0; i < array.length; i++) {
			reversed[i] = array[array.length - 1 - i];
		}
		return reversed;
	}
	
//	public static byte[] reverse(byte[] arr) {
//	    for (int i = 0; i < arr.length / 2; i++) {
//	        byte tmp = arr[i];
//	        arr[i] = arr[arr.length - 1 - i];
//	        arr[arr.length - 1 - i] = tmp;
//	    }
//	    return arr;
//	}
	
	
	// GPU uses littleEndian, Java bigEndian
	public static int bytesToInt(byte[] bytes, boolean littleEndian) {
		int offset = 0;
		if (bytes.length < offset + 4) {
			throw new IllegalArgumentException("Not enough bytes to convert to int");
		}
		ByteBuffer buffer = ByteBuffer.wrap(bytes, offset, 4);
		buffer.order(littleEndian ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
		return buffer.getInt();
	}

	// GPU uses littleEndian, Java bigEndian
	public static byte[] intToBytes(int value, boolean littleEndian) {
		ByteBuffer buffer = ByteBuffer.allocate(4); // 4bytes (32 bits)
		// Big-Endian: [0x12][0x34][0x56][0x78] (Left-to-right = MSB to LSB)
		// Little-Endian: [0x78][0x56][0x34][0x12] (Left-to-right = LSB to MSB)
		buffer.order(littleEndian ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
		buffer.putInt(value);
		return buffer.array();
	}

//	public static byte[] multiply(byte[] a, byte[] b) {
//        int aLen = a.length;
//        int bLen = b.length;
//        int resultLen = aLen + bLen;
//        byte[] result = new byte[resultLen];
//
//        // Allocate GPU memory
//        Pointer pA = new Pointer();
//        Pointer pB = new Pointer();
//        Pointer dResult = new Pointer();
//        JCuda.cudaMalloc(pA, aLen * Sizeof.BYTE);
//        JCuda.cudaMalloc(pB, bLen * Sizeof.BYTE);
//        JCuda.cudaMalloc(dResult, resultLen * Sizeof.BYTE);
//
//        // Copy input to GPU
//        JCuda.cudaMemcpy(pA, Pointer.to(a), aLen, cudaMemcpyKind.cudaMemcpyHostToDevice);
//        JCuda.cudaMemcpy(pB, Pointer.to(b), bLen, cudaMemcpyKind.cudaMemcpyHostToDevice);
//
//        // Launch kernel (adjust blocks/threads as needed)
//        int threadsPerBlock = 256;
//        int numBlocks = (aLen + threadsPerBlock - 1) / threadsPerBlock;
//        bigintMultiply(pA, aLen, pB, bLen, dResult, resultLen, numBlocks, threadsPerBlock);
//
//        // Copy result back to CPU
//        JCuda.cudaMemcpy(Pointer.to(result), dResult, resultLen, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//
//        // Free GPU memory
//        JCuda.cudaFree(pA);
//        JCuda.cudaFree(pB);
//        JCuda.cudaFree(dResult);
//
//        return result;
//    }
}

/*
 * OS:Linux - Architecture: x86_64 - Distribution: Ubuntu - Version: 24.04 lspci
 * | grep -i nvidia 01:00.0 VGA compatible controller: NVIDIA Corporation GP107M
 * [GeForce GTX 1050 Mobile] (rev a1) 01:00.1 Audio device: NVIDIA Corporation
 * GP107GL High Definition Audio Controller (rev a1) 
 * 
 * Step1: Cuda toolkit wget
 * https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/
 * cuda-keyring_1.1-1_all.deb sudo dpkg -i cuda-keyring_1.1-1_all.deb sudo
 * apt-get update sudo apt-get -y install cuda-toolkit-12-9 Step2: Cuda Driver
 * sudo apt-get install -y cuda-drivers 
 * 
 * Step3: Confirm nvidia-smi 
 * 
 * Step4: sudo
 * systemctl status nvidia-persistenced 
 * 
 * Step5: sudo find / -name nvcc
 * 2>/dev/null 
 * - export PATH=/usr/local/cuda-12.9/bin:$PATH, each time before execute maven 
 * 
 * OR - export PATH=/usr/local/cuda-12.9/bin:$PATH - source ~/.bashrc, permanent
 * 
 * Step6: https://github.com/NVlabs/xmp
 *  
 * export PATH=/usr/local/cuda-12.9/bin:$PATH
 * - Patching before build: n older versions of CUDA (before 11.0), cudaPointerAttributes had a field called memoryType:
 * - Replace attrib.memoryType with  attrib.type in xmp.cu and logic.cu
 * - In MakeFile , set arch to only device arch: 
 *  ARCH=  -gencode arch=compute_61,code=\"compute_61,sm_61\"
 *  - make clean && make 
 *  - Validate: nm -D libxmp.so | grep xmpIntegersMul
 *  0000000000036480 T xmpIntegersMul, 00000000000349f0 T xmpIntegersMulAsync >> if not, arch might be the issue
 *  
 *
 * exec:java directly from eclipse: 
 * - export LD_LIBRARY_PATH=/home/altug/git/bit-flip/lib:$LD_LIBRARY_PATH
 * - export LD_LIBRARY_PATH=./lib
 * - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/altug/git/bit-flip/lib
 * 
 * Step: Verify
 * ldd /home/altug/git/bit-flip/lib/libbigint_mult_kernel.so
 * nm -D /home/altug/git/bit-flip/lib/libxmp.so | grep xmpIntegersMul
 *  
 * 
 * 
 */
