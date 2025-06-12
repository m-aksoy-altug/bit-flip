#include <jni.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>
#include "xmp.h"

auto pack_bytes_to_words = [](const std::vector<uint8_t>& bytes, std::vector<uint32_t>& words) {
    size_t byte_len = bytes.size();
    for (size_t i = 0; i < byte_len; ++i) {
        size_t byte_pos = byte_len - 1 - i; // reverse byte index (from big-endian)
        size_t word_index = i / 4;
        size_t byte_index = i % 4;
        words[word_index] |= static_cast<uint32_t>(bytes[byte_pos]) << (8 * byte_index);
    }
};

#define CHECK_XMP(err, msg) \
    if ((err) != xmpErrorSuccess) { \
        printf("XMP Error: %s failed with code %d\n", msg, err); \
        xmpHandleDestroy(handle); \
        return nullptr; \
    }
    
extern "C" {
	JNIEXPORT jbyteArray JNICALL
	Java_com_bitflip_cuda_CudaGpu_xmpMultiply(JNIEnv *env, jclass clazz, jbyteArray a_, jbyteArray b_) {
		// device_check();
		jsize a_len = env->GetArrayLength(a_);
	    jsize b_len = env->GetArrayLength(b_);
	    jsize a_bits = a_len * 8;
	    jsize b_bits = b_len * 8;
	    printf("Array a: %d bytes (%d bits)\n", a_len, a_bits);
	    printf("Array b: %d bytes (%d bits)\n", b_len, b_bits);
	    
	    // uint8_t matchs jbyte
	    std::vector<uint8_t> a_bytes(a_len); // big-endian because of jbyte
	    std::vector<uint8_t> b_bytes(b_len);
	    env->GetByteArrayRegion(a_, 0, a_len, reinterpret_cast<jbyte*>(a_bytes.data()));
	    env->GetByteArrayRegion(b_, 0, b_len, reinterpret_cast<jbyte*>(b_bytes.data()));
	
	    xmpHandle_t handle;
	    CHECK_XMP(xmpHandleCreate(&handle), "xmpHandleCreate");
	    
	    uint32_t precision = std::max(a_len, b_len) * 8;  // bits
	    xmpIntegers_t x_a, x_b, x_res;
	    uint32_t count = 1;
		
		// Calculate limbs based on input size (assuming 32-bit limbs for XMP)
		uint32_t limbs = (precision + 31) / 32;  
		// uint32_t limbs = precision / 32;  // 64 bits / 32 bits per limb = 2 limbs
		printf("limbs=%u \n", limbs);
			
		printf("Creating x_a: a_len=%d, b_len=%d, precision=%u, count=%u\n", a_len, b_len, precision, count);
		CHECK_XMP(xmpIntegersCreate(handle, &x_a, precision, count), "xmpIntegersCreate a");
		CHECK_XMP(xmpIntegersCreate(handle, &x_b, precision, count), "xmpIntegersCreate b");
		CHECK_XMP(xmpIntegersCreate(handle, &x_res, precision * 2, count), "xmpIntegersCreate res");
	
	    printf("Importing x_a&b: sizeof(uint8_t)=%zu, a_bytes.data()=%p, b_bytes.data()=%p, count=%u\n", sizeof(uint32_t),  (void*)a_bytes.data(),  (void*)b_bytes.data(), count);
		
		std::vector<uint32_t> a_words(limbs, 0);
		std::vector<uint32_t> b_words(limbs, 0);
		
		// Fill a_words from a_bytes (big-endian)
		pack_bytes_to_words(a_bytes, a_words);
		pack_bytes_to_words(b_bytes, b_words);

		CHECK_XMP(xmpIntegersImport(handle, x_a, limbs, -1, sizeof(uint32_t), 0, 0, a_words.data(), count), "xmpIntegersImport a");
		CHECK_XMP(xmpIntegersImport(handle, x_b, limbs, -1, sizeof(uint32_t), 0, 0, b_words.data(), count), "xmpIntegersImport b");
	
	    CHECK_XMP(xmpIntegersMul(handle, x_res, x_a, x_b, count), "xmpIntegersMul");
	    
	 
	 	 // The result will be up to 2× limbs in 32-bit words
		uint32_t result_words = limbs * 2;
		printf("result_words=%u \n", result_words);
		std::vector<uint32_t> result_data(result_words);
		
		// Export as 32-bit words
		CHECK_XMP(xmpIntegersExport(handle,result_data.data(),&result_words,-1,sizeof(uint32_t),0,0,x_res,count),"xmpIntegersExport");
	          
		std::vector<uint8_t> result_bytes(result_words * sizeof(uint32_t));
		
		for (size_t i = 0; i < result_words; ++i) {
		    uint32_t word = result_data[i];
		    result_bytes[i * 4 + 0] = word & 0xFF;
		    result_bytes[i * 4 + 1] = (word >> 8) & 0xFF;
		    result_bytes[i * 4 + 2] = (word >> 16) & 0xFF;
		    result_bytes[i * 4 + 3] = (word >> 24) & 0xFF;
		}

	
		// Trim leading 0s (most significant bytes)
		size_t first_non_zero = 0;
		while (first_non_zero < result_bytes.size() && result_bytes[first_non_zero] == 0) {
		    ++first_non_zero;
		}
		
		std::vector<uint8_t> trimmed_bytes(result_bytes.begin() + first_non_zero, result_bytes.end());
	
	    jbyteArray result = env->NewByteArray(trimmed_bytes.size());
	    if (result == nullptr) return nullptr;
	    env->SetByteArrayRegion(result, 0, trimmed_bytes.size(), reinterpret_cast<jbyte*>(trimmed_bytes.data()));
	
	    xmpIntegersDestroy(handle, x_a);
	    xmpIntegersDestroy(handle, x_b);
	    xmpIntegersDestroy(handle, x_res);
	    xmpHandleDestroy(handle);
	
	    return result;
	    }
}

auto device_check=[](){
	int deviceCount;
	cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
	if (cudaErr == cudaSuccess && deviceCount > 0) {
	    printf("CUDA GPU detected: %d device(s) available.\n", deviceCount);
	    int device;
	    cudaGetDevice(&device);
	    cudaDeviceProp prop;
	    cudaGetDeviceProperties(&prop, device);
	    printf("Running on GPU: %s\n", prop.name);
	} else {
	    printf("No CUDA GPU detected (error: %s).\n", cudaGetErrorString(cudaErr));
	}
};
