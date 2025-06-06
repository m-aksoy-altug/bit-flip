package com.bitflip.cuda;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class Memory  {
   
	private final MemoryType type;
	// The pointer to the actual memory
    private final Pointer pointer;
    // The buffer for the memory, if it is no device memory
    private final FloatBuffer buffer;

    /* Creates a block of memory with the given type and size
     * @param type The {@link MemoryType}
     * @param numBytes The size of the memory, in bytes
     */
    Memory(MemoryType type, int numBytes){
        this.type = type;
        switch (type){
            case DEVICE: {
                // Allocate device memory
                pointer = new Pointer();
                buffer = null;
                JCuda.cudaMalloc(pointer, numBytes);
                break;
            }
            case HOST_PINNED:{
                // Allocate pinned (page-locked) host memory
                pointer = new Pointer();
                JCuda.cudaHostAlloc(pointer, numBytes, 
                		JCuda.cudaHostAllocWriteCombined);
                ByteBuffer byteBuffer = pointer.getByteBuffer(0, numBytes);
                byteBuffer.order(ByteOrder.nativeOrder());
                buffer = byteBuffer.asFloatBuffer();
                break;
            }
            case HOST_PAGEABLE_ARRAY:{
                // The host memory is pageable and stored in a Java array
                byte array[] = new byte[numBytes];
                ByteBuffer byteBuffer = ByteBuffer.wrap(array);
                byteBuffer.order(ByteOrder.nativeOrder());
                buffer = byteBuffer.asFloatBuffer();
                pointer = Pointer.to(array);
                break;
            }
            default:
            case HOST_PAGEABLE_DIRECT:{
                // The host memory is pageable and stored in a direct byte buffer
                ByteBuffer byteBuffer = 
                    ByteBuffer.allocateDirect(numBytes);
                byteBuffer.order(ByteOrder.nativeOrder());
                buffer = byteBuffer.asFloatBuffer();
                pointer = Pointer.to(buffer);
            }
        }
    }

    /* Put the data from the given source array into this memory
     * @param source The source array
     */
    public void put(float source[]){
        if (type == MemoryType.DEVICE){
            JCuda.cudaMemcpy(pointer, Pointer.to(source), 
                source.length * Sizeof.FLOAT, 
                	cudaMemcpyKind.cudaMemcpyHostToDevice);
        }else{
            buffer.put(source);
            buffer.rewind();
        }
    }

    /** Write data from this memory into the given target array
     * @param target The target array
     */
    public void get(float target[]){
        if (type == MemoryType.DEVICE){
        	JCuda.cudaMemcpy(Pointer.to(target), pointer, 
                target.length * Sizeof.FLOAT, 
                	cudaMemcpyKind.cudaMemcpyDeviceToHost);
        }else{
            buffer.get(target);
            buffer.rewind();
        }
    }

    /** Returns the pointer to this memory
     * @return The pointer
     */
    public Pointer getPointer(){
        return pointer;
    }

    // Release this memory
    public void release(){
        if (type == MemoryType.DEVICE){
            JCuda.cudaFree(pointer);
        }
        else if (type == MemoryType.HOST_PINNED){
            JCuda.cudaFreeHost(pointer);
        }
    }
    
	/*
	 * CPU (host) & GPU (device).
	*/
    public static int getCudaMemcpyKind(MemoryType targetType,MemoryType sourceType){
        if (targetType == MemoryType.DEVICE){
            if (sourceType == MemoryType.DEVICE){
                return cudaMemcpyKind.cudaMemcpyDeviceToDevice;
            }
            //Copy memory from the CPU (host) to the GPU (device).
            return cudaMemcpyKind.cudaMemcpyHostToDevice;
        }
        if (sourceType == MemoryType.DEVICE){
            return cudaMemcpyKind.cudaMemcpyDeviceToHost;
        }
        // Copy memory from one CPU (host) location to another CPU location.
        return cudaMemcpyKind.cudaMemcpyHostToHost;
    }

    
}