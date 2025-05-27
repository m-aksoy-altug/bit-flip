package com.bitflip.utils;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import com.sun.tools.javac.Main;

public class TestUtils {
	
	public static void writeData(String path, String fileName, byte[] writeBytes) {
		Path filePath= Paths.get(path,fileName);
		 try {
			Files.createDirectories(Paths.get(path));
		} catch (IOException e) {
			e.printStackTrace();
		}
		try (FileOutputStream fos = new FileOutputStream(filePath.toAbsolutePath().toString())) {
		    fos.write(writeBytes); // raw 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public static void writeBigInteger(String path, String fileName, byte[] writeBytes) {
		Path filePath= Paths.get(path,fileName);
		int length = writeBytes.length; // 2084
		
		ByteBuffer buffer= ByteBuffer.allocate(length+ 4);  
		buffer.put(writeBytes);
		buffer.putInt(length);
	    // [BigInteger bytes][length] 
	     try {
			Files.createDirectories(Paths.get(path));
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
		    Files.write(filePath, buffer.array(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/*
	 * TODO: Handle file size
	*/
	public static List<BigInteger> readBigIntegersFromFile(String path, String fileName) throws IOException {
		Path filePath= Paths.get(path,fileName);
	    byte[] fileBytes = Files.readAllBytes(filePath);
	    ByteBuffer buffer = ByteBuffer.wrap(fileBytes);
	    List<BigInteger> numbers = new ArrayList<>();

	    while (buffer.remaining() >= 4) {
	        int length = buffer.getInt();

	        // Check for valid length
	        if (length < 0 || buffer.remaining() < length) {
	            throw new IOException("Corrupted file or unexpected length: " + length);
	        }

	        byte[] numberBytes = new byte[length];
	        buffer.get(numberBytes);
	        BigInteger number = new BigInteger(numberBytes);
	        numbers.add(number);
	    }

	    return numbers;
	}
	
	public static BigInteger readLastBigInteger(String path, String fileName) throws IOException {
		 Path filePath= Paths.get(path,fileName);
		 try {
				Files.createDirectories(Paths.get(path));
			} catch (IOException e) {
				e.printStackTrace();
			}
		try (FileChannel channel = FileChannel.open(filePath, StandardOpenOption.READ)) {
	        long fileSize = channel.size();

	        if (fileSize < 4) {
	            throw new IOException("File too small to contain a BigInteger");
	        }

	        // Read last 4 bytes to get the length  [BigInteger bytes][length]
	        ByteBuffer lengthBuffer = ByteBuffer.allocate(4);
	        channel.position(fileSize - 4);
	        channel.read(lengthBuffer);
	        lengthBuffer.flip();
	        int length = lengthBuffer.getInt();

	        if (fileSize < (4 + length)) {
	            throw new IOException("File corrupted or incomplete BigInteger at end");
	        }

	        // Read the BigInteger bytes
	        ByteBuffer dataBuffer = ByteBuffer.allocate(length);
	        channel.position(fileSize - 4 - length);
	        channel.read(dataBuffer);
	        dataBuffer.flip();

	        byte[] numberBytes = new byte[length];
	        dataBuffer.get(numberBytes);

	        return new BigInteger(numberBytes);
	    }
	}
	
	public static byte[] readData(String path, String fileName) {
		Path filePath= Paths.get(path,fileName);
		try {
			 return Files.readAllBytes(filePath); // raw
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
}
