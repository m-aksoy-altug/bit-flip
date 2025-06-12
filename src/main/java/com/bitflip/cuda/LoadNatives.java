package com.bitflip.cuda;

import java.io.File;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LoadNatives {
	
	private static final Logger log = LoggerFactory.getLogger(LoadNatives.class);

	public static void loadLibrariesFromLib(String path) {
		String libDirPath = System.getProperty("user.dir") + "/"+ path;
		log.info("libDirPath"+libDirPath);
		File libDir = new File(libDirPath);

		if (!libDir.exists() || !libDir.isDirectory()) {
			throw new RuntimeException("Directory not found: " + libDirPath);
		}

	
		File[] libFiles = libDir.listFiles((dir, name) -> {
			return name.endsWith(".so") || name.endsWith(".dll");
		});

		if (libFiles == null || libFiles.length == 0) {
			throw new RuntimeException("No libraries found in " + libDirPath);
		}
		
		Arrays.sort(libFiles, (f1, f2) -> f2.getName().compareTo(f1.getName()));
		
		// Order is important, native that contains headers must load first 
		for (File libFile : libFiles) {  
			try {
//				System.loadLibrary(libFile.getAbsolutePath());
				System.load(libFile.getAbsolutePath());
				log.info("Loaded: " + libFile.getName());
			} catch (UnsatisfiedLinkError e) {
				log.error("Failed to load: " + libFile.getName());
				e.printStackTrace();
				throw e;
			}
		}
	}

}
