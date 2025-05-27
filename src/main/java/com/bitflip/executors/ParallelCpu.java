package com.bitflip.executors;


import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.Base64;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.bitflip.BitFlipperApp;
import com.bitflip.utils.Constant;
import com.bitflip.utils.TestUtils;

public class ParallelCpu {

	private static final Logger log = LoggerFactory.getLogger(ParallelCpu.class);
	private static final String message= "message secret first encrpt than decrpt";
	private static KeyFactory keyFactoryBC ;
//	private static final BigInteger TWO = BigInteger.valueOf(2);
//	private static final BigInteger THREE= BigInteger.valueOf(3);
//	private static final BigInteger FIVE= BigInteger.valueOf(5);
	
	public ParallelCpu() {
		try {
			keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
		} catch (NoSuchAlgorithmException | NoSuchProviderException e) {
			log.error("NoSuchAlgorithmException | NoSuchProviderException::" + e.getLocalizedMessage());
		}
	}
	
	public void calculatePrimeNumbers() {
//		TODO: GPU: BigInteger max2048 = BigInteger.valueOf(2).pow(2048); 
		BigInteger max2048 = BigInteger.valueOf(10_000_000);
		BigInteger num;
		try {
			num =TestUtils.readLastBigInteger("Prime","prime1.bin");
			log.info("Last prime number from file is::"+ num);
			num= num.add(BigInteger.ONE);
		} catch (IOException e) {
			log.info("No number represent.");
			num = BigInteger.ONE;  // later read last number from file
			log.info("Initializing number to "+ num);
		}
		
		while(num.compareTo(max2048) < 0) {
			if(isPrime(num)) {
				log.info("PRIME NUMBER IS FOUND"+ num);
				TestUtils.writeBigInteger("Prime","prime1.bin",num.toByteArray());
			}
			num= num.add(BigInteger.ONE);
		}
		
	}
	
	private boolean isPrime(BigInteger num) {
		if(num.compareTo(BigInteger.ONE) <= 0) {
			return false;
		}
		if (num.equals(BigInteger.valueOf(2)) || num.equals(BigInteger.valueOf(3))) {
		    return true;
	    }
		 if (num.mod(BigInteger.valueOf(2)).equals(BigInteger.ZERO) || 
				 num.mod(BigInteger.valueOf(3)).equals(BigInteger.ZERO)) {
			return false;
		 }

	    BigInteger i = BigInteger.valueOf(5);
	    BigInteger sqrt = num.sqrt(); 
		while (i.compareTo(sqrt) <= 0) {
	        if (num.mod(i).equals(BigInteger.ZERO) || 
	        		num.mod(i.add(BigInteger.valueOf(2))).equals(BigInteger.ZERO)) {
	            return false;
	        }
	        i = i.add(BigInteger.valueOf(6));
	    }
	    return true;
	}
	
	public void start(PublicKey publicKey) {
		String decryptedMessage = rsaEncryptionByPublicKey(publicKey);
		// TODO 
		int cores= Runtime.getRuntime().availableProcessors();
		log.info("Available CPU cores::" + cores);
		BigInteger modulusPuiblic = generatePrivateKey(publicKey);
		log.info("modulusPuiblic::"+modulusPuiblic.toString()); // n
		
		ExecutorService	executorService=	Executors.newFixedThreadPool(cores);
		
		
		byte[] privateKeyBytes = TestUtils.readData("RSA", "private.key");
		rsaEncryptionByPublicKey(decryptedMessage,privateKeyBytes);
	}
	
	private BigInteger generatePrivateKey(PublicKey publicKey) {
		java.security.interfaces.RSAPublicKey rsaPublicKey = (java.security.interfaces.RSAPublicKey) publicKey;
		return rsaPublicKey.getModulus(); // n
	}
	
	private String rsaEncryptionByPublicKey(String encryptedMessage, byte[] privateKeyBytes) {
		try {
			//KeyFactory keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
			PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(privateKeyBytes);
			PrivateKey privateK = keyFactoryBC.generatePrivate(keySpec);

			Cipher decryptCipher = Cipher.getInstance(Constant.RSA_ECB_OAEPWithSHA_256AndMGF1Padding);

			decryptCipher.init(Cipher.DECRYPT_MODE, privateK);
			byte[] encryptedMessageBytes = Base64.getDecoder().decode(encryptedMessage);

			byte[] decryptedMessageBytes = decryptCipher.doFinal(encryptedMessageBytes);
			String decryptedMessage = new String(decryptedMessageBytes, StandardCharsets.UTF_8);
			log.info("decryptedMessage: " + decryptedMessage);
			return decryptedMessage;
		} catch (NoSuchAlgorithmException e) {
			log.error("NoSuchAlgorithmException::" + e.getLocalizedMessage());
		}  catch (InvalidKeySpecException e) {
			log.error("InvalidKeySpecException::" + e.getLocalizedMessage());
		} catch (NoSuchPaddingException e) {
			log.error("NoSuchPaddingException::" + e.getLocalizedMessage());
		} catch (InvalidKeyException e) {
			log.error("InvalidKeyException::" + e.getLocalizedMessage());
		} catch (IllegalBlockSizeException e) {
			log.error("IllegalBlockSizeException::" + e.getLocalizedMessage());
		} catch (BadPaddingException e) {
			log.error("BadPaddingException::" + e.getLocalizedMessage());
		}
		return null;
	}
	
	
	/*- "RSA/ECB/NoPadding", use when bit flip 
	 * - "PKCS1Padding", legacy systems -
	 * "RSA/ECB/OAEPWithSHA-256AndMGF1Padding", modern version
	 */
	private String rsaEncryptionByPublicKey(PublicKey publicKey) {
		try {
			Cipher encryptCipher = Cipher.getInstance(Constant.RSA_ECB_OAEPWithSHA_256AndMGF1Padding);
			encryptCipher.init(Cipher.ENCRYPT_MODE, publicKey);
			byte[] encryptedMessageBytes = encryptCipher.doFinal(message.getBytes(StandardCharsets.UTF_8));
			String encryptedMessage = Base64.getEncoder().encodeToString(encryptedMessageBytes);
			log.info("encryptedMessage: " + encryptedMessage);
			return encryptedMessage;
		} catch (NoSuchAlgorithmException e) {
			log.error("NoSuchAlgorithmException::" + e.getLocalizedMessage());
		} catch (NoSuchPaddingException e) {
			log.error("NoSuchPaddingException::" + e.getLocalizedMessage());
		} catch (InvalidKeyException e) {
			log.error("InvalidKeyException::" + e.getLocalizedMessage());
		} catch (IllegalBlockSizeException e) {
			log.error("IllegalBlockSizeException::" + e.getLocalizedMessage());
		} catch (BadPaddingException e) {
			log.error("BadPaddingException::" + e.getLocalizedMessage());
		}
		return null;
	}

	
}
