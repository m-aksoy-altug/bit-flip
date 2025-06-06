package com.bitflip.executors;

import static java.math.BigInteger.ONE;
import static java.math.BigInteger.TWO;
import static java.math.BigInteger.ZERO;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.ProviderException;
import java.security.PublicKey;
import java.security.SecureRandom;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.RSAKeyGenParameterSpec;
import java.security.spec.RSAPrivateKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;

import org.bouncycastle.crypto.RuntimeCryptoException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.bitflip.BitFlipperApp;
import com.bitflip.utils.Constant;
import com.bitflip.utils.TestUtils;

public class ParallelCpu {

	private static final Logger log = LoggerFactory.getLogger(ParallelCpu.class);
	private static final String message = "message secret first encrpt than decrpt";
	private static KeyFactory keyFactoryBC;

	public ParallelCpu() {
		try {
			keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
		} catch (NoSuchAlgorithmException | NoSuchProviderException e) {
			log.error("NoSuchAlgorithmException | NoSuchProviderException::" + e.getLocalizedMessage());
		}
	}

	public void start(PublicKey publicKey) {
		String decryptedMessage = rsaEncryptionByPublicKey(publicKey);
		byte[] privateKeyBytes = TestUtils.readData("RSA", "private.key");
		PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(privateKeyBytes);
		try {
			PrivateKey privateKey = keyFactoryBC.generatePrivate(keySpec);
			java.security.interfaces.RSAPrivateKey rsaPrivateKey = 
					(java.security.interfaces.RSAPrivateKey) privateKey;
		    BigInteger modulusPrivate = rsaPrivateKey.getModulus(); // n
			log.info("Searching for n::" + modulusPrivate);
		} catch (InvalidKeySpecException e) {}
		
		int cores = Runtime.getRuntime().availableProcessors();
		log.info("Available CPU cores::" + cores);
		ExecutorService executorService = Executors.newFixedThreadPool(cores);
		List<Future<PrivateKey>> futures = new ArrayList<>();
		PrivateKey pkFound = null;
		for (int i = 0; i < cores; i++) {
			futures.add(executorService.submit(() -> randomBruteForceForPrimes(publicKey)));
		}

		try {
			for (Future<PrivateKey> future : futures) {
				try {
					pkFound = future.get();
					break;
				} catch (ExecutionException e) {
					log.error("ExecutionException" + e.getMessage());
				}
			}
		} catch (InterruptedException e) {
			log.error("InterruptedException" + e.getMessage());
			Thread.currentThread().interrupt();
		} finally {
			futures.forEach(f -> f.cancel(true));
		}
		if (pkFound == null)
			throw new RuntimeException("PrivateKey pkFound==null");

		rsaDencryptionByPublicKey(decryptedMessage, pkFound.getEncoded());
	}

	// FIPS 186-4 B.3.3 / FIPS 186-5 A.1.3
	// Generation of Random Primes that are Probably Prime
	public PrivateKey randomBruteForceForPrimes(PublicKey publicKey) {
		SecureRandom random = new SecureRandom(); // Auto-seeds using OS entropy source
		if (random.getAlgorithm().isEmpty())
			throw new RuntimeException("SecureRandom is null!!");
		log.info("random Algorithm()::" + random.getAlgorithm());
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

		BigInteger minValue = getSqrt(keySize);
		int lp = (keySize + 1) >> 1;
		int lq = keySize - lp;
		log.info("lp::" + lp); log.info("lq::" + lq);
		int pqDiffSize = lp - 100;
		log.info("pqDiffSize::" + pqDiffSize);

		while (true) {
			BigInteger p = null;
			BigInteger q = null;

			int i = 0;
			while (i++ < 10 * lp) {
				BigInteger tmpP = BigInteger.probablePrime(lp, random);
				if ((!useNew || tmpP.compareTo(minValue) == 1) 
						&& isRelativePrime(e, tmpP.subtract(ONE))) {
					p = tmpP;
					break;
				}
			}
			if (p == null)
				throw new ProviderException("Cannot find prime P");

			i = 0;
			while (i++ < 20 * lq) {
				BigInteger tmpQ = BigInteger.probablePrime(lq, random);

				if ((!useNew || tmpQ.compareTo(minValue) == 1)
						&& (p.subtract(tmpQ).abs().compareTo(TWO.pow(pqDiffSize)) == 1)
						&& isRelativePrime(e, tmpQ.subtract(ONE))) {
					q = tmpQ;
					break;
				}
			}
			if (q == null) {
				throw new ProviderException("Cannot find prime Q");
			}
			// log.info("q::" + q);log.info("q::" + q.toString().length());
			BigInteger generatedN = p.multiply(q);
			// log.info("p::" + p.toString().length());log.info("q::" + q.toString().length());
			log.info("generatedN::" + generatedN);
			if (generatedN.bitLength() != keySize)
				throw new RuntimeException("generated.bitLength() != keySize");
			if (n.compareTo(generatedN) == 0) {
				log.info("P::" + p);
				log.info("Q::" + q);
				BigInteger phi = (p.subtract(BigInteger.ONE)).multiply(q.subtract(BigInteger.ONE));
				BigInteger d = e.modInverse(phi);
				log.info("PrivateExponent::" + d);
				KeyFactory kf = keyFactoryBC;
				RSAPrivateKeySpec privateSpec = new RSAPrivateKeySpec(n, d);
				try {
					PrivateKey privateKey = kf.generatePrivate(privateSpec);
					log.info("Private key: " + privateKey);
					return privateKey;
				} catch (InvalidKeySpecException e1) {
					log.error("InvalidKeySpecException::" + e1.getLocalizedMessage());
				}
			}
		}

	}

	private static boolean isRelativePrime(BigInteger e, BigInteger bi) {
		// optimize for common known public exponent prime values
		// The public exponent-value F4 = 65537. & The public-exponent value F0 = 3.
		if (e.compareTo(RSAKeyGenParameterSpec.F4) == 0 || e.compareTo(RSAKeyGenParameterSpec.F0) == 0) {
			return !bi.mod(e).equals(ZERO);
		} else {
			return e.gcd(bi).equals(ONE);
		}
	}

	private static BigInteger getSqrt(int keySize) {
		BigInteger sqrt;
		switch (keySize) {
		case 2048:
			sqrt = Constant.SQRT_2048;
			break;
		case 3072:
			sqrt = Constant.SQRT_3072;
			break;
		case 4096:
			sqrt = Constant.SQRT_4096;
			break;
		default:
			sqrt = TWO.pow(keySize - 1).sqrt();
		}
		return sqrt;
	}


	private String rsaDencryptionByPublicKey(String encryptedMessage, byte[] privateKeyBytes) {
		try {
			// KeyFactory keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
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
		} catch (InvalidKeySpecException e) {
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
