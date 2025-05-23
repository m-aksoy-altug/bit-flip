package com.bitflip.executors;

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

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.bitflip.BitFlipperApp;
import com.bitflip.utils.Constant;
import com.bitflip.utils.TestUtils;

public class Parallel {

	private static final Logger log = LoggerFactory.getLogger(BitFlipperApp.class);
	private static final String message= "message secret first encrpt than decrpt";
	private static KeyFactory keyFactoryBC ;
	
	public Parallel() {
		try {
			keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
		} catch (NoSuchAlgorithmException | NoSuchProviderException e) {
			log.error("NoSuchAlgorithmException | NoSuchProviderException::" + e.getLocalizedMessage());
		}
	}
	
	
	public void start(byte[] publicKeyBytes) {
		String decryptedMessage = rsaEncryptionByPublicKey(publicKeyBytes);
		// TODO 
		byte[] privateKeyBytes = TestUtils.readData("RSA", "private.key");
		rsaEncryptionByPublicKey(decryptedMessage,privateKeyBytes);
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
	private String rsaEncryptionByPublicKey(byte[] publicKeyBytes) {
		try {
			X509EncodedKeySpec keySpec = new X509EncodedKeySpec(publicKeyBytes);
			//KeyFactory keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
			PublicKey publicKeyBC = keyFactoryBC.generatePublic(keySpec);
			Cipher encryptCipher = Cipher.getInstance(Constant.RSA_ECB_OAEPWithSHA_256AndMGF1Padding); // is older
																										// version
			encryptCipher.init(Cipher.ENCRYPT_MODE, publicKeyBC);
			byte[] encryptedMessageBytes = encryptCipher.doFinal(message.getBytes(StandardCharsets.UTF_8));
			String encryptedMessage = Base64.getEncoder().encodeToString(encryptedMessageBytes);
			log.info("encryptedMessage: " + encryptedMessage);
			return encryptedMessage;
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

	
}
