package com.bitflip;

import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Security;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.Base64;
import java.util.List;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;

import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.conscrypt.Conscrypt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.bitflip.executors.Parallel;
import com.bitflip.utils.Constant;
import com.bitflip.utils.TestUtils;

public class BitFlipperApp {
	// exec:java -Dexec.mainClass="com.bitflip.BitFlipperApp"
	private static final Logger log = LoggerFactory.getLogger(BitFlipperApp.class);

	public static void main(String[] args) {
		Security.addProvider(new BouncyCastleProvider());
		Security.insertProviderAt(Conscrypt.newProvider(), Security.getProviders().length + 1);
		byte[] publicKeyBytes = null;
		if (args.length != 0) {
			publicKeyBytes = Base64.getDecoder().decode(args[0]);
		} else {
			publicKeyBytes = TestUtils.readData(Constant.RSA, "public.key");
		}

		if (findAlgortihm(publicKeyBytes).equals(Constant.RSA)) {
			// rsaPublicKeyDetailsByVendor(publicKeyBytes);
			Parallel parallel= new Parallel();
			parallel.start(publicKeyBytes);
		}
	}

	
	private static String findAlgortihm(byte[] publicKeyBytes) {
		for (String each : Constant.algorithms) {
			try {
				X509EncodedKeySpec spec = new X509EncodedKeySpec(publicKeyBytes);
				KeyFactory factory;
				if (List.of(Constant.Ed25519, Constant.Ed448, Constant.X25519, Constant.X448, Constant.DiffieHellman)
						.contains(each)) {
					factory = KeyFactory.getInstance(each, Constant.BC);
				} else {
					factory = KeyFactory.getInstance(each);
				}
				PublicKey publicKey = factory.generatePublic(spec);
				return each;
			} catch (Exception silentIgnore) {
			}
		}
		throw new RuntimeException("Unknown key algorithm or unsupported format");
	}

	private static void rsaPublicKeyDetailsByVendor(byte[] publicKeyBytes) {
		try {
			X509EncodedKeySpec keySpec = new X509EncodedKeySpec(publicKeyBytes);
			KeyFactory keyFactory = KeyFactory.getInstance(Constant.RSA);
			PublicKey publicKey = keyFactory.generatePublic(keySpec);
			log.info("publicKey.toString()" + publicKey.toString());
			KeyFactory keyFactoryBC = KeyFactory.getInstance(Constant.RSA, Constant.BC);
			PublicKey publicKeyBC = keyFactoryBC.generatePublic(keySpec);
			log.info("publicKeyBC.toString()" + publicKeyBC.toString());
			KeyFactory keyFactoryConscrypt = KeyFactory.getInstance(Constant.RSA, Constant.CONSCRYPT);
			PublicKey publicKeyConscrypt = keyFactoryConscrypt.generatePublic(keySpec);
			log.info("publicKeyConscrypt.toString()" + publicKeyConscrypt.toString());
		} catch (InvalidKeySpecException e) {
			log.error("Not a valid public key");
		} catch (NoSuchAlgorithmException e) {
			log.error("Not a valid algorithm");
		} catch (NoSuchProviderException e) {
			log.error("Not a valid provider for bouncycastle");
		}
	}

}
