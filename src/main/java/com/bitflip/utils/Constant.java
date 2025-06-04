package com.bitflip.utils;

import static java.math.BigInteger.TWO;

import java.math.BigInteger;
import java.util.List;

public class Constant {
	
	public static final String RSA= "RSA";
	public static final String BC= "BC";
	public static final String CONSCRYPT="Conscrypt";
	public static final String DSA="DSA";
	public static final String EC="EC";
	public static final String Ed25519= "Ed25519";
	public static final String Ed448= "Ed448";
	public static final String X25519= "X25519";
	public static final String X448= "X448";
	public static final String DiffieHellman= "DiffieHellman";
	
	public static final List<String> algorithms= List.of(
	        RSA, // Asymmetric (Public-Key), Encryption, TLS
	        DSA, // Digital Signature Algorithm, Asymmetric, for signatures only, older SSH/TLS
	        EC,  // Elliptic Curve),  Elliptic Curve Cryptography, Key agreement (ECDH), digital signatures (ECDSA)       
	        Ed25519,// Elliptic Curve Cryptography (ECC) , signature only, Digital signatures (modern SSH, TLS, JWTs)
	        Ed448,  // Elliptic Curve Cryptography (ECC) , signature only, Digital signatures (modern SSH, TLS, JWTs), longer keys
	        X25519, // Elliptic Curve Cryptography (ECC) , key exchange (not signature),  ECDH-based key exchange (TLS 1.3, Signal protocol), Designed specifically for Diffie-Hellman over Curve25519
	        X448, // Elliptic Curve Cryptography (ECC) , key exchange (not signature), longer keys than X25519 , Uses Curve448, suitable for long-term secure communication
	        DiffieHellman // Legacy Asymmetric key exchange,  Key agreement in early TLS versions, VPNs

	    );
	public static final String RSA_ECB_PKCS1Padding= "RSA/ECB/PKCS1Padding";
	public static final String RSA_ECB_OAEPWithSHA_1AndMGF1Padding= "RSA/ECB/OAEPWithSHA-1AndMGF1Padding";
	public static final String RSA_ECB_OAEPWithSHA_256AndMGF1Padding= "RSA/ECB/OAEPWithSHA-256AndMGF1Padding";
	public static final String RSA_ECB_NoPadding= "RSA/ECB/NoPadding";
	public static final BigInteger SQRT_2048 = TWO.pow(2047).sqrt();
	public static final BigInteger SQRT_3072 = TWO.pow(3071).sqrt();
	public static final BigInteger SQRT_4096 = TWO.pow(4095).sqrt();
	
}
