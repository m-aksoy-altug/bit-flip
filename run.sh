#!/bin/bash

# chmod +x run.sh  # ./run.sh
# Absolute path to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

# Set JAVA_HOME
export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"

# Set MAVEN_HOME to Maven 3.5.4
export MAVEN_HOME="/home/altug/.m2/wrapper/dists/apache-maven-3.5.4-bin/4lcg54ki11c6mp435njk296gm5/apache-maven-3.5.4"
MVN="$MAVEN_HOME/bin/mvn"

# Verify JAVA_HOME points to a JDK
if [ ! -x "$JAVA_HOME/bin/javac" ]; then
  echo " JAVA_HOME ($JAVA_HOME) does not point to a valid JDK. Please set JAVA_HOME to a JDK directory." >&2
  exit 1
fi

# Verify Maven exists
if [ ! -x "$MVN" ]; then
  echo "Maven not found at $MVN. Please verify the Maven installation." >&2
  exit 1
fi

# Execute the Java app with native lib path and local repository
LD_LIBRARY_PATH="$LIB_DIR" "$MVN" \
		exec:java -Dexec.mainClass="com.bitflip.BitFlipperApp" \
		-Dexec.args="" -Djava.library.path=./lib  \
		-Djava.library.path="$LIB_DIR" \
		-Dmaven.repo.local=/home/altug/.m2/repository "$@"