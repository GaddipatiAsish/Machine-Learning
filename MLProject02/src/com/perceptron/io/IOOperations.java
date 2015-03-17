package com.perceptron.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

/**
 * IOOperations class reads and writes data from/to inputData and output folders
 * from/in respective files
 * 
 * @author AsishKumar
 *
 */
public class IOOperations {
	/* vector that stores all the features of all input data fi(X) */
	List<List<Double>> featureVector = new ArrayList<List<Double>>();
	/* vector that stores all the features of single input data fi(Xn) */
	List<Double> featureOfXi;
	/* C(X) : list that has the respective class for each Xi */
	List<Integer> targetClasses = new ArrayList<Integer>();

	/**
	 * @description readInputData method reads the input data from the file
	 *              thats passed as argument and fills the featureVector and
	 *              featureOfXi vectors.
	 * 
	 * @param arg
	 */
	public void readInputData(String arg) throws IOException {
		String row;
		BufferedReader breader = new BufferedReader(new FileReader(
				new File(arg)));
		while ((row = breader.readLine()) != null) {

			List<Double> featureOfXi = new ArrayList<Double>();
			/* find the position of last occurring comma */
			int lastCommaPosition = row.lastIndexOf(",");
			/* adding the corresponding class Ci that the Xi belongs */
			targetClasses.add(Integer.parseInt(row.substring(
					lastCommaPosition + 1, row.length())));
			/* breaks the row with delimiter as , */
			StringTokenizer tokenizer = new StringTokenizer(row.substring(0,
					lastCommaPosition), ",");
			while (tokenizer.hasMoreElements()) {
				featureOfXi.add(Double.parseDouble(tokenizer.nextElement()
						.toString()));
			}
			/* appending the feature vector for all Xi's */
			featureVector.add(featureOfXi);
		}
		breader.close();
	}

	public List<List<Double>> getFeatureVector() {
		return featureVector;
	}

	public List<Integer> getTargetVector() {
		return targetClasses;
	}

	public Set<Integer> getKClasses() {
		return new HashSet<Integer>(targetClasses);
	}

	public void appendFile(List<BufferedWriter> bWriters, String newLine,
			int digit) throws IOException {

		for (int i = 0; i < bWriters.size(); i++) {
			if (i == digit) {
				String newLine1 = newLine + ",1";
				bWriters.get(i).write(newLine1);
				bWriters.get(i).newLine();
			} else {
				String newLine2 = newLine + ",-1";
				bWriters.get(i).write(newLine2);
				bWriters.get(i).newLine();
			}
		}
	}

	public void generateClassificationFiles(String arg) throws IOException {
		List<File> trainingFiles = new ArrayList<File>();
		List<File> developmentFiles = new ArrayList<File>();

		for (int i = 0; i <= 9; i++) {
			trainingFiles.add(new File("./inputData/"+i + ".tra"));
			developmentFiles.add(new File("./inputData/"+i + ".dev"));
		}
		List<BufferedWriter> bWritersT = new ArrayList<BufferedWriter>();
		List<BufferedWriter> bWritersD = new ArrayList<BufferedWriter>();

		for (int i = 0; i <= 9; i++) {
			bWritersT.add(new BufferedWriter(new FileWriter(trainingFiles
					.get(i))));
			bWritersD.add(new BufferedWriter(new FileWriter(developmentFiles
					.get(i))));
		}

		String line;
		BufferedReader breader = new BufferedReader(new FileReader(arg));

		int counter = 1;
		while ((line = breader.readLine()) != null && counter <= 3823) {
			String newLine;

			/* find the position of last occurring comma */
			int lastCommaPosition = line.lastIndexOf(",");
			newLine = line.substring(0, lastCommaPosition);

			int digit = Integer.parseInt(line.substring(lastCommaPosition + 1,
					line.length()));
			if (counter < 1001) { /* development data */
				appendFile(bWritersD, newLine, digit);
			} else { /* training data */
				appendFile(bWritersT, newLine, digit);
			}

			counter++;

		}
		/* closing all the buffered Writers */
		for (int i = 0; i < bWritersT.size(); i++) {
			bWritersT.get(i).close();
			bWritersD.get(i).close();
		}
		breader.close();
	}
}
