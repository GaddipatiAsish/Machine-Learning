package com.perceptron.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import weka.core.matrix.Matrix;

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
	List<Integer> trueLabels = new ArrayList<Integer>();

	/**
	 * @description readInputData method reads the input data from the file
	 *              thats passed as argument and fills the featureVector and
	 *              featureOfXi vectors.
	 * 
	 * @param arg
	 */
	public void readInputData(String arg, boolean lPercepFlag)
			throws IOException {
		
		String row;
		BufferedReader breader = new BufferedReader(new FileReader(
				new File(arg)));
		while ((row = breader.readLine()) != null) {

			List<Double> featureOfXi = new ArrayList<Double>();
			if(lPercepFlag){/*add 1 to features for linear perceptron*/
				featureOfXi.add(1.0);
			}
			/* find the position of last occurring comma */
			int lastCommaPosition = row.lastIndexOf(",");
			/* adding the corresponding class Ci that the Xi belongs */
			trueLabels.add(Integer.parseInt(row.substring(
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

	/**
	 * @description getFeatureMatrix returns the training data features as a
	 *              List<Matrix>
	 * 
	 * @return
	 */
	public List<Matrix> getFeatureMatrix() {
		List<Matrix> featureMatrix = new ArrayList<Matrix>();
		Iterator<List<Double>> iterator = featureVector.iterator();
		while (iterator.hasNext()) {
			List<Double> temp = iterator.next();
			// System.out.println("temp.size() : "+temp.size());
			Matrix featuresOfXi = new Matrix(temp.size(), 1);
			for (int i = 0; i < temp.size(); i++) {
				featuresOfXi.set(i, 0, temp.get(i));
			}
			featureMatrix.add(featuresOfXi);
		}
		return featureMatrix;
	}

	/**
	 * @description getTargetVector method returns the corresponding true labels
	 *              of the training data set
	 * @return
	 */
	public List<Integer> getTrueLabels() {
		return trueLabels;
	}

	public Set<Integer> getKClasses() {
		return new HashSet<Integer>(trueLabels);
	}

	public void appendFile(List<BufferedWriter> bWriters, String newLine,
			int digit) throws IOException {
		// System.out.println("called");
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

	/**
	 * @description generateClassificationFiles generates a file for each digit
	 *              marking the true label as +1 if it true(Example: 0.tra
	 *              contains true labels as +1 if it feature vector belongs to 0
	 *              else -1) else it marks -1
	 * @param arg
	 *            dataFile that contains complete data for 10 digits
	 * @throws IOException
	 */
	public void generateClassificationFiles(String arg) throws IOException {
		List<File> trainingFiles = new ArrayList<File>();
		List<File> developmentFiles = new ArrayList<File>();
		List<File> testFiles = new ArrayList<File>();

		for (int i = 0; i <= 9; i++) {
			trainingFiles.add(new File("./inputData/" + i + ".tra"));
			developmentFiles.add(new File("./inputData/" + i + ".dev"));
			testFiles.add(new File("./inputData/" + i + ".tes"));
		}
		List<BufferedWriter> bWritersT = new ArrayList<BufferedWriter>();
		List<BufferedWriter> bWritersD = new ArrayList<BufferedWriter>();
		List<BufferedWriter> bWritersTe = new ArrayList<BufferedWriter>();

		for (int i = 0; i <= 9; i++) {
			bWritersT.add(new BufferedWriter(new FileWriter(trainingFiles
					.get(i))));
			bWritersD.add(new BufferedWriter(new FileWriter(developmentFiles
					.get(i))));
			bWritersTe
					.add(new BufferedWriter(new FileWriter(testFiles.get(i))));
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

	public void generateClassificationFilesTest(String arg) throws IOException {

		List<File> testFiles = new ArrayList<File>();

		for (int i = 0; i <= 9; i++) {
			testFiles.add(new File("./inputData/" + i + ".tes"));
		}

		List<BufferedWriter> bWritersTe = new ArrayList<BufferedWriter>();

		for (int i = 0; i <= 9; i++) {

			bWritersTe
					.add(new BufferedWriter(new FileWriter(testFiles.get(i))));
		}

		String line;
		BufferedReader breader = new BufferedReader(new FileReader(arg));
		// System.out.println(testFiles);
		// System.out.println(bWritersTe);
		int counter = 1;
		while ((line = breader.readLine()) != null) {
			String newLine;
			// System.out.println("Looped");
			/* find the position of last occurring comma */
			int lastCommaPosition = line.lastIndexOf(",");
			newLine = line.substring(0, lastCommaPosition);

			int digit = Integer.parseInt(line.substring(lastCommaPosition + 1,
					line.length()));

			appendFile(bWritersTe, newLine, digit);

			counter++;

		}
		/* closing all the buffered Writers */
		for (int i = 0; i < bWritersTe.size(); i++) {
			bWritersTe.get(i).close();
		}
		breader.close();
	}

	public void generateSVMDataFile(List<Matrix> features,
			List<Integer> trueLabels, String fileName) throws IOException {
		BufferedWriter bwriter = new BufferedWriter(new FileWriter(new File(
				"./svmipData/" + fileName)));

		for (int i = 0; i < features.size(); i++) {
			String newLine = trueLabels.get(i).toString() + " ";
			Matrix featuresOfXi = features.get(i);
			for (int k = 0; k < featuresOfXi.getRowDimension(); k++) {
				double fval = featuresOfXi.get(k, 0);
				if (fval > 0) {
					newLine += (k + 1) + ":" + fval + " ";
				}
			}
			bwriter.write(newLine);
			bwriter.newLine();
		}
		bwriter.close();
	}
}
