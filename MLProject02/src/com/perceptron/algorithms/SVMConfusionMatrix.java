package com.perceptron.algorithms;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.core.matrix.Matrix;
/**
 * SVMConfusionMatrix class generates the confusion matrices for KErnels of SVMs
 * @author AsishKumar
 *
 */
public class SVMConfusionMatrix {
	List<List<Double>> allResults = new ArrayList<List<Double>>();
	List<Integer> testTrueLabels = new ArrayList<Integer>();

	void readTestDataFile(String testDataFile) throws NumberFormatException,
			IOException {
		BufferedReader breaderTestFile = new BufferedReader(new FileReader(
				testDataFile));
		String newLineTestFile;
		while ((newLineTestFile = breaderTestFile.readLine()) != null) {

			int index = newLineTestFile.lastIndexOf(",");

			testTrueLabels.add(Integer.parseInt(newLineTestFile
					.substring(index + 1)));
		}
		breaderTestFile.close();

	}

	void readResultFiles(String resultFile, char kernelType) throws IOException {

		for (int digit = 0; digit < 10; digit++) {

			String rFile = resultFile;
			switch (kernelType) {
			case 'a':
				rFile += digit + "_LinearKernel_epochs5.result";
				break;
			case 'b':
				rFile += digit + "_polyK_deg5_epochs5.result";
				break;

			case 'c':
				rFile += digit + "_GauK_sigma10_epochs5.result";
				break;
			}

			List<Double> result = new ArrayList<Double>();
			BufferedReader breaderResultFile = new BufferedReader(
					new FileReader(rFile));
			String newLineResultFile;
			while ((newLineResultFile = breaderResultFile.readLine()) != null) {
				result.add(Double.parseDouble(newLineResultFile));
			}
			breaderResultFile.close();
			allResults.add(result);
		}
	}

	public Matrix generateConfusionMatrix(char KernelType) throws IOException {
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}

		String resultFile = "./svm_light_OS10.8.4_i7/svmTestResults/";
		readResultFiles(resultFile, KernelType);

		String testFile = "./inputData/optDigits.tes";
		readTestDataFile(testFile);
		/* loop over the 10 digits */
		for (int i = 0; i < testTrueLabels.size(); i++) {
			List<Double> yiList = new ArrayList<Double>();
			int tLabel = testTrueLabels.get(i);
			for (int j = 0; j < 10; j++) {
				yiList.add(allResults.get(j).get(i));
			}
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);

		}
		conMatrix.print(6, 2);
		return conMatrix;

	}
}
