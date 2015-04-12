package com.features.test;

import java.util.List;

import weka.core.matrix.Matrix;

import com.features.io.GenerateInputs;
import com.features.io.IOOperations;

public class SVMTestFiles {
	public static void main(String args[]) throws Exception {
		int NValue[] = { 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700,
				800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
				19000, 20000 };

		boolean normalize = true; /* Normalization parameter */
		String testDataFile = args[0];/* Test File */
		String testLabelsFile = args[1];/* Test Labels File */

		/* Get the training data matrix */
		IOOperations ioTrainFeatures = new IOOperations();
		Matrix testData = ioTrainFeatures.readData(testDataFile, 300, 20000);

		/* Get the training data labels */
		IOOperations ioTrainLabels = new IOOperations();
		Matrix testlabels = ioTrainLabels.readLabels(testLabelsFile, 300);

		/******************* Pearson Test File Generation **************************/
		/* Read the Pearson ranks from the file */
		IOOperations iop = new IOOperations();
		List<Integer> pRankedFeatureid = iop.readRanksFromFile("Pearson");

		/* Generate the SVM Test files using Pearson Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			System.out.println("Working on N = " + NValue[j]);
			GenerateInputs svm = new GenerateInputs();
			Matrix newFeatures = svm.generateSVMFeatures(testData, testlabels,
					NValue[j], pRankedFeatureid, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, testlabels, "Pearson", normalize, 'v');
		}

		/******************* S2Noise Test File Generation *************************/
		/* Read the S2Noise ranks from the file */
		IOOperations ios2n = new IOOperations();
		List<Integer> s2nRankedFeatureid = ios2n.readRanksFromFile("S2Noise");

		/* Generate the SVM Inputs using S2Noise Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			System.out.println("Working on N = " + NValue[j]);
			GenerateInputs svm = new GenerateInputs();
			Matrix newFeatures = svm.generateSVMFeatures(testData, testlabels,
					NValue[j], s2nRankedFeatureid, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, testlabels, "S2Noise", normalize,'v');
		}
		
		/****************** T Test Test File Generation **************************/
		/* Read the T Test ranks from the file */
		IOOperations iott = new IOOperations();
		List<Integer> ttRankedFeatureid = iott.readRanksFromFile("TTest");
		
		 /*Generate the SVM Inputs using T Test Correlation Coef*/
		 for (int j = 0; j < NValue.length; j++) {
			 System.out.println("Working on N = "+NValue[j]);
			 GenerateInputs svm = new GenerateInputs();
			 Matrix newFeatures = svm.generateSVMFeatures(testData,
					 testlabels, NValue[j], ttRankedFeatureid, normalize);
			 IOOperations io = new IOOperations();
			 io.writeToFileSVM(newFeatures, testlabels, "TTest",normalize,'v');
		 }
	}
}
