package com.features.test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.core.matrix.Matrix;

import com.features.filterMethods.PearsonCorrelationCoef;
import com.features.filterMethods.S2NRatio;
import com.features.filterMethods.TTest;
import com.features.io.GenerateInputs;
import com.features.io.IOOperations;
import com.features.threads.ThreadRun;

public class SVMTrainFiles {
	/* Ranking the features */

	public static void main(String args[]) throws Exception {
		int NValue[] = { 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700,
				800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
				19000, 20000 };

		boolean normalize = true; /* Normalization parameter */
		String trainDataFile = args[0];/* Training File */
		String trainLabelsFile = args[1];/* Training Labels File */

		/* Get the training data matrix */
		IOOperations ioTrainFeatures = new IOOperations();
		Matrix trainData = ioTrainFeatures.readData(trainDataFile, 300, 20000);

		/* Get the training data labels */
		IOOperations ioTrainLabels = new IOOperations();
		Matrix trainlabels = ioTrainLabels.readLabels(trainLabelsFile, 300);

		/**************** computing Pearson Correlation Coefficient ***********************/
		PearsonCorrelationCoef pearson = new PearsonCorrelationCoef();
		List<Integer> pRankedFeatureid = pearson
				.compute(trainData, trainlabels);
		System.out.println("Prinitng Pearson Ranked Features");
		System.out.println(pRankedFeatureid);

		/* Write the ranks to a file */
		IOOperations iop = new IOOperations();
		iop.writeRanksToFile(pRankedFeatureid, "Pearson");

		/* Generate the SVM Inputs using Pearson Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			GenerateInputs svm = new GenerateInputs();
			Matrix newFeatures = svm.generateFeatures(trainData, trainlabels,
					NValue[j], pRankedFeatureid, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, trainlabels, "Pearson", normalize,
					't');
		}

		/********** compute S2N Ratio, rank the features and generate svm input files **************/
		S2NRatio s2nRatio = new S2NRatio();
		List<Integer> s2nRankedFeatureid = s2nRatio.compute(trainData,
				trainlabels);

		System.out.println("Printing Ranked S2Noise Features");
		System.out.println(s2nRankedFeatureid);

		/* Write the ranks to a file */
		IOOperations ios2n = new IOOperations();
		ios2n.writeRanksToFile(s2nRankedFeatureid, "S2Noise");

		/* Generate the SVM Inputs using S2Noise Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			System.out.println("Working on N = " + NValue[j]);
			GenerateInputs svm = new GenerateInputs();
			Matrix newFeatures = svm.generateFeatures(trainData, trainlabels,
					NValue[j], s2nRankedFeatureid, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, trainlabels, "S2Noise", normalize,
					't');
		}

		/********** Compute TTest ,rank the features and generate the svm input files *************/
		TTest ttest = new TTest();
		List<Integer> ttRankedFeatureid = ttest.compute(trainData, trainlabels);

		System.out.println("Printing T Test Ranked Features");
		System.out.println(ttRankedFeatureid);

		/* Write the ranks to a file */
		IOOperations iott = new IOOperations();
		iott.writeRanksToFile(ttRankedFeatureid, "TTest");

		/* Generate the SVM Inputs using T Test Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			System.out.println("Working on N = " + NValue[j]);
			GenerateInputs svm = new GenerateInputs();
			Matrix newFeatures = svm.generateFeatures(trainData, trainlabels,
					NValue[j], ttRankedFeatureid, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, trainlabels, "TTest", normalize, 't');
		}

	}
}
