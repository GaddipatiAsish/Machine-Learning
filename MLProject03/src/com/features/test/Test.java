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

public class Test {
	/* Ranking the features */

	public static void main(String args[]) throws Exception {
		int NValue[] = { 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700,
				800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
				19000, 20000 };

		boolean normalize = false;
		String trainDataFile = args[0];/* Training File */
		String trainLabelsFile = args[1];

		/* Get the training data matrix */
		IOOperations ioTrainFeatures = new IOOperations();
		Matrix trainData = ioTrainFeatures.readData(trainDataFile, 300, 20000);

		/* Get the training data labels */
		IOOperations ioTrainLabels = new IOOperations();
		Matrix trainlabels = ioTrainLabels.readLabels(trainLabelsFile, 300);
		
		System.out.println("Pearson");
		/* computing Pearson Correlation Coefficient */
		PearsonCorrelationCoef pearson = new PearsonCorrelationCoef();
		Map pearsonMap = pearson.compute(trainData, trainlabels);

		/* Generate the SVM Inputs using Pearson Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			GenerateInputs svmInputs = new GenerateInputs();
			Matrix newFeatures = svmInputs.generateSVMInputs(trainData,
					trainlabels, NValue[j], pearsonMap, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, trainlabels, "Pearson");
		}
		
//		String methodName = "Pearson";
//		for (int p = 0; p < NValue.length; p += 5) {
//			int batchSize = 5;
//			if (p == 30) {
//				batchSize = 4;
//			}
//
//			final ExecutorService service;
//			List<Future<Double>> task = new ArrayList<Future<Double>>();
//			service = Executors.newFixedThreadPool(batchSize);
//
//			for (int i = p; i < batchSize; i++) {
//
//				GenerateInputs svmInputs = new GenerateInputs();
//				Matrix newFeatures = svmInputs.generateSVMInputs(trainData,
//						trainlabels, NValue[i], pearsonMap, normalize);
//				
//				IOOperations io = new IOOperations();
//				io.writeToFileSVM(newFeatures, trainlabels, "Pearson");
//				task.add(service.submit(new ThreadRun(NValue[i], newFeatures,
//						trainlabels, methodName)));
//			}
//			/* shut down the service */
//			service.shutdownNow();
//		}

		/* compute S2N Ratio */
		S2NRatio s2nRatio = new S2NRatio();
		Map s2nMap = s2nRatio.compute(trainData, trainlabels);
		System.out.println("S2Noise");
		/* Generate the SVM Inputs using S2Noise Correlation Coef */
		for (int j = 0; j < NValue.length; j++) {
			GenerateInputs svmInputs = new GenerateInputs();
			Matrix newFeatures = svmInputs.generateSVMInputs(trainData, trainlabels,
					NValue[j], s2nMap, normalize);
			IOOperations io = new IOOperations();
			io.writeToFileSVM(newFeatures, trainlabels, "S2Noise");
			// io.writeToFileKNN(newFeatures, trainlabels, "S2Noise");
		}
		// String methodName = "Pearson";
		// for (int j = 0; j < NValue.length; j += 5) {
		// int batchSize = 5;
		// if (j == 30) {
		// batchSize = 4;
		// }
		//
		// final ExecutorService service;
		// List<Future<Double>> task = new ArrayList<Future<Double>>();
		// service = Executors.newFixedThreadPool(batchSize);
		//
		// for (int i = j; i < batchSize; i++) {
		// task.add(service.submit(new ThreadRun(NValue[i], newFeatures,
		// trainlabels, methodName)));
		// }
		//
		// /* shut down the service */
		// service.shutdownNow();
		//
		// }
		//
		//
		//
		//
		//
		// /* Compute TTest and ranking the features */
		 TTest ttest = new TTest();
		 Map ttestMap = ttest.compute(trainData, trainlabels);
		 System.out.println("T Test");
		 /* Generate the SVM Inputs using T Test Correlation Coef */
		 for (int j = 0; j < NValue.length; j++) {
		 GenerateInputs svmInputs = new GenerateInputs();
		 Matrix newFeatures = svmInputs.generateSVMInputs(trainData,
		 trainlabels, NValue[j], ttestMap, normalize);
		 IOOperations io = new IOOperations();
		 io.writeToFileSVM(newFeatures, trainlabels, "TTest");
		 //io.writeToFileKNN(newFeatures, trainlabels, "TTest");
		
		 }

	}
}
