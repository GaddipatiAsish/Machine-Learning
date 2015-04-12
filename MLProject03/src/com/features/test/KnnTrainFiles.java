package com.features.test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.core.matrix.Matrix;

import com.features.io.IOOperations;
import com.features.threads.ThreadRun;

public class KnnTrainFiles {
	public static void main(String args[]) throws Exception {
		int NValue[] = { 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700,
				800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
				19000, 20000 };

		boolean normalize = false; /* Normalization Parameter */
		String trainDataFile = args[0];/* Training File */
		String trainLabelsFile = args[1];/* Training Labels File */

		/* Get the training data matrix */
		IOOperations ioTrainFeatures = new IOOperations();
		Matrix trainData = ioTrainFeatures.readData(trainDataFile, 300, 20000);

		/* Get the training data labels */
		IOOperations ioTrainLabels = new IOOperations();
		Matrix trainlabels = ioTrainLabels.readLabels(trainLabelsFile, 300);

		/************************ Pearson Files Generation for KNN **************************/
		/* Read the Pearson ranks from the file */
		IOOperations iop = new IOOperations();
		List<Integer> pRankedFeatureid = iop.readRanksFromFile("Pearson");

		/* Generate the KNN Training Files using Multi Threading */

		int batchCount = 7;
		for (int b = 0; b < batchCount; b++) {
			int threadsPerBatch = 5;
			if (b == 6)
				threadsPerBatch = 4;

			final ExecutorService service = Executors
					.newFixedThreadPool(threadsPerBatch);
			List<Future<int[]>> threadList = new ArrayList<Future<int[]>>();

			for (int n = 0; n < threadsPerBatch; n++) {
				threadList.add(service.submit(new ThreadRun(NValue[b * 5 + n],
						trainData, trainlabels, pRankedFeatureid, "Pearson",
						normalize, 't')));
			}
			/* Wait for the batch to complete */
			int[] returnValues = { 0, 0 };
			try {
				for (int n = 0; n < threadsPerBatch; n++) {
					returnValues[0] += threadList.get(n).get()[0];
					returnValues[1] += threadList.get(n).get()[1];
				}
			} catch (final InterruptedException ex) {
				ex.printStackTrace();
			} catch (final ExecutionException ex) {
				ex.printStackTrace();
			}
			/* shut down the batch */
			service.shutdown();
		}

		/************************ S2Noise Files Generation for KNN **************************/
		/* Read the S2Noise ranks from the file */
		IOOperations ios2n = new IOOperations();
		List<Integer> s2nRankedFeatureid = ios2n.readRanksFromFile("S2Noise");

		/* Generate the KNN Training Files using Multi Threading */

		for (int b = 0; b < batchCount; b++) {
			int threadsPerBatch = 5;
			if (b == 6)
				threadsPerBatch = 4;

			final ExecutorService service = Executors
					.newFixedThreadPool(threadsPerBatch);

			List<Future<int[]>> threadList = new ArrayList<Future<int[]>>();
			for (int n = 0; n < threadsPerBatch; n++) {
				threadList.add(service.submit(new ThreadRun(NValue[b * 5 + n],
						trainData, trainlabels, s2nRankedFeatureid, "S2Noise",
						normalize, 't')));
			}
			/* Wait for the batch to complete */
			int[] returnValues = { 0, 0 };
			try {
				for (int n = 0; n < threadsPerBatch; n++) {
					returnValues[0] += threadList.get(n).get()[0];
					returnValues[1] += threadList.get(n).get()[1];
				}
			} catch (final InterruptedException ex) {
				ex.printStackTrace();
			} catch (final ExecutionException ex) {
				ex.printStackTrace();
			}
			/* shut down the batch */
			service.shutdown();
		}

		/************************ T Test Files Generation for KNN **************************/
		/* Read the Pearson ranks from the file */
		IOOperations iott = new IOOperations();
		List<Integer> ttRankedFeatureid = iott.readRanksFromFile("TTest");

		/* Generate the KNN Training Files using Multi Threading */

		for (int b = 0; b < batchCount; b++) {
			int threadsPerBatch = 5;
			if (b == 6)
				threadsPerBatch = 4;

			final ExecutorService service = Executors
					.newFixedThreadPool(threadsPerBatch);

			List<Future<int[]>> threadList = new ArrayList<Future<int[]>>();
			for (int n = 0; n < threadsPerBatch; n++) {
				threadList.add(service.submit(new ThreadRun(NValue[b * 5 + n],
						trainData, trainlabels, ttRankedFeatureid, "TTest",
						normalize, 't')));
			}
			/* Wait for the batch to complete */
			int[] returnValues = { 0, 0 };
			try {
				for (int n = 0; n < threadsPerBatch; n++) {
					returnValues[0] += threadList.get(n).get()[0];
					returnValues[1] += threadList.get(n).get()[1];
				}
			} catch (final InterruptedException ex) {
				ex.printStackTrace();
			} catch (final ExecutionException ex) {
				ex.printStackTrace();
			}
			/* shut down the batch */
			service.shutdown();
		}
	}
}
