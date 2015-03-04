package com.linearRegression.test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.core.matrix.Matrix;

import com.linearRegression.core.ERMS;
import com.linearRegression.core.WeightVector;
import com.linearRegression.io.IOOperations;

/**
 * LinearRegressionWithoutReg is the test class that is used to perform Linear
 * regression with OUT Regularization for the test data sets given.
 * 
 * @author AsishKumar
 *
 */
public class LinearRegressionWithoutReg {
	/**
	 * @description Main Method to test Linear regression with OUT Regularisation
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception {
		System.out.println("***** Linear Regression with OUT Regularisation *****");
		System.out.println("-----------------------------------------------------");
		int orderOfPolynomial = 9;
		/**
		 * Hash Map that stores the ERMS values of corresponding M's for
		 * training or training+validation data. Combines the Training Data with
		 * Validation data for Calculations when validation data is sent along
		 * with training data.
		 */
		Map<Integer, Double> eRMSvsM_trainValidData_Map = new HashMap<Integer, Double>();
		/**
		 * Hash Map that stores ERMS values of corresponding M's for test data.
		 */
		Map<Integer, Double> eRMSvsM_testData_Map = new HashMap<Integer, Double>();

		/**
		 * *******READING THE DATA FILES*****
		 */
		/* (x,t) data set values for Training Data */
		List<Double> xValuesTrainData = new ArrayList<Double>();
		List<Double> tValuesTrainData = new ArrayList<Double>();
		/* (x,t) data set values for Test Data */
		List<Double> xValuesTestData = new ArrayList<Double>();
		List<Double> tValuesTestData = new ArrayList<Double>();
		int testFlag = 1; /* Test file is generally passed as argument 1 */
		if (args.length > 2) { /* IF Validation Data is received */
			/* Read Training+Validation Data */
			testFlag = 2;
			IOOperations ioTrainValidData = new IOOperations();
			for (int i = 0; i < (args.length - 1); i++) {
				ioTrainValidData.readDataFromInput(args[i]);
			}
			xValuesTrainData = ioTrainValidData.getXValues();
			tValuesTrainData = ioTrainValidData.getTValues();
		} else if (args.length == 2) { /* Training data */
			/* Read Training Data */
			IOOperations ioTrainData = new IOOperations();
			ioTrainData.readDataFromInput(args[0]);
			xValuesTrainData = ioTrainData.getXValues();
			tValuesTrainData = ioTrainData.getTValues();
		}
		/* Read Test Data */
		IOOperations ioTestData = new IOOperations();
		ioTestData.readDataFromInput(args[testFlag]);
		xValuesTestData = ioTestData.getXValues();
		tValuesTestData = ioTestData.getTValues();

		/**
		 * FIND ERMS VS M VALUES TO PLOT GRAPH
		 */
		for (int i = 0; i <= orderOfPolynomial; i++) {
			WeightVector computeW = new WeightVector();
			ERMS erms = new ERMS();
			/* Computing WEIGHT VECTOR for M=0..9 */
			Matrix W = computeW.getW(i, xValuesTrainData, tValuesTrainData);
			/* ERMS vs M map for Training Data */
			eRMSvsM_trainValidData_Map.put(i,
					erms.compute(W, xValuesTrainData, tValuesTrainData));
			/* ERMS vs M map for Test Data */
			eRMSvsM_testData_Map.put(i,
					erms.compute(W, xValuesTestData, tValuesTestData));
		}

		/* generates R file */
		drawGraph(eRMSvsM_trainValidData_Map, eRMSvsM_testData_Map);

	}

	/**
	 * Description: drawGraph method generates the .R file that's when opened
	 * with R program generates the graph.
	 * 
	 * @param trainDataeRMSMap
	 *            : holds the ERMS values for the train Data
	 * @param testDataeRMSMap
	 *            : holds the ERMS values for the Test Data
	 * @throws IOException
	 */
	public static void drawGraph(Map<Integer, Double> trainDataeRMSMap,
			Map<Integer, Double> testDataeRMSMap) throws IOException {

		String trainData = "";
		String testData = "";
		File file = new File("./OutputData/Erms_vs_M_WOReg.r");
		BufferedWriter bwriter = new BufferedWriter(new FileWriter(file));

		Set<Integer> keysTrain = trainDataeRMSMap.keySet();
		Set<Integer> keysTest = testDataeRMSMap.keySet();

		Iterator<Integer> iteratorTrain = keysTrain.iterator();
		Iterator<Integer> iteratorTest = keysTest.iterator();

		while (iteratorTrain.hasNext() && iteratorTest.hasNext()) {
			trainData += trainDataeRMSMap.get(iteratorTrain.next()) + ",";
			testData += testDataeRMSMap.get(iteratorTest.next()) + ",";
		}

		bwriter.write("## R File to plot Graph for ERMS VS M values for Train and Test Data for With out Regularization ## \n");
		bwriter.write("## Author: Asish Kumar Gaddipati, ag615513@ohio.edu, Ohio University, Athens.                    ## \n");
		bwriter.write("mvalue <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)\n");
		bwriter.write("train <-c("
				+ trainData.substring(0, trainData.length() - 1) + ")\n");
		bwriter.write("test <-c("
				+ testData.substring(0, testData.length() - 1) + ")\n");
		bwriter.write("plot(mvalue, test, type='o', pch=10, col=\"green\", xlab=\"M\", ylab=\"Erms\", ylim=c(0, 0.25), xlim=c(0,10))\n");
		bwriter.write("lines(mvalue, train, type='o', pch=20, col=\"purple\",lty=3)\n");
		bwriter.write("legend(\"topleft\", c(\"Test Data\",\"Train Data\"), cex=1.0, col=c(\"green\",\"purple\"), lty=1:2, lwd=2, bty=\"n\")\n");
		bwriter.close();
		System.out.println("Erms_vs_M_WOReg.r File Generated.");
	}

}
