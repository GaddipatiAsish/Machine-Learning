package com.linearRegression.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.core.matrix.Matrix;

import com.linearRegression.io.IOOperations;

/**
 * OptimalLambda has methods that computes the optimal lnLambda value by
 * evaluating the validation data and Training data. and its also generates an R
 * File that plots the graph of ERMS vs lnLambda for Training data and
 * Validation Data.
 * 
 * @author AsishKumar
 *
 */
public class OptimalLambda {
	/**
	 * Hash Map that stores ERMS values of corresponding lnLambda's for Training
	 * data.
	 */
	Map<Integer, Double> eRMSvslnLambda_trainData_Map = new HashMap<Integer, Double>();
	/**
	 * Hash Map that stores ERMS values of corresponding lnLambda's for
	 * validation data.
	 */
	Map<Integer, Double> eRMSvslnLambda_ValidData_Map = new HashMap<Integer, Double>();

	/**
	 * @description getOptimalLnLambda methods return the optimal value of
	 *              lnLambda for Regularizer
	 * @param orderOfPolynomial
	 * @return lnLambda
	 * @throws Exception
	 */
	public int getOptimalLnLambda(int orderOfPolynomial, String args[])
			throws Exception {
		int lnLambda = 0;
		/* (x,t) data set values for Training Data */
		List<Double> xValuesTrainData = new ArrayList<Double>();
		List<Double> tValuesTrainData = new ArrayList<Double>();
		/* (x,t) data set values for Valid Data */
		List<Double> xValuesValidData = new ArrayList<Double>();
		List<Double> tValuesValidData = new ArrayList<Double>();

		/* Read Training Data */
		IOOperations ioTrainData = new IOOperations();
		ioTrainData.readDataFromInput(args[0]);
		xValuesTrainData = ioTrainData.getXValues();
		tValuesTrainData = ioTrainData.getTValues();

		/* Read Validation Data */
		IOOperations ioValidData = new IOOperations();
		ioValidData.readDataFromInput(args[1]);
		xValuesValidData = ioValidData.getXValues();
		tValuesValidData = ioValidData.getTValues();

		/* Compute W on Training Data for every lnLambda */
		for (lnLambda = -50; lnLambda <= 0; lnLambda += 5) {
			WeightVector WObject = new WeightVector();
			Matrix W = WObject.getW(9, xValuesTrainData, tValuesTrainData,
					lnLambda);

			ERMS erms = new ERMS();
			/* for Training Data */
			eRMSvslnLambda_trainData_Map.put(lnLambda, erms.compute(W,
					xValuesTrainData, tValuesTrainData, lnLambda));
			/* for Validation Data */
			eRMSvslnLambda_ValidData_Map.put(lnLambda, erms.compute(W,
					xValuesValidData, tValuesValidData, lnLambda));
		}
		/*generate the R File to see the Graph*/
		drawGraph(eRMSvslnLambda_trainData_Map, eRMSvslnLambda_ValidData_Map);

		/*Find the optimal lnLambda Value*/
		double minValue = Collections
				.min(eRMSvslnLambda_ValidData_Map.values());
		Set<Integer> keys = eRMSvslnLambda_ValidData_Map.keySet();
		Iterator<Integer> iterator = keys.iterator();
		while (iterator.hasNext()) {
			if (minValue == eRMSvslnLambda_ValidData_Map
					.get(lnLambda = iterator.next())) {
				break;
			}
		}
		
		/*return the optimal lnLambda Value*/
		return lnLambda;
	}

	/**
	 * Description: drawGraph method generates the .R file that's when opened
	 * with R program generates the graph
	 * 
	 * @param trainDataeRMSMap
	 *            : holds the ERMS values for the Training Data
	 * @param validDataeRMSMap
	 *            : holds the ERMS values for the validation Data
	 * @throws IOException
	 */
	public static void drawGraph(Map<Integer, Double> trainDataeRMSMap,
			Map<Integer, Double> validDataeRMSMap) throws IOException {

		String trainData = "";
		String validData = "";
		String lnLambda = "";

		File file = new File(
				"./OutputData/Erms_vs_lnLambda_Model_Selction_WReg.r");
		BufferedWriter bwriter = new BufferedWriter(new FileWriter(file));

		for (int i = -50; i <= 0; i += 5) {
			lnLambda += i + ",";
			trainData += trainDataeRMSMap.get(i) + ",";
			validData += validDataeRMSMap.get(i) + ",";
		}

		bwriter.write("## R File to plot Graph for ERMS VS lnLambda values for Train and Valid Data for Linear Regression With Regularization ## \n");
		bwriter.write("## Author: Asish Kumar Gaddipati, ag615513@ohio.edu, Ohio University, Athens.                ## \n");
		bwriter.write("lnlambdavalue <- c("
				+ lnLambda.substring(0, lnLambda.length() - 1) + ")\n");
		bwriter.write("train <-c("
				+ trainData.substring(0, trainData.length() - 1) + ")\n");
		bwriter.write("valid <-c("
				+ validData.substring(0, validData.length() - 1) + ")\n");
		bwriter.write("plot(lnlambdavalue, valid, type='o', pch=10, col=\"green\", xlab=\"ln Lambda\", ylab=\"ERMS\", ylim=c(0, 1), xlim=c(-50,0))\n");
		bwriter.write("lines(lnlambdavalue, train, type='o', pch=20, col=\"purple\",lty=3)\n");
		bwriter.write("legend(\"topleft\", c(\"Valid Data\",\"Train Data\"), cex=1.0, col=c(\"green\",\"purple\"), lty=1:2, lwd=2, bty=\"n\")\n");
		bwriter.close();
		System.out.println("Erms_vs_lnLambda_Model_Selction_WReg.r file Generated.");
	}

}
