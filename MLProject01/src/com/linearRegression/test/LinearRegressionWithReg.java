package com.linearRegression.test;

import java.util.ArrayList;
import java.util.List;

import weka.core.matrix.Matrix;

import com.linearRegression.core.ERMS;
import com.linearRegression.core.OptimalLambda;
import com.linearRegression.core.WeightVector;
import com.linearRegression.io.IOOperations;

/**
 * LinearRegressionWithReg
 * 
 * @author AsishKumar
 *
 */
public class LinearRegressionWithReg {
	/**
	 * @description : question5_C_3_A method computes ERMS value for Train on
	 *              the training data, test on the test data.
	 * @return ERMS Value
	 * @throws Exception
	 */
	public static double question5_C_3_A(String args[], int optimallnLambda)
			throws Exception {

		/* (x,t) data set values for Training Data */
		List<Double> xValuesTrainData = new ArrayList<Double>();
		List<Double> tValuesTrainData = new ArrayList<Double>();
		/* (x,t) data set values for Test Data */
		List<Double> xValuesTestData = new ArrayList<Double>();
		List<Double> tValuesTestData = new ArrayList<Double>();

		/* Read Training Data */
		IOOperations ioTrainData = new IOOperations();
		ioTrainData.readDataFromInput(args[0]);
		xValuesTrainData = ioTrainData.getXValues();
		tValuesTrainData = ioTrainData.getTValues();

		/* Read Test Data */
		IOOperations ioTestData = new IOOperations();
		ioTestData.readDataFromInput(args[2]);
		xValuesTestData = ioTestData.getXValues();
		tValuesTestData = ioTestData.getTValues();

		WeightVector computeW = new WeightVector();
		ERMS erms = new ERMS();

		/* Computing WEIGHT VECTOR for M=9 */
		Matrix W = computeW.getW(9, xValuesTrainData, tValuesTrainData);

		/* return ERMS value for Test Data */
		return erms.compute(W, xValuesTestData, tValuesTestData,
				optimallnLambda);
	}

	/**
	 * @description question5_C_3_B method computes the ERMS value for the case
	 *              Train on Training+Validation Data and test on Test data
	 * @param args
	 * @param optimallnLambda
	 * @return ERMS Value
	 * @throws Exception
	 */
	public static double question5_C_3_B(String args[], int optimallnLambda)
			throws Exception {

		/* (x,t) data set values for Training Data */
		List<Double> xValuesTrainData = new ArrayList<Double>();
		List<Double> tValuesTrainData = new ArrayList<Double>();
		/* (x,t) data set values for Test Data */
		List<Double> xValuesTestData = new ArrayList<Double>();
		List<Double> tValuesTestData = new ArrayList<Double>();

		int testFlag = 1; /* Test file is generally passed as argument 1 */
		if (args.length > 2) { /* account validation data as train data */
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

		WeightVector computeW = new WeightVector();
		ERMS erms = new ERMS();
		/* Computing WEIGHT VECTOR for M=9 */
		Matrix W = computeW.getW(9, xValuesTrainData, tValuesTrainData);

		/* return ERMS Value for Test data */
		return erms.compute(W, xValuesTestData, tValuesTestData,
				optimallnLambda);
	}

	/**
	 * @description Main Method to test Linear regression with Regularisation.
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception {
		System.out.println("***** Linear Regression with Regularisation *****");
		System.out.println("-----------------------------------------------------");
		int optimallnLambda;
		int orderOfPolynomial = 9;

		/* Find OPTIMAL lnLambda Value */
		OptimalLambda optLambda = new OptimalLambda();
		optimallnLambda = optLambda.getOptimalLnLambda(orderOfPolynomial, args);

		/**
		 * FIND ERMS for Question a and b.
		 */
		System.out.println("Optimal lnLambda(Model Selection) : "+optimallnLambda);
		System.out.println("question5_C_3_A: ERMS Value : "
				+ question5_C_3_A(args, optimallnLambda));
		System.out.println("question5_C_3_B: ERMS Value : "
				+ question5_C_3_B(args, optimallnLambda));

	}
}
