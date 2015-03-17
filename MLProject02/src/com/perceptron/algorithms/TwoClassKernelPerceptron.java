package com.perceptron.algorithms;

import java.util.ArrayList;
import java.util.List;

import weka.core.matrix.Matrix;

public class TwoClassKernelPerceptron {

	/**
	 * @description computeAlfaMatrix computes the alfa matrix and return it.
	 * @param noOfEpochs
	 * @param features
	 * @param T
	 * @return alfa matrix
	 */
	public Matrix computeAlfaMatrix(Matrix W, int noOfEpochs,
			List<Matrix> features, List<Integer> trueLabels, char kernelType,
			int exponent, double sigma) {

		Matrix alfa; /* alfa matrix NX1 */
		int epochCounter = 0; /* counts the no of epochs */
		int dataSetSize = 0; /* size of the input data set */
		int dataPointsCounter; /* used for convergence check. */
		boolean isConverged = false;

		/* check if the input data is invalid */
		if (features.size() != trueLabels.size()) {
			System.exit(-1);
		} else {
			dataSetSize = features.size();
		}
		System.out.println("data set Size " + dataSetSize);
		/* Initial : load alfa matrix with 0 */
		alfa = new Matrix(trueLabels.size(), 1);
		for (int i = 0; i < trueLabels.size(); i++) {
			alfa.set(i, 0, 0);
		}

		/* alfa computation steps */

		do {/* External loop on noOfEpochs */
			epochCounter++;
			dataPointsCounter = 0;
			System.out.println("Epoch : "+ epochCounter);
			for (int i = 0; i < features.size(); i++) {
				System.out.println("**** Data Point " + i + " ****");
				Double yi = 0.0; /* value of the discriminant function */
				/*feature vector of Test Xi*/
				Matrix featuresOfXi = features.get(i);
				Integer trueLabel = trueLabels.get(i);

				/* compute the discriminantFn value */
				yi = discriminantFn(features, trueLabels, featuresOfXi,
						alfa, kernelType, exponent, sigma);


				/* compute system label */
				Integer sysLabel = sgn(yi);
				System.out.println("Discriminant Value : " + yi);
				System.out.println("System label "+ sysLabel);
				System.out.println("true Label "+ trueLabel);

				/* check True Label not equals system label */
				if (sysLabel != trueLabel) {
					/*
					 * increment the ith cell of alfa if featureOfXi contributes
					 * to W
					 */
					alfa.set(i, 0, alfa.get(i, 0) + 1);
				} else {
					dataPointsCounter++;
				}
			}
			System.out.println("data points Counter "+dataPointsCounter);
			if (dataPointsCounter == dataSetSize) {/* convergence check */
				isConverged = true;
			}
			
			System.out.println("IsConverged? "+ isConverged);
		} while (noOfEpochs != epochCounter && !isConverged);

		/* print the results of convergence */
		if (!isConverged) {
			System.out.println("Data set doesnt Converge!");
		} else {
			System.out.println("Data set Converges");
		}

		System.out.println("Final epochCounter " + epochCounter);
		System.out.println("Printing alfa matrix");
		alfa.print(1, 1);
		return alfa;
	}

	/**
	 * Step function that takes yi and classify it to one of the possible two
	 * classes.
	 * 
	 * @param yi
	 * @return system Label
	 */
	private Integer sgn(Double yi) {
		Integer sysLabel = 0;
		if (yi >= 0) {
			sysLabel = +1;
		} else {
			sysLabel = -1;
		}
		return sysLabel;

	}

	/* Different Kernel Matrices Implementation */

	/* Linear Kernel */
	/**
	 * @description computeLinearKij computes a single cell that goes into gram
	 *              matrix
	 * @param Xi
	 *            feature vector of ith Training data
	 * @param X
	 *            feature vector of Test data point
	 * @return kernel value
	 */
	Double computeLinearKij(Matrix Xi, Matrix X) {
		return Xi.transpose().times(X).get(0, 0);
	}

	/**
	 * @description computePolynomialKij computes a single cell that goes into
	 *              gram matrix
	 * @param Xi
	 *            feature vector of ith Training data
	 * @param X
	 *            feature vector of Test data point
	 * @param exponent
	 * @return polynomial Kernel value
	 */
	Double computePolynomialKij(Matrix Xi, Matrix X, int exponent) {
		return Math.pow((1 + computeLinearKij(Xi, X)), exponent);
	}

	/**
	 * @description computeGausianKij computes a single cell that goes into gram
	 *              matrix
	 * @param Xi
	 *            feature vector of ith Training data
	 * @param X
	 *            feature vector of Test data point
	 * @param sigma
	 * @return Gausian Kernel value
	 */
	Double computeGausianKij(Matrix Xi, Matrix X, double sigma) {
		return Math.exp(-(Xi.minus(X).norm2()) / (2 * sigma * sigma));
	}

	/**
	 * @description classify method is a 2 class classifier that takes a test
	 *              data point and compares it with entire training data set to
	 *              compute yi which is sent to sgn() method to classify into
	 *              (-1 or +1 class).
	 * @param featuresTrain
	 *            features of all points of Training data set
	 * @param featureOfXiTest
	 *            features of all points of Test data set
	 * @param alfa
	 *            learned matrix
	 * @param kernelType
	 *            which kernel to be used : polynomial,Gausian,Linear etc.
	 * @param exponent
	 *            used by the polynomial kernel
	 * @param sigma
	 *            used by the Gausian Kernel
	 */
	public void classify(List<Matrix> featuresTrain,
			List<Integer> trueLabelTrain, List<Matrix> featuresTest,
			Matrix alfa, char kernelType, int exponent, double sigma) {

		/* complete list of sysLabels for the whole test data set */
		List<Integer> syslabelsTest = new ArrayList<Integer>();

		/* loop for test Xi's */
		for (int i = 0; i < featuresTest.size(); i++) {

			Double yi = 0.0; /* value of the discriminant function */
			/*feature vector of Test Xi*/
			Matrix featuresOfXiTest = featuresTest.get(i);

			/* compute the discriminantFn value */
			yi = discriminantFn(featuresTrain, trueLabelTrain, featuresOfXiTest,
					alfa, kernelType, exponent, sigma);

			/* compute system label for Test Xi */
			syslabelsTest.add(sgn(yi));
		}
	}

	/**
	 * @description discriminantFn method return the value of the discriminant
	 *              function which is then passed to sgn() to find the class{-1
	 *              or +1}. This can be used for testing(to classify the given
	 *              data point Xi) as well as training (to find alfa).
	 * @param featuresTrain
	 * @param trueLabelTrain
	 * @param featureOfXiTest
	 * @param alfa
	 * @param kernelType
	 * @param exponent
	 * @param sigma
	 * @return
	 */

	double discriminantFn(List<Matrix> featuresTrain,
			List<Integer> trueLabelTrain, Matrix featureOfXiTest, Matrix alfa,
			char kernelType, int exponent, double sigma) {

		Double yi = 0.0;
		/* loop for train Xj's */
		for (int j = 0; j < featuresTrain.size(); j++) {
			
			double ai = alfa.get(j, 0); /* get alfa ai of training Xi */
			double ti = trueLabelTrain.get(j); /* true label for Xi */

			if (ai != 0) { /* compute yi */
				double k = 0.0;
				/* select the required kernel */
				switch (kernelType) {
				case 'a': /* Linear Kernel */
					k = computeLinearKij(featuresTrain.get(j), featureOfXiTest);
					break;
				case 'b': /* polynomial Kernel */
					if (exponent > 1) {
						k = computePolynomialKij(featuresTrain.get(j),
								featureOfXiTest, exponent);
					}
					break;
				case 'c': /* Gausian Kernel */
					k = computeGausianKij(featuresTrain.get(j),
							featureOfXiTest, sigma);
					break;
				}
				yi += ai * ti * k;
			}
		}

		return yi;
	}

}
