package com.perceptron.threads;

import java.util.List;
import java.util.concurrent.Callable;

import weka.core.matrix.Matrix;

import com.perceptron.algorithms.KernelPerceptron;
import com.perceptron.io.IOOperations;

public class ExponentThreadRun implements Callable<Double> {
	int exponent;
	List<Matrix> featureMatrixTrainData; /* features Matrix: Training Data */
	List<Integer> trueLabelsTrainData;/* true labels : Training data */
	List<Matrix> featureMatrixDevData; /* features Matrix : Development data */
	List<Integer> trueLabelsDevData;/* true labels : Development data */

	public ExponentThreadRun(int exponent) {
		this.exponent = exponent;
	}

	/**
	 * 
	 */
	public Double call() throws Exception {
		Double accuracy = 0.0;
		int favorable = 0;
		int count = 0;
		
		for (int i = 0; i <= 9; i++) {
			
			//System.out.println("For digit : "+ i);
			/* Get the Training and development data */
			IOOperations ioTrainData = new IOOperations();
			ioTrainData.readInputData("./inputData/" + i + ".tra",false);
			this.featureMatrixTrainData = ioTrainData.getFeatureMatrix();
			this.trueLabelsTrainData = ioTrainData.getTrueLabels();

			IOOperations ioDevdata = new IOOperations();
			ioDevdata.readInputData("./inputData/" + i + ".dev",false);
			this.featureMatrixDevData = ioDevdata.getFeatureMatrix();
			this.trueLabelsDevData = ioDevdata.getTrueLabels();

			/* Set alfa=[0,0....0] initially */
			
			/* Initialization of alfa Vector */
			Matrix alfaMatrix = new Matrix(trueLabelsTrainData.size(), 1);
			for (int j = 0; j < trueLabelsTrainData.size(); j++) {
				alfaMatrix.set(j, 0, 0); /* W=[0,0,0.....] */
			}

			/* learn W using Training data using kernel Perceptron */
			KernelPerceptron perceptron = new KernelPerceptron();
			alfaMatrix = perceptron.computeAlfaMatrix(alfaMatrix, 5,
					featureMatrixTrainData, trueLabelsTrainData, 'b', exponent);

			/* loop through development data to classify */
			for (int Xi = 0; Xi < featureMatrixDevData.size(); Xi++) {
				Matrix featureOfXi = featureMatrixDevData.get(Xi);
				int trueLabel = trueLabelsDevData.get(Xi);/* True label of Xi */
				if (trueLabel == 1) {
					count++;
					
					double yi = perceptron.discriminantFn(
							featureMatrixTrainData, trueLabelsTrainData,
							featureOfXi, alfaMatrix, 'b', exponent);
					if (sgn(yi) > 0) {
						favorable++;
					}
				}
				
			}
			
			//System.out.println("Digit "+i+ " Success: "+a+" Total: "+b);
			
		}
		
		System.out.println("for Exponent : " + exponent);
		System.out.println("favorable : " + favorable);
		System.out.println("Total Count : " + count);
		accuracy = favorable / (new Double(count)) * 100;
		return accuracy;
	}
	/**
	 * Step function that takes yi and classify it to one of the possible two
	 * classes.
	 * 
	 * @param yi
	 * @return system Label
	 */
	public Integer sgn(Double yi) {
		Integer sysLabel = 0;
		if (yi >= 0) {
			sysLabel = +1;
		} else {
			sysLabel = -1;
		}
		return sysLabel;

	}
}
