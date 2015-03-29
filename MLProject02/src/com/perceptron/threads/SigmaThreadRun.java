package com.perceptron.threads;

import java.util.List;
import java.util.concurrent.Callable;

import weka.core.matrix.Matrix;

import com.perceptron.algorithms.KernelPerceptron;
import com.perceptron.io.IOOperations;

public class SigmaThreadRun implements Callable<Double> {
	double sigma;
	List<Matrix> featureMatrixTrainData; /* features Matrix: Training Data */
	List<Integer> trueLabelsTrainData;/* true labels : Training data */
	List<Matrix> featureMatrixDevData; /* features Matrix : Development data */
	List<Integer> trueLabelsDevData;/* true labels : Development data */

	public SigmaThreadRun(double sigma) {
		this.sigma = sigma;
	}

	/**
	 * 
	 */
	public Double call() throws Exception {
		Double accuracy = 0.0;
		int favorable = 0;
		int count = 0;
		for (int i = 0; i <= 9; i++) {

			/* Get the Training and development data */
			IOOperations ioTrainData = new IOOperations();
			ioTrainData.readInputData("./inputData/" + i + ".tra",false);
			this.featureMatrixTrainData = ioTrainData.getFeatureMatrix();
			this.trueLabelsTrainData = ioTrainData.getTrueLabels();

			IOOperations ioDevdata = new IOOperations();
			ioDevdata.readInputData("./inputData/" + i + ".dev",false);
			this.featureMatrixDevData = ioDevdata.getFeatureMatrix();
			this.trueLabelsDevData = ioDevdata.getTrueLabels();

			/* Set W=[0,0....0] initially */
			/* n denotes the data size N */
			/* Initial : load alfa matrix with 0 */
			
			Matrix alfaMatrix = new Matrix(trueLabelsTrainData.size(), 1);
			for (int j = 0; j < trueLabelsTrainData.size(); j++) {
				alfaMatrix.set(i, 0, 0);
			}


			/* learn W using Training data using kernel Perceptron Gausian */
			KernelPerceptron perceptron = new KernelPerceptron();
			alfaMatrix = perceptron.computeAlfaMatrix(alfaMatrix, 5,
					featureMatrixTrainData, trueLabelsTrainData, 'c', sigma);

			/* loop through development data to classify */
			for (int Xi = 0; Xi < featureMatrixDevData.size(); Xi++) {
				Matrix featureOfXi = featureMatrixDevData.get(Xi);
				double devTrueLabel= trueLabelsDevData.get(Xi);
				int trueLabel = trueLabelsDevData.get(Xi);/* True label of Xi */
				if (trueLabel == 1) {
					count++;

					double yi = perceptron.discriminantFn(
							featureMatrixTrainData, trueLabelsTrainData,
							featureOfXi, alfaMatrix, 'c', sigma);
					if (perceptron.sgn(yi) > 0) {
					//if(yi*devTrueLabel>=0){
						favorable++;
					}
				}

			}
			// System.out.println("Digit "+i+ " Success: "+a+" Total: "+b);
		}
		System.out.println("for Sigma : " + sigma);
		System.out.println("favorable : " + favorable);
		System.out.println("Total Count : " + count);
		accuracy = favorable / (new Double(count)) * 100;
		return accuracy;
	}

}
