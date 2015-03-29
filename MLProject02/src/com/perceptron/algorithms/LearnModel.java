package com.perceptron.algorithms;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import weka.core.matrix.Matrix;

import com.perceptron.io.IOOperations;

/**
 * LearnModel class has methods that learns a model from the training data and
 * the parameters thats are passed
 * 
 * @author AsishKumar
 *
 */
public class LearnModel {
	boolean isAvg;
	char kernelType;
	double val;
	int epochs;

	/**
	 * 
	 * @param kernelType
	 *            defines what type of kernel.
	 * @param val
	 *            could be the sigma or exponent value depends upon kernelType
	 * @param epochs
	 *            epochs to run.
	 */
	public LearnModel(char kernelType, double val, int epochs, boolean isAvg) {
		this.kernelType = kernelType;
		this.val = val;
		this.epochs = epochs;
		this.isAvg = isAvg;
	}

	public boolean generateModelFileAvgKP() throws IOException {
		long totalTime=0;
		for (int i = 0; i <= 9; i++) {
			IOOperations ioTrainData = new IOOperations();
			ioTrainData.readInputData("./inputData/" + i + ".tra",false);
			List<Matrix> featureMatrixTrainData = ioTrainData
					.getFeatureMatrix();
			List<Integer> trueLabelsTrainData = ioTrainData.getTrueLabels();

			/* compose a file name and open it */
			String fileName = "Digit_" + i + "_Epochs_" + epochs;

			switch (kernelType) {
			case 'b':
				fileName += "_AvgPolyK_deg_" + val;
				break;
			case 'c':
				fileName += "_AvgGauK_sigma_" + val;
			}

			BufferedWriter bwriter = new BufferedWriter(new FileWriter(
					new File("./LearntModels/" + fileName + ".model")));

			AvgKernelPerceptron perceptron = new AvgKernelPerceptron();

			/* Initialization of alfa Vector */
			Matrix alfaMatrix = new Matrix(trueLabelsTrainData.size(), 1);
			for (int j = 0; j < trueLabelsTrainData.size(); j++) {
				alfaMatrix.set(j, 0, 0.0); /* W=[0,0,0.....] */
			}
			long startTime = System.currentTimeMillis();
			alfaMatrix = perceptron.computeAlfaMatrix(alfaMatrix, epochs,
					featureMatrixTrainData, trueLabelsTrainData, kernelType,
					val);
			long endTime   = System.currentTimeMillis();
			totalTime += endTime - startTime;
			/* write to model file */
			for (int k = 0; k < featureMatrixTrainData.size(); k++) {
				Matrix featuresOfXi = featureMatrixTrainData.get(k);
				String newLine = Double.toString(alfaMatrix.get(k, 0)
						* trueLabelsTrainData.get(k))
						+ " ";
				for (int l = 0; l < featuresOfXi.getRowDimension(); l++) {
					double te;
					if ((te = featuresOfXi.get(l, 0)) > 0) {
						newLine += (l+1) + ":" + te + " ";
					}
				}
				bwriter.write(newLine);
				bwriter.newLine();
			}
			bwriter.close();
		}
		System.out.println("Run Time in Milli Sec" +totalTime);
		return true;
	}

	public boolean generateModelFileKP() throws IOException {
		long totalTime=0;
		for (int i = 0; i <= 9; i++) {
			IOOperations ioTrainData = new IOOperations();
			ioTrainData.readInputData("./inputData/" + i + ".tra",false);
			List<Matrix> featureMatrixTrainData = ioTrainData
					.getFeatureMatrix();
			List<Integer> trueLabelsTrainData = ioTrainData.getTrueLabels();

			/* compose a file name and open it */
			String fileName = "Digit_" + i + "_Epochs_" + epochs;

			switch (kernelType) {
			case 'b':
				fileName += "_polyK_deg_" + val;

				break;
			case 'c':
				fileName += "_GauK_sigma_" + val;
			}

			BufferedWriter bwriter = new BufferedWriter(new FileWriter(
					new File("./LearntModels/" + fileName + ".model")));

			KernelPerceptron perceptron = new KernelPerceptron();

			/* Initialization of alfa Vector */
			Matrix alfaMatrix = new Matrix(trueLabelsTrainData.size(), 1);
			for (int j = 0; j < trueLabelsTrainData.size(); j++) {
				alfaMatrix.set(j, 0, 0); /* W=[0,0,0.....] */
			}
			long startTime = System.currentTimeMillis();
			alfaMatrix = perceptron.computeAlfaMatrix(alfaMatrix, epochs,
					featureMatrixTrainData, trueLabelsTrainData, kernelType,
					val);
			long endTime   = System.currentTimeMillis();
			totalTime += endTime - startTime;
			/* write to model file */
			for (int k = 0; k < featureMatrixTrainData.size(); k++) {
				Matrix featuresOfXi = featureMatrixTrainData.get(k);
				String newLine = Double.toString(alfaMatrix.get(k, 0)
						* trueLabelsTrainData.get(k))
						+ " ";
				for (int l = 0; l < featuresOfXi.getRowDimension(); l++) {
					double te;
					if ((te = featuresOfXi.get(l, 0)) > 0) {
						newLine += (l+1) + ":" + te + " ";
					}
				}
				bwriter.write(newLine);
				bwriter.newLine();
			}
			bwriter.close();
		}
		System.out.println("Run Time in Milli Sec" +totalTime);
		return true;
	}
}
