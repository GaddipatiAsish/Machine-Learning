package com.features.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import weka.core.matrix.Matrix;

public class IOOperations {

	public Matrix readData(String dataFile, int samples, int featuresPerSample)
			throws IOException {

		Matrix trainData = new Matrix(samples, featuresPerSample, 0);
		BufferedReader breader = new BufferedReader(new FileReader(dataFile));
		String sample;/* data sample */
		/* add 0 for the missing feature ids. */
		int j = 1;/* sample number */
		while ((sample = breader.readLine()) != null) {/* read a sample */
			StringTokenizer tokenzr = new StringTokenizer(sample, " ");
			while (tokenzr.hasMoreTokens()) {
				String token = tokenzr.nextToken();
				String[] arr = token.split(":");
				trainData.set(j - 1, Integer.parseInt(arr[0]) - 1,
						Integer.parseInt(arr[1]));
			}
			j++;
		}
		return trainData;
	}

	public Matrix readLabels(String labelsFile, int samples) throws IOException {
		Matrix labels = new Matrix(samples, 1);
		BufferedReader breader = new BufferedReader(new FileReader(labelsFile));
		String label;
		int j = 0;
		while ((label = breader.readLine()) != null) {
			labels.set(j, 0, Integer.parseInt(label));
			j++;
		}
		return labels;
	}

	public void writeToFileSVM(Matrix newFeatures, Matrix labels,
			String MethodName, boolean normalize, char type) throws Exception {
		String fileName;
		if (type == 'v') {/* for Validation FIles */
			if (normalize) {
				fileName = "./svm_light_OS10.8.4_i7/svmData/" + MethodName
						+ "_FCount_" + newFeatures.getColumnDimension()
						+ "_WNorm.svmvalid";
			} else {
				fileName = "./svm_light_OS10.8.4_i7/svmData/" + MethodName
						+ "_FCount_" + newFeatures.getColumnDimension()
						+ "_WOutNorm.svmvalid";
			}
		} else {/* for training files */
			if (normalize) {
				fileName = "./svm_light_OS10.8.4_i7/svmData/" + MethodName
						+ "_FCount_" + newFeatures.getColumnDimension()
						+ "_WNorm.svmtrain";
			} else {
				fileName = "./svm_light_OS10.8.4_i7/svmData/" + MethodName
						+ "_FCount_" + newFeatures.getColumnDimension()
						+ "_WOutNorm.svmtrain";
			}

		}
		File file = new File(fileName);
		BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		for (int row = 0; row < newFeatures.getRowDimension(); row++) {
			String line = new String();
			for (int col = 0; col < newFeatures.getColumnDimension(); col++) {
				if (newFeatures.get(row, col) != 0) {
					line += " " + (col + 1) + ":" + newFeatures.get(row, col);
				}
			}
			writer.append(labels.get(row, 0) + " " + line + "\n");
		}
		writer.close();
	}

	public void writeToFileKNN(Matrix newFeatures, Matrix trainlabels,
			String MethodName) throws IOException {
		String fileName = "./input/knninputs/" + MethodName + "_FCount_"
				+ newFeatures.getColumnDimension() + ".knntrain";
		File file = new File(fileName);
		BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		writer.append("@relation newFeatures\n");
		for (int col = 0; col < newFeatures.getColumnDimension(); col++) {
			writer.append("@attribute Attribute" + col + " NUMERIC\n");
		}
		writer.append("@attribute Class {Y,N}");
		writer.append("\n@data\n");
		for (int row = 0; row < newFeatures.getRowDimension(); row++) {
			String line = new String();
			for (int col = 0; col < newFeatures.getColumnDimension(); col++) {
				line += newFeatures.get(row, col) + ",";
			}
			line += ((trainlabels.get(row, 0) > 0) ? "Y" : "N") + "\n";
			writer.append(line);
		}
		writer.close();
	}

	public void writeRanksToFile(List<Integer> rankedFeatureid,
			String methodName) throws Exception {
		System.out.println(rankedFeatureid.size());
		String fileName = "./ranks/" + methodName + ".rank";
		File file = new File(fileName);
		BufferedWriter bwriter = new BufferedWriter(new FileWriter(file));
		Iterator<Integer> iterator = rankedFeatureid.iterator();
		while (iterator.hasNext()) {
			bwriter.append(iterator.next().toString());
			bwriter.append("\n");
		}
		bwriter.close();
	}

	public List<Integer> readRanksFromFile(String methodName) throws Exception {
		System.out.println("Read ranks for " + methodName);
		List<Integer> rankedFeatureid = new LinkedList<Integer>();
		String fileName = "./ranks/" + methodName + ".rank";
		File file = new File(fileName);
		BufferedReader breader = new BufferedReader(new FileReader(file));
		String line;
		while ((line = breader.readLine()) != null) {
			rankedFeatureid.add(Integer.parseInt(line));
		}
		return rankedFeatureid;
	}
}
