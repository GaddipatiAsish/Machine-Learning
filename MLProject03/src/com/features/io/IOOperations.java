package com.features.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
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

}
