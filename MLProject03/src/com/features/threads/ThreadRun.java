package com.features.threads;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.concurrent.Callable;

import weka.core.matrix.Matrix;

public class ThreadRun implements Callable<int[]> {
	int nValue;
	Matrix data;
	Matrix labels;
	String methodName;
	boolean normalize;
	char dataType;
	List<Integer> rankedFeatureid;

	public ThreadRun(int nValue, Matrix trainData, Matrix trainlabels,
			List<Integer> rankedFeatureid, String methodName,
			boolean normalize, char type) {
		this.nValue = nValue;
		this.data = trainData;
		this.labels = trainlabels;
		this.methodName = methodName;
		this.dataType = type;
		this.normalize = normalize;
		this.rankedFeatureid=rankedFeatureid;

	}

	public int[] call() throws Exception {

		System.out.println("Thread call recieved for N value: " + nValue);
		Matrix newFeatures = new Matrix(data.getRowDimension(), nValue, 0);
		for (int row = 0; row < data.getRowDimension(); row++) {
			/* loop through the top ranked features */
			for (int i = 0; i < nValue; i++) {
				newFeatures.set(row, i, data.get(row, rankedFeatureid.get(i)));
			}
			if (normalize) {/* Normalize */
				Matrix line = newFeatures.getMatrix(row, row, 0, nValue - 1);
				double norm = line.normF();
				line = line.times((norm > 0) ? (1 / norm) : 0);
				newFeatures.setMatrix(row, row, 0, nValue - 1, line);
			}
		}

		String fileName;
		if (dataType == 'v') {/* for Validation FIles */
			if (normalize) {
				fileName = "./input/knninputs/" + methodName + "_FCount_"
						+ newFeatures.getColumnDimension() + "_WNorm.knnvalid";
			} else {
				fileName = "./input/knninputs/" + methodName + "_FCount_"
						+ newFeatures.getColumnDimension()
						+ "_WOutNorm.knnvalid";
			}
		} else {/* for training files */
			if (normalize) {
				fileName = "./input/knninputs/" + methodName + "_FCount_"
						+ newFeatures.getColumnDimension() + "_WNorm.knntrain";
			} else {
				fileName = "./input/knninputs/" + methodName + "_FCount_"
						+ newFeatures.getColumnDimension()
						+ "_WOutNorm.knntrain";
			}

		}
		// System.out.println("File name of N value "+ nValue+ "is : "+fileName);
		
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
			line += ((labels.get(row, 0) > 0) ? "Y" : "N") + "\n";
			writer.append(line);
		}
		writer.close();
		System.out.println("Finished for Nvalue "+ nValue);
		int ret[] = { 0, 1 };
		return ret;
	}

}
