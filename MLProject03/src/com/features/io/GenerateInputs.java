package com.features.io;

import java.util.List;


import weka.core.matrix.Matrix;

public class GenerateInputs {
	public Matrix generateFeatures(Matrix data, Matrix labels,
			int NoOfTopFeatures, List<Integer> rankedFeatures, boolean normalise) {
		
		List<Integer> rankedKeys = rankedFeatures;
		
		Matrix newFeatures = new Matrix(data.getRowDimension(),
				NoOfTopFeatures, 0);
		for (int row = 0; row < data.getRowDimension(); row++) {
			/* loop through the top ranked features */
			for (int i = 0; i < NoOfTopFeatures; i++) {
				newFeatures.set(row, i, data.get(row, rankedKeys.get(i)));
			}
			if (normalise) {/* Normalize */
				Matrix line = newFeatures.getMatrix(row, row, 0,
						NoOfTopFeatures - 1);
				double norm = line.normF();
				line = line.times((norm > 0) ? (1 / norm) : 0);
				newFeatures.setMatrix(row, row, 0, NoOfTopFeatures - 1, line);
			}
		}
		return newFeatures;
	}

}
