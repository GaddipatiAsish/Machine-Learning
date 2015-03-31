package com.perceptron.algorithms;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

import com.perceptron.io.IOOperations;

import weka.core.matrix.Matrix;

public class ConfMatrix {
	char algoType;
	double val;

	List<Matrix> featuresTestData;
	List<Integer> trueLabelsTestData;

	List<List<Matrix>> featuresModels = new ArrayList<List<Matrix>>();
	List<Matrix> alfaModels = new ArrayList<Matrix>();
	List<Matrix> wModels = new ArrayList<Matrix>();
	List<List<Integer>> trueLabelsModels = new ArrayList<List<Integer>>();

	public ConfMatrix(char algoType, double val) {
		this.algoType = algoType;
		this.val = val;
	}

	public void readModels() throws IOException {
		for (int p = 0; p < 10; p++) {
			List<Matrix> features = new ArrayList<Matrix>();
			List<Double> alfa = new ArrayList<Double>();
			List<Integer> trueLabels = new ArrayList<Integer>();
			String fileName = new String();
			if (algoType == 'c') {
				fileName = "./LearntModels/Digit_" + p + "_Epochs_5_polyK_deg_"
						+ val + ".model";
			} else if (algoType == 'e') {
				fileName = "./LearntModels/Digit_" + p
						+ "_Epochs_5_GauK_sigma_" + val + ".model";
			} else if (algoType == 'd') {
				fileName = "./LearntModels/Digit_" + p
						+ "_Epochs_5_AvgPolyK_deg_" + val + ".model";
			} else if (algoType == 'f') {
				fileName = "./LearntModels/Digit_" + p
						+ "_Epochs_5_AvgGauK_sigma_" + val + ".model";
			}
			String newLine;
			BufferedReader breader = new BufferedReader(new FileReader(
					new File(fileName)));

			while ((newLine = breader.readLine()) != null) {
				Matrix featOfXi = new Matrix(64, 1);
				int ind = newLine.indexOf(" ");
				double ai = Double.parseDouble(newLine.substring(0, ind));
				// System.out.println(ai);
				alfa.add(ai);
				trueLabels.add(1);

				// System.out.println("test");
				StringTokenizer tokenizr = new StringTokenizer(
						newLine.substring(ind), " ");
				int featureID = 1;
				while (featureID <= 64 && tokenizr.hasMoreTokens()) {
					String h = " " + featureID + ":";
					if (newLine.substring(ind).contains(h)) {
						String token = tokenizr.nextToken();
						double fValue = Double.parseDouble(token
								.substring(token.indexOf(":") + 1));
						// System.out.println(fValue);
						featOfXi.set(featureID-1, 0, fValue);
					} else {
						// System.out.println(featureID);
						featOfXi.set(featureID-1, 0, 0);
					}
					featureID++;
				}
				// featOfXi.transpose().print(5, 1);
				features.add(featOfXi);
			}
			breader.close();
			Matrix alfaMatrix = new Matrix(alfa.size(), 1);
			for (int i = 0; i < alfa.size(); i++) {
				alfaMatrix.set(i, 0, alfa.get(i));
			}

			featuresModels.add(features);
			alfaModels.add(alfaMatrix);
			trueLabelsModels.add(trueLabels);

		}
		System.out.println("Model read complete");
	}

	public Matrix creatConfusionMatrixPoly(String testdataFile)
			throws IOException {
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}

		/* Read the Test data from data file */
		IOOperations io = new IOOperations();
		io.readInputData(testdataFile,false);
		featuresTestData = io.getFeatureMatrix();
		trueLabelsTestData = io.getTrueLabels();

		KernelPerceptron perceptron = new KernelPerceptron();
		for (int i = 0; i < featuresTestData.size(); i++) {
			// System.out.println("Test Xi " + i);
			Matrix featureOfXi = featuresTestData.get(i);
			Integer tLabel = trueLabelsTestData.get(i);

			/* update confusion Matrix */

			List<Double> yiList = new ArrayList<Double>();
			for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
				Matrix alfaMatrix = alfaModels.get(sysLabel);
				List<Matrix> features = featuresModels.get(sysLabel);
				List<Integer> trueLabel = trueLabelsModels.get(sysLabel);

				// System.out.println(sysLabel+" : "+trueLabelsModels.get(sysLabel));
				// alfaMatrix.transpose().print(2, 1);
				//

				yiList.add(perceptron.discriminantFn(features, trueLabel,
						featureOfXi, alfaMatrix, 'b', val));

			}
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);

		}
		conMatrix.print(6, 0);
		return conMatrix;
	}

	public Matrix creatConfusionMatrixPolyAvg(String testdataFile)
			throws IOException {
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}

		/* Read the Test data from data file */
		IOOperations io = new IOOperations();
		io.readInputData(testdataFile,false);
		featuresTestData = io.getFeatureMatrix();
		trueLabelsTestData = io.getTrueLabels();

		System.out.println("size f" + featuresTestData.size());
		System.out.println("size t" + trueLabelsTestData.size());

		AvgKernelPerceptron perceptron1 = new AvgKernelPerceptron();
		for (int i = 0; i < featuresTestData.size(); i++) {
			// System.out.println("Test Xi " + i);
			Matrix featureOfXi = featuresTestData.get(i);
			Integer tLabel = trueLabelsTestData.get(i);

			/* update confusion Matrix */

			List<Double> yiList = new ArrayList<Double>();
			for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
				Matrix alfaMatrix = alfaModels.get(sysLabel);

				List<Matrix> features = featuresModels.get(sysLabel);
				List<Integer> trueLabel = trueLabelsModels.get(sysLabel);

				// System.out.println(sysLabel+" : "+trueLabelsModels.get(sysLabel));
				// alfaMatrix.transpose().print(2, 1);
				//

				yiList.add(perceptron1.discriminantFn(features, trueLabel,
						featureOfXi, alfaMatrix, 'b', val));

			}
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);

		}
		conMatrix.print(6, 0);
		return conMatrix;
	}

	public Matrix creatConfusionMatrixGau(String testdataFile)
			throws IOException {
		
		System.out.println("Sigma "+ val);
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}

		/* Read the Test data from data file */
		IOOperations io = new IOOperations();
		io.readInputData(testdataFile,false);
		featuresTestData = io.getFeatureMatrix();
		trueLabelsTestData = io.getTrueLabels();

		KernelPerceptron perceptron = new KernelPerceptron();
		for (int i = 0; i < featuresTestData.size(); i++) {
			// System.out.println("Test Xi " + i);
			Matrix featureOfXi = featuresTestData.get(i);
			Integer tLabel = trueLabelsTestData.get(i);

			/* update confusion Matrix */

			List<Double> yiList = new ArrayList<Double>();
			for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
				Matrix alfaMatrix = alfaModels.get(sysLabel);
				List<Matrix> features = featuresModels.get(sysLabel);
				List<Integer> trueLabel = trueLabelsModels.get(sysLabel);

				// System.out.println(sysLabel+" : "+trueLabelsModels.get(sysLabel));
				// alfaMatrix.transpose().print(2, 1);
				//
				double temp=perceptron.discriminantFn(features, trueLabel,
						featureOfXi, alfaMatrix, 'c', val);
				if(i==0){
					System.out.println("Temp"+temp);
				}
				yiList.add(temp);

			}
			// System.out.println(yiList);
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);

		}
		conMatrix.print(6, 0);
		return conMatrix;
	}

	public Matrix creatConfusionMatrixGauAvg(String testdataFile)
			throws IOException {
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}

		/* Read the Test data from data file */
		IOOperations io = new IOOperations();
		io.readInputData(testdataFile,false);
		featuresTestData = io.getFeatureMatrix();
		trueLabelsTestData = io.getTrueLabels();

		AvgKernelPerceptron perceptron = new AvgKernelPerceptron();
		for (int i = 0; i < featuresTestData.size(); i++) {
			// System.out.println("Test Xi " + i);
			Matrix featureOfXi = featuresTestData.get(i);
			Integer tLabel = trueLabelsTestData.get(i);

			/* update confusion Matrix */

			List<Double> yiList = new ArrayList<Double>();
			for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
				Matrix alfaMatrix = alfaModels.get(sysLabel);
				List<Matrix> features = featuresModels.get(sysLabel);
				List<Integer> trueLabel = trueLabelsModels.get(sysLabel);

				// System.out.println(sysLabel+" : "+trueLabelsModels.get(sysLabel));
				// alfaMatrix.transpose().print(2, 1);
				//

				yiList.add(perceptron.discriminantFn(features, trueLabel,
						featureOfXi, alfaMatrix, 'c', val));

			}
			// System.out.println(yiList);
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);

		}
		conMatrix.print(6, 0);
		return conMatrix;
	}

	public Matrix creatConfusionMatrixP(String testdataFile) throws IOException {
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}
		conMatrix.print(6, 0);
		
		/* Read the Test data from data file */
		IOOperations io = new IOOperations();
		io.readInputData(testdataFile,true);
		featuresTestData = io.getFeatureMatrix();
		trueLabelsTestData = io.getTrueLabels();
		Perceptron perceptron = new Perceptron();
		long totalTime=0;
		
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			/* read the specific data file */
			IOOperations io1 = new IOOperations();
			io1.readInputData("./inputData/" + sysLabel + ".tra",true);
			List<Matrix> features = io1.getFeatureMatrix();
			List<Integer> trueLabel = io1.getTrueLabels();
			Matrix wMatrix = new Matrix(65, 1, 0);
			long startTime = System.currentTimeMillis();
			wMatrix = perceptron.computeW(wMatrix, (int) val, features,
					trueLabel);
			long endTime   = System.currentTimeMillis();
			totalTime += endTime - startTime;
			wModels.add(wMatrix);
			featuresModels.add(features);
			trueLabelsModels.add(trueLabel);
		}
		System.out.println("Per. Learn Time in milli sec"+ totalTime);
		for (int i = 0; i < featuresTestData.size(); i++) {
			// System.out.println("Test Xi " + i);
			Matrix featureOfXi = featuresTestData.get(i);
			Integer tLabel = trueLabelsTestData.get(i);

			/* update confusion Matrix */

			List<Double> yiList = new ArrayList<Double>();
			for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
				Matrix wMatrix = wModels.get(sysLabel);
//				List<Matrix> features = featuresModels.get(sysLabel);
//				List<Integer> trueLabel = trueLabelsModels.get(sysLabel);

				yiList.add(perceptron.discriminantFn(featureOfXi, wMatrix));

			}
			// System.out.println(yiList);
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);
			
		}

		conMatrix.print(6, 0);
		return conMatrix;

	}

	public Matrix creatConfusionMatrixAvgP(String testdataFile)
			throws IOException {
		/* create confusion matrix and initialize to 0 */
		Matrix conMatrix = new Matrix(10, 10);
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			for (int trueLabel = 0; trueLabel < 10; trueLabel++) {
				conMatrix.set(sysLabel, trueLabel, 0);
			}
		}
		/* Read the Test data from data file */
		IOOperations io = new IOOperations();
		io.readInputData(testdataFile,true);
		featuresTestData = io.getFeatureMatrix();
		trueLabelsTestData = io.getTrueLabels();
		AvgPerceptron perceptron = new AvgPerceptron();
		long totalTime=0;
		for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
			/* read the specific data file */
			IOOperations io1 = new IOOperations();
			io1.readInputData("./inputData/" + sysLabel + ".tra",true);
			List<Matrix> features = io1.getFeatureMatrix();
			List<Integer> trueLabel = io1.getTrueLabels();
			Matrix wMatrix = new Matrix(65, 1, 0);
			long startTime = System.currentTimeMillis();
			wMatrix = perceptron.computeW(wMatrix, (int) val, features,
					trueLabel);
			long endTime   = System.currentTimeMillis();
			totalTime += endTime - startTime;
			wModels.add(wMatrix);
			if(sysLabel==0){
				wModels.get(0).print(5, 2);
			}
			featuresModels.add(features);
			trueLabelsModels.add(trueLabel);
		}
		System.out.println("Avg Per. Learn Timein milli sec"+ totalTime);
		for (int i = 0; i < featuresTestData.size(); i++) {
			// System.out.println("Test Xi " + i);
			Matrix featureOfXi = featuresTestData.get(i);
			Integer tLabel = trueLabelsTestData.get(i);

			/* update confusion Matrix */

			List<Double> yiList = new ArrayList<Double>();
			for (int sysLabel = 0; sysLabel < 10; sysLabel++) {
				Matrix wMatrix = wModels.get(sysLabel);
//				List<Matrix> features = featuresModels.get(sysLabel);
//				List<Integer> trueLabel = trueLabelsModels.get(sysLabel);

				yiList.add(perceptron.discriminantFn(featureOfXi, wMatrix));

			}
			// System.out.println(yiList);
			int label = yiList.indexOf(Collections.max(yiList));
			conMatrix.set(label, tLabel, conMatrix.get(label, tLabel) + 1);

		}

		conMatrix.print(6, 0);
		return conMatrix;

	}
}
