package com.features.test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KNNAlgorithm {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static void main(String args[]) throws Exception {
		int NValue[] = { 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700,
				800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
				19000, 20000 };

		boolean normalize = false;

		String MethodName = null;
		for (int h = 0; h < 3; h++) {
			System.out.println("\n\n");
			switch (h) {
			case 0:
				MethodName = "Pearson";
				break;
			case 1:
				MethodName = "S2Noise";
				break;
			case 2:
				MethodName = "TTest";
				break;
			}

			for (int l = 1; l < 4; l++) {
				int k=0;
				if(l==1)
					k=1;
				else if(l==2)
					k=5;
				else if(l==3)
					k=10;
				
				System.out.println("\n\n");
				for (int n = 0; n < NValue.length; n++) {

					String trainData;
					String validData;
					if (normalize) {/* Normalized data */
						trainData = "./input/knninputs/" + MethodName
								+ "_FCount_" + NValue[n] + "_WNorm.knntrain";
						validData = "./input/knninputs/" + MethodName
								+ "_FCount_" + NValue[n] + "_WNorm.knnvalid";
					} else {/* Actual data */
						trainData = "./input/knninputs/" + MethodName
								+ "_FCount_" + NValue[n] + "_WOutNorm.knntrain";
						validData = "./input/knninputs/" + MethodName
								+ "_FCount_" + NValue[n] + "_WOutNorm.knnvalid";
					}

					BufferedReader datafileTrain = readDataFile(trainData);
					Instances dataTrain = new Instances(datafileTrain);
					dataTrain.setClassIndex(dataTrain.numAttributes() - 1);

					BufferedReader datafileValid = readDataFile(validData);
					Instances dataValid = new Instances(datafileValid);
					dataValid.setClassIndex(dataValid.numAttributes() - 1);

					Classifier iBk = new IBk(k);
					iBk.buildClassifier(dataTrain);
					double num = 0, den = 0;
					for (int j = 0; j < dataValid.numInstances(); j++) {
						Instance ins = dataValid.instance(j);

						if (dataTrain.classAttribute().value(
								(int) iBk.classifyInstance(ins)) == dataTrain
								.classAttribute().value(
										(int) dataValid.instance(j)
												.classValue())) {
							num++;
						}
						den++;
					}
					System.out.println("Normalization : " + normalize
							+ " For K = " + k + " Algorithm :" + MethodName
							+ " N: " + NValue[n] + " Accuracy: " + (num / den)*100);
				}
			}
		}
	}
}
