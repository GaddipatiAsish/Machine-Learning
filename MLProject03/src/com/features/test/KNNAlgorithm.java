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
		String MethodName = "Pearson";
		int k = 1;
		for (int n = 0; n < NValue.length; n++) {
			System.out.println(MethodName+":"+NValue[n]);
			String trainData = "./input/knninputs/" + MethodName + "_FCount_"
					+ NValue[n] + ".knntrain";
			String validData = "./input/knninputs/" + MethodName + "_FCount_"
					+ NValue[n] + ".knnvalid";
			BufferedReader datafileTrain = readDataFile(trainData);
			Instances dataTrain = new Instances(datafileTrain);

			BufferedReader datafileValid = readDataFile(validData);
			Instances dataValid = new Instances(datafileValid);

			Classifier iBk = new IBk(k);
			iBk.buildClassifier(dataTrain);
			double num = 0, den = 0;
			for (int j = 0; j < dataValid.numInstances(); j++) {
				Instance ins = dataValid.instance(j);

				if (dataTrain.classAttribute().value(
						(int) iBk.classifyInstance(ins)) == dataTrain
						.classAttribute().value(
								(int) dataValid.instance(j).classValue())) {
					num++;
				}
				den++;
			}
			System.out.println(num / den);
		}

	}
}
