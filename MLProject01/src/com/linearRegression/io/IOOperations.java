package com.linearRegression.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * IOOperations class has methods to read the input data files and store them in
 * xValues and tValues lists.
 * 
 * @author AsishKumar
 *
 */
public class IOOperations {

	/**
	 *  ArrayList that stores the x Values of the input data 
	 */
	public List<Double> xValues = new ArrayList<Double>();
	/**
	 * ArrayList that stores the true labels of the corresponding input data 
	 */
	public List<Double> tValues = new ArrayList<Double>();

	/**
	 * @description readDataFromInput Method reads the data from the input file
	 *              and stores them in xValues and tValues Lists.
	 * @param arg
	 *            : data file as input
	 * @throws Exception
	 *             : Invalid File
	 */
	public void readDataFromInput(String arg) throws Exception {
		String line;
		BufferedReader breader = new BufferedReader(new FileReader(arg));
		while ((line = breader.readLine()) != null) { /*
													 * Read all points (x,t) one
													 * by one from data file.
													 */
			boolean flag = true;
			StringTokenizer sToknzr = new StringTokenizer(line, " ");
			while (sToknzr.hasMoreTokens()) {
				if (flag) {
					xValues.add(Double.parseDouble(sToknzr.nextToken()));
					flag = false;
				} else
					tValues.add(Double.parseDouble(sToknzr.nextToken()));
			}
		}
		breader.close();
	}

	/**
	 * @description readDataFromInput Method: Overridden method to read both
	 *              validation data and training data together.
	 * @param args
	 *            : data files as inputs.
	 * @throws Exception
	 *             : Invalid File
	 */
	public void readDataFromInput(String args[]) throws Exception {
		String line;
		for (int i = 0; i < args.length; i++) {
			BufferedReader breader = new BufferedReader(new FileReader(args[i]));
			while ((line = breader.readLine()) != null) { /*
														 * Read all points (x,t)
														 * one by one from data
														 * file.
														 */
				boolean flag = true;
				StringTokenizer sToknzr = new StringTokenizer(line, " ");
				while (sToknzr.hasMoreTokens()) {
					if (flag) {
						xValues.add(Double.parseDouble(sToknzr.nextToken()));
						flag = false;
					} else
						tValues.add(Double.parseDouble(sToknzr.nextToken()));
				}
			}
			breader.close();
		}
	}

	/**
	 * Description: getXValues returns the xValues list
	 * 
	 * @return xValues List
	 */
	public List<Double> getXValues() {
		return xValues;
	}

	/**
	 * Description: getTValues returns the tValues list
	 * 
	 * @return tValues List
	 */
	public List<Double> getTValues() {
		return tValues;
	}
}
