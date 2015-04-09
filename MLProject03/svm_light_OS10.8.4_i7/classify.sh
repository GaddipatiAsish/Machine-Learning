#! /bin/bash
# Script to test for the models for digit 0 to 9.
# Author : Gaddipati Asish Kumar
# Version 1.0


# loop to generate the result files using Polynomial Kernel.
cd /Users/AsishKumar/BitBucketRepos/MachineLearning/MLProject03/svm_light_OS10.8.4_i7
# loop to generate the model files using Linear Kernel.
rankAlgo=TTest  #Pearson"
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	# linear
	./svm_classify  ./svmipData/${rankAlgo}_FCount_${topN}.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}.svmmodel ./svmTestResults/${rankAlgo}_FCount_${topN}.svmresult
done
