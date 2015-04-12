#! /bin/bash
# Script to Learn the Models using Linear SVM Kernel.
# Author : Gaddipati Asish Kumar
# Version 1.0

cd /Users/AsishKumar/BitBucketRepos/MachineLearning/MLProject03/svm_light_OS10.8.4_i7

# With Normalization 
# Learn the models using SVM Linear Kernel for Pearson.
echo "With Normalization"
echo ""
echo ""
rankAlgo="Pearson" 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For Pearson with N = ${topN}"
	echo ""
	./svm_learn -t 0 -c 1 ./svmData/${rankAlgo}_FCount_${topN}_WNorm.svmtrain ./svmModels/${rankAlgo}_FCount_${topN}_WNorm.svmmodel
done
echo ""
echo ""
# Learn the models using SVM Linear Kernel for S2Noise.
rankAlgo="S2Noise" 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For S2Noise with N = ${topN}"
	echo ""
	./svm_learn -t 0 -c 1 ./svmData/${rankAlgo}_FCount_${topN}_WNorm.svmtrain ./svmModels/${rankAlgo}_FCount_${topN}_WNorm.svmmodel
done
echo ""
echo ""
# Learn the models using SVM Linear Kernel for TTest.
rankAlgo="TTest"
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For TTest with N = ${topN}"
	echo ""
	./svm_learn -t 0 -c 1 ./svmData/${rankAlgo}_FCount_${topN}_WNorm.svmtrain ./svmModels/${rankAlgo}_FCount_${topN}_WNorm.svmmodel
done
echo ""
echo ""

echo "Without Normalization"
echo ""
echo ""
# With Out Normalization
# Learn the models using SVM Linear Kernel for Pearson.
rankAlgo="Pearson" 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	# linear
	./svm_learn -t 0 -c 1 ./svmData/${rankAlgo}_FCount_${topN}_WOutNorm.svmtrain ./svmModels/${rankAlgo}_FCount_${topN}_WOutNorm.svmmodel
done
echo ""
echo ""
# Learn the models using SVM Linear Kernel for S2Noise.
rankAlgo="S2Noise" 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	# linear
	./svm_learn -t 0 -c 1 ./svmData/${rankAlgo}_FCount_${topN}_WOutNorm.svmtrain ./svmModels/${rankAlgo}_FCount_${topN}_WOutNorm.svmmodel
done
echo ""
echo ""
# Learn the models using SVM Linear Kernel for TTest.
rankAlgo="TTest"
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	# linear
	./svm_learn -t 0 -c 1 ./svmData/${rankAlgo}_FCount_${topN}_WOutNorm.svmtrain ./svmModels/${rankAlgo}_FCount_${topN}_WOutNorm.svmmodel
done