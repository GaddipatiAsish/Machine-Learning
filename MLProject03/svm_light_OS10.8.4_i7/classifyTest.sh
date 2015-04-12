#! /bin/bash
# Author : Gaddipati Asish Kumar
# Version 1.0

cd /Users/AsishKumar/BitBucketRepos/MachineLearning/MLProject03/svm_light_OS10.8.4_i7

# With Normalization 
# Test the model and get the results
echo "With Normalization"
echo ""
echo ""
rankAlgo=Pearson 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For Pearson with N = ${topN}"
	echo ""
	./svm_classify  ./svmData/${rankAlgo}_FCount_${topN}_WNorm.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}_WNorm.svmmodel ./svmResults/${rankAlgo}_FCount_${topN}_WNorm.svmresult
done
echo ""
echo ""

rankAlgo=S2Noise 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For S2Noise with N = ${topN}"
	echo ""
	./svm_classify  ./svmData/${rankAlgo}_FCount_${topN}_WNorm.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}_WNorm.svmmodel ./svmResults/${rankAlgo}_FCount_${topN}_WNorm.svmresult
done
echo ""
echo ""

rankAlgo=TTest 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For TTest with N = ${topN}"
	echo ""
	./svm_classify  ./svmData/${rankAlgo}_FCount_${topN}_WNorm.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}_WNorm.svmmodel ./svmResults/${rankAlgo}_FCount_${topN}_WNorm.svmresult
done
echo ""
echo ""

echo "Without Normalization"
echo ""
echo ""
# With Out Normalization

rankAlgo=Pearson 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For Pearson with N = ${topN}"
	echo ""
	./svm_classify  ./svmData/${rankAlgo}_FCount_${topN}_WOutNorm.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}_WOutNorm.svmmodel ./svmResults/${rankAlgo}_FCount_${topN}_WOutNorm.svmresult
done
echo ""
echo ""

rankAlgo=S2Noise 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For S2Noise with N = ${topN}"
	echo ""
	./svm_classify  ./svmData/${rankAlgo}_FCount_${topN}_WOutNorm.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}_WOutNorm.svmmodel ./svmResults/${rankAlgo}_FCount_${topN}_WOutNorm.svmresult
done
echo ""
echo ""


rankAlgo=TTest 
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	echo "For TTest with N = ${topN}"
	echo ""
	./svm_classify  ./svmData/${rankAlgo}_FCount_${topN}_WOutNorm.svmvalid ./svmModels/${rankAlgo}_FCount_${topN}_WOutNorm.svmmodel ./svmResults/${rankAlgo}_FCount_${topN}_WOutNorm.svmresult
done
echo ""
echo ""

