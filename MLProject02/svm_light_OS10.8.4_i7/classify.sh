#! /bin/bash
# Script to test for the models for digit 0 to 9.
# Author : Gaddipati Asish Kumar
# Version 1.0

cd /Users/AsishKumar/BitBucketRepos/MachineLearning/MLProject02/svm_light_OS10.8.4_i7
# loop to generate the 10 result files using Polynomial Kernel.
for i in {0..9}
do
./svm_classify ./svmipData/${i}.svmtes ./svmModels/${i}_polyK_deg5_epochs5.model ./svmTestResults/${i}_polyK_deg5_epochs5.result
done

# loop to generate the 10 result files using Gausian Kernel.
for i in {0..9}
do	
./svm_classify ./svmipData/${i}.svmtes  ./svmModels/${i}_GauK_sigma10_epochs5.model ./svmTestResults/${i}_GauK_sigma10_epochs5.result
done

# loop to generate the 10 result files using Gausian Kernel.
for i in {0..9}
do
./svm_classify  ./svmipData/${i}.svmtes ./svmModels/${i}_LinearKernel_epochs5.model ./svmTestResults/${i}_LinearKernel_epochs5.result
done
