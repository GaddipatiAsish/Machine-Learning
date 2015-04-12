# Author : G Asish Kumar*/
# Description: Plot that shows the comparison of Accuracy vs No of Top features Selected for 
# 			   S2Noise and With Normalisation of the features 

 x <- c(1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000)
 y <- c(59.00,66.33,81.00,86.33,87.67,90.33,93.00,93.67,93.67,92.00,92.67,92.67,93.00,93.00,93.67,93.67,94.00,94.33,94.00,93.67,93.67,92.33,92.33,92.33,92.33,92.33,92.33,92.33,92.33,92.33,92.33,92.33,91.67,91.67)
 plot(x, y, type='o', pch=1, col="blue", xlab="N", ylab="Accuracy(%)", ylim=c(0, 100), xlim=c(0,20000))