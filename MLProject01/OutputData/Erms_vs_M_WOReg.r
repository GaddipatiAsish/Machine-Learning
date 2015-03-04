## R File to plot Graph for ERMS VS M values for Train and Test Data for With out Regularization ## 
## Author: Asish Kumar Gaddipati, ag615513@ohio.edu, Ohio University, Athens.                    ## 
mvalue <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
train <-c(0.1434003776051384,0.1106534455634656,0.10944775619448517,0.01812091832686774,0.018040194821393064,0.010555221317039136,0.010343048208216249,0.0075958721725502675,0.007024773376670265,0.006624744041331562)
test <-c(0.19236548146213014,0.1446670274414901,0.1434619901743467,0.036017227959899575,0.03591310711493175,0.027554217629103,0.02881218526941761,0.026760351480826533,0.02953338558073436,0.0288106728038121)
plot(mvalue, test, type='o', pch=10, col="green", xlab="M", ylab="Erms", ylim=c(0, 0.25), xlim=c(0,10))
lines(mvalue, train, type='o', pch=20, col="purple",lty=3)
legend("topleft", c("Test Data","Train Data"), cex=1.0, col=c("green","purple"), lty=1:2, lwd=2, bty="n")
