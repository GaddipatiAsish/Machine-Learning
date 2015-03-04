## R File to plot Graph for ERMS VS lnLambda values for Train and Valid Data for Linear Regression With Regularization ## 
## Author: Asish Kumar Gaddipati, ag615513@ohio.edu, Ohio University, Athens.                ## 
lnlambdavalue <- c(-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0)
train <-c(1.0097037924736774E-4,1.0190669165015224E-4,2.0715769498977975E-4,0.0017118373528319038,0.0038360607478740674,0.010073731914893993,0.012528343049910257,0.014000341931798286,0.051879212404507734,0.13318076308436566,0.20238025333379747)
valid <-c(0.044343701762335194,0.0443437039041987,0.04475672410033934,0.17601804652263486,0.4767914736367317,0.11200524795083106,0.04518160246416567,0.07491491117260687,0.04925148649189773,0.11113330087541107,0.21033871687378697)
plot(lnlambdavalue, valid, type='o', pch=10, col="green", xlab="ln Lambda", ylab="ERMS", ylim=c(0, 1), xlim=c(-50,0))
lines(lnlambdavalue, train, type='o', pch=20, col="purple",lty=3)
legend("topleft", c("Valid Data","Train Data"), cex=1.0, col=c("green","purple"), lty=1:2, lwd=2, bty="n")
