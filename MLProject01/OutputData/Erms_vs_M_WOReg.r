## R File to plot Graph for ERMS VS M values for Train and Test Data for With out Regularization ## 
## Author: Asish Kumar Gaddipati, ag615513@ohio.edu, Ohio University, Athens.                    ## 
mvalue <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
train <-c(0.20439011578503496,0.15142046562732744,0.14891243848972885,0.025680797378786358,0.025025296598739404,0.014849950319799599,0.012421669538109085,0.011659185083890461,0.002357420767293643,1.391649789383447E-4)
test <-c(0.1931182912183057,0.1489108720798555,0.1555844345674157,0.03840222897610668,0.03968647570598843,0.030251415174809972,0.04061930561081281,0.03010589284679013,0.10971874503325096,0.05089626851675621)
plot(mvalue, test, type='o', pch=10, col="green", xlab="M", ylab="Erms", ylim=c(0, 0.25), xlim=c(0,10))
lines(mvalue, train, type='o', pch=20, col="purple",lty=3)
legend("topleft", c("Test Data","Train Data"), cex=1.0, col=c("green","purple"), lty=1:2, lwd=2, bty="n")
