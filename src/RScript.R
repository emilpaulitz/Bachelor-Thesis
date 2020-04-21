setwd("C:/Users/emilp/Documents/Uni/Bachelorarbeit/")

n <- 10 # sample size
tab<-sample(c(0,1), replace=TRUE, size=n)
tab # gibt den vektor von 0-en und 1-en aus
fdrQVal <- function(tab){
  fdr <- 1-(cumsum(tab)/c(1:length(tab)))
  tab$fdr <- fdr
  tab$qval <- rev(cummin(rev(fdr)))
  return(tab)
}
fdrQVal(tab) # berechnet FDRs und q-values
