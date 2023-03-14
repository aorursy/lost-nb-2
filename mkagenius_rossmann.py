for (i in 1:1115) 
{
  tmp <- train[train$Store == i,]
  tmp <- tmp[tmp$Open == 1, ]
  fname <- paste("trainStore",i,".csv",sep="_")
  write.csv(tmp,fname, row.names = FALSE)
  
}


for (i in 1:1115) 
{
  tmp <- test[test$Store == i,]
  tmp <- tmp[tmp$Open == 1, ]
  fname <- paste("testStore",i,".csv",sep="_")
  write.csv(tmp,fname, row.names = FALSE)
  
}




trainArr <- list.files(pattern = glob2rx("train*_.csv"))

for (i in 1:length(trainArr))
{
  assign(trainArr[i], read.csv(trainArr[i]))
}

testArr <- list.files(pattern = glob2rx("testStore*_.csv"))

for (i in 1:length(testArr))
{
  assign(testArr[i],read.csv(testArr[i]))
}

final <- testArr[1]
for (i in 1:length(trainArr))
{
  tst <- get(testArr[i])
  
  trn <- get(trainArr[i])
  trn$StateHoliday <- as.factor(trn$StateHoliday)
  trn$DayOfWeek <- as.factor(trn$DayOfWeek)
  trn$Open <- as.factor(trn$Open)
  trn$Promo <- as.factor(trn$Promo)
  trn$SchoolHoliday <- as.factor(trn$SchoolHoliday)
  
  tst$StateHoliday <- as.factor(tst$StateHoliday)
  tst$DayOfWeek <- as.factor(tst$DayOfWeek)
  tst$Open <- as.factor(tst$Open)
  tst$Promo <- as.factor(tst$Promo)
  tst$SchoolHoliday <- as.factor(tst$SchoolHoliday)
  
  fit <- rpart(Sales ~ DayOfWeek + Open + Promo + StateHoliday + SchoolHoliday, data=trn, method="anova")
  Prediction <- predict(fit, tst)
  submit <- data.frame(Id = tst$Id, Sales = Prediction)
  if(i == 1)
  {
    final <- submit
  }
  else
  {
    final <- rbind(final, submit)
  }
}

write.csv(final, "final.csv", row.names = FALSE)

