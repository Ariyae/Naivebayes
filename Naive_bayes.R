#Naive_Bayes Algorithm
data<-read.csv(file.choose(), stringsAsFactors = F,header = F)
View(data)
data<-data[-1, ]
sum(is.na(data))
str(data)
data<-data[ ,1:2]
View(data)
names(data)=c("LABEL","TEXT")
length(which(!complete.cases(data)))
data$LABEL=factor(data$LABEL,labels=c("HAM","SPAM"))
table(data$LABEL)
prop.table(table(data$LABEL))*100


#
install.packages("NLP")
install.packages("tm")
library(NLP)
library(tm)
datacorp<-VCorpus(VectorSource(data$TEXT))
print(datacorp)
inspect(datacorp[1:2])
as.character(datacorp[[1]])
lapply(datacorp[1:3],as.character)
datacorp_clean<-tm_map(datacorp,content_transformer(tolower))
as.character(datacorp[[1]])
as.character(datacorp_clean[[1]])
datacorp_clean<-tm_map(datacorp_clean,removeNumbers)
getTransformations()
datacorp_clean<-tm_map(datacorp_clean,removeWords,stopwords())
datacorp_clean<-tm_map(datacorp_clean,removePunctuation)
library(SnowballC)
datacorp_clean<-tm_map(datacorp_clean,stemDocument)
datacorp_clean<-tm_map(datacorp_clean,stripWhitespace)
as.character(datacorp_clean[[1]])
data_dtm<-DocumentTermMatrix(datacorp_clean)
View(data_dtm)
dtm_train<-data_dtm[4572, ]
dtm_test<-data_dtm[4573, ]
data_train_labels<-data[1:3, ]$LABEL
data_test_labels<-data[3:5, ]$LABEL
prop.table(table(data_train_labels))
prop.table(table(data_test_labels))
library(wordcloud)
wordcloud(datacorp_clean, random.order = T)
spam<-subset(data, LABEL == "SPAM")
ham<-subset(data, LABEL =="HAM")
wordcloud(spam$TEXT, max.words = 50, random.order = T, scale = c(3,0.5))
wordcloud(ham$TEXT, max.words = 50, random.order = T, scale = c(3,0.5))
install.packages("e1071")
library(e1071)
sms_freq_words<-findFreqTerms(dtm_train,1)
sms_dtm_freq_train<- dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- dtm_test[ , sms_freq_words]
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                    convert_counts)
View(sms_train)
sms_classifier<-naiveBayes(sms_train,data_train_labels, laplace=1)
sms_test_pred <- predict(sms_classifier, sms_test)
library(gmodels)
CrossTable(sms_test_pred, data_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
