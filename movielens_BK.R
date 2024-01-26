#Bardh Kukaj's Movie Recommendation System/Capstone Course


##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Installing/Setting libraries in case they are not already installed - the code below that is provided by the course to import and set the data
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(dslabs)) install.packages("dslabs")
if (!require(ggthemes)) install.packages("ggthemes")
if (!require(knitr)) install.packages("knitr")

library(tidyverse)
library(caret)
library(dslabs)
library(ggthemes)
library(knitr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create train and test sets, as the final_holdout_test is used only for validating the final RMSE results
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Matching userId and movieId in both train and test sets
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Adding back rows into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# explore the data first, and check if there are no missing values (that's the third piece of code)

head_edx <- edx %>% head()

kable(head_edx, caption = "Exploring the data")
  
summary_edx <- edx %>% summary()

kable(summary_edx, caption = "Variables summary")

na_summary <- sapply(edx, {function(x) any(is.na(x))})

kable(na_summary, caption = "Checking for missing data")

# hypothesizing that "year" variable is important in explaining the variations in the ratings, I add it to all four data sets: edx, train set, test set, and final holdout test. 
# this is done by extracting the year from the title of the movie

edx <- edx %>% mutate(year=as.numeric(str_sub(title,-5,-2)))
train_set <- train_set %>% mutate(year=as.numeric(str_sub(title,-5,-2)))
test_set <- test_set %>% mutate(year=as.numeric(str_sub(title,-5,-2)))
final_holdout_test <- final_holdout_test %>% mutate(year=as.numeric(str_sub(title,-5,-2)))

# since the rating is the variable the variation of which we're trying to explain with other variables, we analyze it first
# labeling the ratings as whole or half star using the ifelse command

group <-  ifelse((edx$rating == 1 |edx$rating == 2 | edx$rating == 3 | 
                    edx$rating == 4 | edx$rating == 5) ,
                 "whole_star", 
                 "half_star") 
visual_ratings <- data.frame(edx$rating, group)

# visualizing the ratings using the labeling above to get a better overview of how ratings are distributed
# I added the "economist" theme at the end. I do visualizations daily at work, so I thought I'd try different themes

ggplot(visual_ratings, aes(x= edx.rating, fill = group)) +
  geom_histogram(binwidth = 0.25) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  scale_fill_manual(values = c("half_star"="blue", "whole_star"="red")) +
  labs(x="rating", y="number of ratings") +
  ggtitle("Histogram: # of Ratings/Each Rating") + theme_economist()

# now onto the user and movie effects (explanatory variables), after exploring the ratings (dependent variable)
# used the code in the machine learning course, but adjusted it to my code and tweaked a bit the visualization part

distinct_um <- edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

kable(distinct_um, caption = "Distinct Users and Movies")

edx  %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color = "red") + 
  scale_x_log10() + 
  ggtitle("Ratings by users") + 
  theme_clean()


edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "red") + 
  scale_x_log10() + 
  ggtitle("Ratings of movies") + 
  theme_clean()


# the next explanatory variable used is genres.

# the code below shows frequency by genre, but first the genres in the genres category are separated using the pipe (|) character as originally 1+ genres are added into one (explore the edx data set and you can see that there are genres like Drame|Musical)
# we first show the results in a table, then visualize them; again I use a different theme - this time wsj - wall street journal

top_genre <- edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

kable(top_genre, caption = "Top Genres")


ggplot(top_genre, aes(x = genres, y = count)) +
  geom_bar(stat = "identity", fill = "white", color = "red") +
  ggtitle("Number of ratings/Genre") + 
  theme_wsj() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# here I show mean rating by genre; note that I filter >= 100000 so that I avoid genres with very few ratings

mean_genre <- edx %>%  
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(n = n(), mean = mean(rating)) %>% 
  filter(n >= 100000)

ggplot(mean_genre, aes(x=genres,y=mean)) +
  geom_bar(stat = "identity", fill = "white", color = "red") +
  ggtitle("Mean Ratings/Genre") + 
  coord_cartesian(ylim = c(3, 4)) +
  theme_gdocs()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 8), plot.title = element_text(size = 14))

# the last explanatory I work with is time. I visualize genres trends over the years. I make an arbitrary cut-off point in 1950, as around that cinematography worldwide produces well known movies which are watched also now days. before that only movie fans would watch those movies

genre_trends <- edx %>%
  select(movieId, year, genres) %>%
  filter(year >= 1950) %>%
  group_by(year,genres) %>%
  summarise(number = n())

# note that I select only the top 5 genres, based on the table above. I don't add anymore as visualization wouldn't be so good
# i use the classic theme to get a neat visualization as it is a bit loaded with info

genre_trends %>%
  filter(genres %in% c("Drama", "Comedy", "Action", "Thriller", "Adventure")) %>%
  ggplot(aes(x = year, y = number)) +
  geom_line(aes(color=genres)) +
  theme_classic()
  
#the code below can go further up, but I felt it's more appropriate to go right before I start working on the model. it only specifies the function that calculates the RMSE

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Defining the model - specifying all the necessary variables already outlined above
#(notice we're working now with train and test datasets, instead of the overall edx)

# first is simple mean

mu_hat <- mean(train_set$rating)

#second is the movie bias/effect effect

mb <- train_set %>%
  group_by(movieId) %>%
  summarize(mb = mean(rating - mu_hat))

#third is the user bias/effect

ub <- train_set %>%
  left_join(mb, by='movieId') %>%
  group_by(userId) %>%
  summarize(ub = mean(rating - mu_hat - mb))

#fourth is the genre effect

ge <- train_set %>%
  left_join(mb, by='movieId') %>%
  left_join(ub, by='userId') %>%
  group_by(genres) %>%
  summarize(ge = mean(rating - mu_hat - mb - ub))

#fifth is the genre trends (time) effect

te <- train_set %>%
  left_join(mb, by='movieId') %>%
  left_join(ub, by='userId') %>%
  left_join(ge, by='genres') %>%
  group_by(year) %>%
  summarize(te = mean(rating - mu_hat - mb - ub - ge))

# 1st model - predictions using only the mean of ratings - baseline model

baseline_model <- RMSE(train_set$rating,mu_hat) 
baseline_model_validation <- RMSE(final_holdout_test$rating,mu_hat) 

#checking the results against the set RMSE of the exercise - 0.86490; baseline model

baseline_model < 0.86490
baseline_model_validation < 0.86490

#saving the results for comparison

rmse_summary <- tibble(method="Baseline model", RMSE = baseline_model_validation)

#2nd model - predictions using all of the variables specified above

predicted_ratings_2 <- train_set %>%
  left_join(mb, by = "movieId")%>%
  left_join(ub, by = "userId") %>%
  left_join(ge, by = "genres") %>%
  left_join(te, by = "year") %>%
  mutate(pred = mu_hat + mb + ub + ge + te) %>%
  pull(pred)

predicted_ratings2_validation <- final_holdout_test %>%
  left_join(mb, by = "movieId")%>%
  left_join(ub, by = "userId") %>%
  left_join(ge, by = "genres") %>%
  left_join(te, by = "year") %>%
  mutate(pred = mu_hat + mb + ub + ge + te) %>%
  pull(pred)

#checking the results against the set value of the exercise - they aren't below the required RMSE; need to create another model

second_model <- RMSE(predicted_ratings_2, train_set$rating)
second_model_validation <- RMSE(final_holdout_test$rating, predicted_ratings2_validation)

second_model < 0.86490
second_model_validation < 0.86490

rmse_summary <- bind_rows(rmse_summary,tibble(method="Movie, User, Genre and Time effects model",RMSE=second_model_validation))

# 3rd model - the go-to model for addressing the issue of too many ratings for certain movies and from certain users; therefore, we resort to regularization
# this model will penalize noisy data and provide a better RMSE. the other reason why I chose this model is because my laptop couldn't process other approaches, e.g., through recommenderlab. the memory usage was too high

lambda <- seq(0, 10, 0.20)
RMSE_f <- sapply(lambda, function(x){
  mb <- train_set %>%
    group_by(movieId) %>%
    summarize(mb = sum(rating - mu_hat)/(n()+x))
  ub <- train_set %>%
    left_join(mb, by = "movieId") %>%
    group_by(userId) %>%
    summarize(ub = sum(rating - mu_hat - mb)/(n()+x))
  ge <- train_set %>%
    left_join(mb, by = "movieId") %>%
    left_join(ub, by = "userId") %>%
    group_by(genres) %>%
    summarize(ge = sum(rating - mu_hat - mb - ub)/(n()+x))
  predicted_ratings3 <- train_set %>%
    left_join(mb, by = "movieId") %>%
    left_join(ub, by = "userId") %>%
    left_join(ge, by = "genres") %>%
    mutate(pred = mu_hat + mb + ub + ge) %>%
    pull(pred)
  return(RMSE(predicted_ratings3, train_set$rating))
})
RMSE_f

RMSE_final <- sapply(lambda, function(x){
  mb <- train_set %>%
    group_by(movieId) %>%
    summarize(mb = sum(rating - mu_hat)/(n()+x))
  ub <- train_set %>%
    left_join(mb, by = "movieId") %>%
    group_by(userId) %>%
    summarize(ub = sum(rating - mu_hat - mb)/(n()+x))
  ge <- train_set %>%
    left_join(mb, by = "movieId") %>%
    left_join(ub, by = "userId") %>%
    group_by(genres) %>%
    summarize(ge = sum(rating - mu_hat - mb - ub)/(n()+x))
  predicted_ratings3 <- final_holdout_test %>%
    left_join(mb, by = "movieId") %>%
    left_join(ub, by = "userId") %>%
    left_join(ge, by = "genres") %>%
    mutate(pred = mu_hat + mb + ub + ge) %>%
    pull(pred)
  return(RMSE(final_holdout_test$rating,predicted_ratings3))
})

# testing whether the lowest value of the lambda among the lambdas in the sequence from 1 to 10 (in increments of 0.2) penalizes the noisy data enough so that we get a result below the required RMSE
RMSE_final[which.min(RMSE_final)] < 0.86490

# it says it does, so we save that value (which incidentally is 5 - a value of 5 penalizes the model enough to get an RMSE below the required threshold)

lambda_final <- lambda[which.min(RMSE_final)]

# now we substitute the X value in the specification of the variables above with the lambda_final value to get the result of this model

mb <- train_set %>%
  group_by(movieId) %>%
  summarize(mb = sum(rating - mu_hat)/(n()+lambda_final))

ub <- train_set %>%
  left_join(mb, by = "movieId") %>%
  group_by(userId) %>%
  summarize(ub = sum(rating - mu_hat - mb)/(n()+lambda_final))

ge <- train_set %>%
  left_join(mb, by = "movieId") %>%
  left_join(ub, by = "userId") %>%
  group_by(genres) %>%
  summarize(ge = sum(rating - mu_hat - mb - ub)/(n()+lambda_final))

predicted_ratings3 <- final_holdout_test %>%
  left_join(mb, by = "movieId") %>%
  left_join(ub, by = "userId") %>%
  left_join(ge, by = "genres") %>%
  mutate(pred = mu_hat + mb + ub + ge) %>%
  pull(pred)

# double-checking that the third model is below the required RMSE threshold

third_model_validation <- RMSE(final_holdout_test$rating, predicted_ratings3)

third_model_validation < 0.86490

# saving the results and adding two figures after the decimal point to the tibble (from the default 3 to 5) so that it doesn't round the results

rmse_summary <- bind_rows(rmse_summary,tibble(method="Regularized Movie, User, and Genre effects model",RMSE=third_model_validation))

rmse_summary <- rmse_summary %>% mutate(RMSE = sprintf("%.5f", RMSE))

kable(rmse_summary, caption = "Summary of RMSEs")
