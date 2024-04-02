---
layout: main
---
# CS 7641 Group 44 

## Introduction and Background
The goal of the project is to develop a system that recommends books tailored to an individual's preferences. With a vast array of books available, choosing one can be challenging. This system aims to simplify the process by using a dataset from Wikipedia that includes information on 16,559 books, such as titles, authors, genres, and summaries. The current literature includes work related to predicting genre of fiction through short Goodreads descriptions (Sobkowicz, Kozlowski & Buczkowski 2017) or from texts of fanfictions (Rahul, Ayush, Agarwal & Vijay 2021) and predicting the likability of books using the book cover images (Maharjan, Montes-y-Gomez, Gonzales & Solorio 2018). By focusing on personal preferences and a wide range of genres, the project addresses the shortcomings of current recommendation systems, which tend to overlook the specific interests of readers and favor popular titles over personalized selections. The intention is to enhance the book-finding experience, making it more enjoyable and efficient for users seeking books that match their tastes. 

 

Dataset Link: [CMU Book Summary Dataset](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset)


## Problem statement

Our project aims to develop a natural language system to classify book genres from text summaries and further build a recommendation engine to suggest books based on summaries. 

## Method
Before applying any machine learning algorithms, data preprocessing was necessary, particularly for the genre and summary columns of our dataset. The genre information was formatted in a dictionary structure with multiple genres listed for each book. Within the dictionary structure, the keys were the genre ID from the database that the genre information was sourced from. The values were the genres themselves. For each data point, we first extracted the genres and transformed them into a single string, with commas separating each of the multiple genres. We then found that there were 226 unique genres in our dataset, but many of them were very specific with only 1 or 2 books belonging to each of those granular genres. Thus, we decided to manually consolidate the 226 unique genres into 11 broader genres: biography, comedy, dystopia, fantasy, fiction, history, literature, non-fiction, romance, short story, and thriller. After this pre-processing step, we found that each book was associated with 1 to 5 genres. Even with the number of genres we found that the 11 classes were very imbalanced, as presented in Figure 1 of the Results and Discussion section. We are fine-tuning our model to predict the smaller classes more accurately currently. For now, we will present our results only for the prediction of the 3 largest classes: fiction, literature, and thriller.  

 

For the summary text, we began by applying some natural language processing. With the help of the NLTK package, we tokenized the summary text, lemmatized the resulting tokens, and filtered for stopwords included in the NLTK stopwords dictionary. We have used the output to create TF-IDF (Term Frequency-Inverse Document Frequency) vectors. We also tried an alternate vectorizer in a GloVe (Global Vectors for Word Representation) model to create word embeddings. To implement this, we used the pre-trained "glove-wiki-gigaword-50" which was trained on 2 billion Tweets, 27 billion tokens, and 1.2 million vocabularies available on Hugging Face. Currently, we are in the process of learning about BERT (Bidirectional Encoder Representations from Transformers), another method of creating vector representations of text data.  

 

Using the vectorized book summary, we can finally train classification models to predict genre. The fact that our data contains more than 2 classes and the fact that each book can have more than 1 genre makes our problem more complex compared to a standard binary classification problem. We take a one vs. all approach for each of the classes. The 2 classifier models that yielded the best results for us are the logistic regression and random forest models. We will present the performance of each combination of the 2 vectorizers and 2 classifiers. In the next section we will present 12 F-1 scores in total – for each vectorizer and classifier combination and for each of the 3 classes.

## Results and Discussion 

### Class-wise F1 score

**Fiction**
|        | Logistic Regression | Random Forest |
| ------ | ------------------- | ------------- |
| GloVe  | 87.68               | 87.78         |
| TF-IDF | 87.43               | 88.38         |



**Literature**
|        | Logistic Regression | Random Forest |
| ------ | ------------------- | ------------- |
| GloVe  | 55.10               | 55.97         |
| TF-IDF | 43.19               | 29.52         |


**Thriller**
|        | Logistic Regression | Random Forest |
| ------ | ------------------- | ------------- |
| GloVe  | 62.36               | 61.48         |
| TF-IDF | 59.21               | 60.54         |


**Average across classes**
|        | Logistic Regression | Random Forest |
| ------ | ------------------- | ------------- |
| GloVe  | 68.38               | 68.41         |
| TF-IDF | 59.48               | 63.27         |

Our models perform decently well to predict book genres from their summaries. GloVe embeddings combined with Random Forest classifier yielded the best overall performance. However, further fine-tuning and experimentation with other techniques could potentially improve the classification accuracy, especially for the less dominant genres such as literature and thriller. 

Exploring alternative word embedding techniques (e.g., Word2Vec, FastText) or classifiers (e.g., Support Vector Machines, Gradient Boosting) could provide insights into which combinations work best for our specific task. Additionally, neural network-based architectures, such as recurrent neural networks (RNNs) or transformers, could be explored for their potential to capture sequential information in book summaries and improve classification accuracy. 

 
## Visualizations

![Number of books](assets/Number_Books.png)

The graph shows the number of books in each genre. The graph shows that fiction has the greatest number of books. We can see that we have very imbalanced classes in our data. 

![Publication Year](assets/Pub_Year.png)

This time series graph displays the number of books published over time in the three most popular genres: “fiction”, “thriller”, and “literature”. 

![Fiction](assets/Fiction.png)

This word cloud displays the most popular words in the summaries of all the books in the fiction genre. The three most common words are “find”, “kill”, and “take”. 

![Literature](assets/Literature.png)

The word cloud of the literature genre shows that words “find”, “go”, “one” and “tell” are the most repeated words among all the books in the genre. 

![Thriller](assets/Thriller.png)

The word cloud of the thriller genre shows that words “find”, “kill”, and “one” are the most repeated words among all the books in the genre. From Figure 3, 4, and 5, we can see there are overlaps in the set of commonly occurring words. Words like “kill” and “make” occur frequently in 2 of the 3 classes. This pattern can make it difficult for an algorithm to discern between the 2 classes. 

## Contribution Table 

|               Task               |       Contributor        |
| :------------------------------: | :----------------------: |
|              Report              |           All            |
|          Data Cleaning           |          Andrew          |
| Summary Text Data Pre-Processing |          Andrew          |
|    Exploratory Data Analysis     |           Urvi           |
|       Implementing models        |       Urvi, Aditya       |
|           Gantt Chart            |         Sanjana          |
|        Contribution Table        |     Dhruval, Sanjana     |
|          Visualizations          | Andrew, Sanjana, Dhruval |
|   Quantitative Scoring Metric    |          Aditya          |


## Gantt chart

<object data="assets/ML_GanttChart_midterm.pdf" width="1000" height="500" type='application/pdf'>Gantt chart</object>
