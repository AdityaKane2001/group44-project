---
layout: main
---
# CS 7641 Group 44 

## Introduction and Background

The goal of the project is to develop a system that recommends books tailored to an individual's preferences. With a vast array of books available, choosing one can be challenging. This system aims to simplify the process by using a dataset from Wikipedia that includes information on 16,559 books, such as titles, authors, genres, and summaries. The current literature includes work related to predicting genre of fiction through short Goodreads descriptions (Sobkowicz, Kozlowski & Buczkowski 2017) or from texts of fanfictions (Rahul, Ayush, Agarwal & Vijay 2021) and predicting the likability of books using the book cover images (Maharjan, Montes-y-Gomez, Gonzales & Solorio 2018). By focusing on personal preferences and a wide range of genres, the project addresses the shortcomings of current recommendation systems, which tend to overlook the specific interests of readers and favor popular titles over personalized selections. The intention is to enhance the book-finding experience, making it more enjoyable and efficient for users seeking books that match their tastes. 

 

Dataset Link: [CMU Book Summary Dataset](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset)


## Problem statement

Our project aims to develop a natural language system to classify book genres from text summaries and further build a recommendation engine to suggest books based on user’s preferences. 

 

## Method

To comprehensively analyze our book dataset, we embark on a multi-step process. Initially, we will preprocess the summary column using Natural Language Toolkit (NLTK) methods like tokenization, stopword removal, lemmatization, and stemming. This prepares the textual data for further analysis. Subsequently, we employ word embedding techniques, leveraging TensorFlow's word2vec functionality, to transform the processed summaries into numerical vectors. These vectors encode the semantic meaning of the text, facilitating machine learning analysis. 

Moving forward, we delve into supervised learning techniques using scikit-learn. Using training decision tree and support vector machine models, we aim to classify books into genres based on their preprocessed summaries. Validation is conducted against the genre column in our dataset, ensuring the accuracy of our predictions. 

Furthermore, our analysis extends to providing personalized book recommendations. Utilizing unsupervised learning, we process user prompts akin to book summaries. By comparing the prompt's vector representation with those of the book summaries, we employ techniques like cosine similarity and k-means clustering to recommend books that closely match the user's interests. This comprehensive approach allows us to extract valuable insights and offer tailored recommendations, enhancing the user experience and engagement with the dataset. 

 

## Potential results and discussion

For evaluating supervised learning models, we can begin with the F1-score. We may also want to look at precision and recall, but to predict genre, assigning equal weight to precision and recall may be fine. To evaluate our unsupervised learning models, we can start with the silhouette coefficient which works well when we are not sure how many clusters are appropriate. However, especially once we have decided on the number of clusters, we would like to try other evaluation metrics like Davies-Bouldin Index for robustness. 

 


## References

[1] Genre Classification Using Character Networks: 

R. Rahul, A. Ayush, D. Agarwal, and D. Vijay, "Genre Classification using Character Networks," in Proc. 5th Int. Conf. Intelligent Computing and Control Systems (ICICCS), Madurai, India, 2021, pp. 216-222. doi: 10.1109/ICICCS51141.2021.9432303. 

 

[2] A Genre-Aware Attention Model to Improve the Likability Prediction of Books: 

S. Maharjan, M. Montes, F. A. González, and T. Solorio, "A Genre-Aware Attention Model to Improve the Likability Prediction of Books," in Proc. of the 2018 Conf. on Empirical Methods in Natural Language Processing (EMNLP), Brussels, Belgium, 2018, pp. 3381-3391. 

 

[3] Reading Book by the Cover—Book Genre Detection Using Short Descriptions 

A. Sobkowicz, M. Kozłowski, and P. Buczkowski, "Reading Book by the Cover—Book Genre Detection Using Short Descriptions," in A. Gruca, T. Czachórski, K. Harezlak, S. Kozielski, A. Piotrowska, Eds., Man-Machine Interactions 5. ICMMI 2017. Advances in Intelligent Systems and Computing, vol. 659, Springer, Cham, 2018. https://doi.org/10.1007/978-3-319-67792-7_43. 

 
## Gantt chart

<object data="assets/css/ML_GanttChart.pdf" width="1000" height="1000" type='application/pdf'></object>

## Contribution table

|           Task           |   Contributor   |
|:-----------------------|:-------------|
|     Problem Statement     |  All   |
|     Potential Dataset     | Andrew, Aditya  |
|     Literature Review     | Sanjana, Aditya |
| Exploratory Data Analysis |       All       |
|        Methodology        |     Dhruval     |
|     Potential Results     |     Andrew      |
|    Setup GitHub Pages     |  Aditya, Urvi   |
|        Gantt Chart        |      Urvi       |
|    Contribution Table     |     Sanjana     |
|      Video Recording      |     Andrew      |


<!-- ---
layout: default
---

Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
``` -->
