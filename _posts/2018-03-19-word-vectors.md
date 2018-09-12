---
layout: post
title: I Love, Love, Love Word Vectors!
subtitle: A simple introduction to word vectors, and how to use them
image: /img/posts/word_vectors/vectors.png
tags:
  - natural language processing
published: true
---

#### *What is this about?*
In this post, I walk through a typical workflow using word vectors to analyze unstructured text documents. I also show how a dimensionality reduction technique like TSNE can be leveraged to visualize patterns in the vector space.  
#### *Why should I care?*
If you know nothing about word vectors, they may seem mysterious, complex, or even magical. While these points may be true, they are also really simple, powerful, and easy to use.

## Background
In my time as an analytics professional, I've been in a lot of conversations about how best to analyze text. The most effective -and simplest- approach often includes word vectors.

In 2013, Tomas Mikolov et al from Google made a big (lasting) splash with their introduction of the Skip-gram method for vectorizing words -*[Efficient Estimation of Word Representations in
Vector Space](https://arxiv.org/pdf/1301.3781.pdf)*- followed shortly after by their NIPS paper *[Distributed Representations of Words and Phrases and their Compositionality. ](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)*

Though not the first known use of word vectors, their work was seminal because of their impressive performance, and the intrigue their results generated. They demonstrate the consistent relationship between like words in vector space:

<p align="center">
    <img src="../../img/posts/word_vectors/country_capital.png" width="500">
</p>

They also demonstrated the impressive learned mathematical properties of their vectors, such as:
* `vector(king) - vector(man) + vector(woman) ≈ vector(queen)`

* `vector(copper) - vector(Cu) + vector(zinc) ≈ vector(Zn)`

* `vector(Einstein) - vector(scientist) + vector(Messi) ≈ vector(midfielder)`

Needless-to-say, since 2013, many advancements have been made in the development of word vectors, and they have been ubiquitous with the latest research in NLP.

## Demo
Here's a fun example from Kaggle datasets -[Women's E-Commerce Clothing Reviews.](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. It spans over 20000 rows across 10 feature variables, including a free-form text review.

<p align="center">
    <img src="../../img/posts/word_vectors/reviews-data-frame.png" width="800">
</p>

Now lets find some word vectors.  [Glove -*Global Vectors for Word Representation*-](https://nlp.stanford.edu/projects/glove/) are some of the most widely used, and they can be freely downloaded at the link above. After downloading them, we can  load them up using the python package `gensim`:

``` python
import gensim
from gensim.scripts import glove2word2vec

glove2word2vec.glove2word2vec('path/to/glove.840B.300d.txt', 'reformatted_glove_file')

model = gensim.models.KeyedVectors.load_word2vec_format('reformatted_glove_file')
```

It's generally a good idea to clean up text data before dumping it into analyses. Here I'll do some very light text cleansing:

``` python
from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces, preprocess_string

def prep_string(x):
    filters = [strip_multiple_whitespaces, strip_punctuation, str.lower]
    clean_string = preprocess_string(x, filters)
    return clean_string

df['clean_review'] = df['Review Text'].apply(prep_string)  
```

Now we have reviews, which are collections of words, and we have a mapping of words to word vectors.  Our challenge is to compute a *single* vector to represent an *entire review*, so we can analyze them quantitatively. One simple and effective way to do this is by computing the average word vector of each review:

``` python
import numpy as np

def vectorize_document(document, model):
    vector_list = []
    for word in document:
        try:
            word_vector = model.get_vector(word)
            vector_list.append(word_vector)
        except KeyError:
            pass
    mean_document_vector = np.mean(np.array(vector_list), axis=0)
    return mean_document_vector

document_vectors = df['clean_review'].apply(lambda x: vectorize_document(x, model))
```

Let's wrap the results into a dataframe, and check its shape:

``` python
vec_df = pd.DataFrame(list(document_vectors))
vec_df.shape
```

`(22641, 300)`

Our resulting dataframe contains an observation for each review, and each observation spans 300 columns -the length of our input word vectors. In short, each observation in this dataframe represents the average word vector of its review. This may seem strange... ***what exactly is an average word vector*** anyway? Consider a simpler case: a vector space with 2 dimensions.  Say we have 2 observations in this 2D space -point A and point B.

<p align="center">
    <img src="../../img/posts/word_vectors/midpoint.png" width="800">
</p>

The average vector representation of A and B would be a new point right between them -point C. This concept holds true in word-vector space too, only across many more dimensions.

Beyond understanding the vector space, its *still* kind of a strange concept. A word vector represents the learned semantic / syntactic meaning of a word compressed into a vector of numbers. And, we're pooling these across all the words of a given review. Though it may feel weird to compute average word vectors, the result captures the ***collective meaning*** conveyed in a review.

Now we just need a way to explore our multi-dimensional results. One strategy is to use dimensionality reduction to reduce our 300 dimension vector space to 2 dimensions, so it can be visualized.  That's what I'll do here, with [T-SNE](https://lvdmaaten.github.io/tsne/). The [scikit documentation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) suggests an initial pruning of the vector space with PCA to suppress some noise and speed up the computation, so we start with that.

``` python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
compressed = pca.fit_transform(vec_df)
compressed.shape
```
`(22641, 50)`

The initial PCA pruning reduces our vector space from 300 dimensions to 50.

``` python
tsne_space = TSNE(n_components=2).fit_transform(compressed)
tsne_space.shape
```
`(22641, 2)`

TSNE finishes the job, leaving us with just 2 dimensions -which can be visualized much more easily. Now to visualize and explore these results. [Bokeh](https://bokeh.pydata.org/en/latest/) is one nice option for interactive visualization that might lend to exploring our new compressed vector space.

<div class="darker"> 
  <div class="container">
    {% include posts/word_vectors/tsne_bokeh.html %} 
  </div>
</div>

The above interactive visual can be used to explore the vector space. You can observe some clear patterns, where similar reviews appear closely together. For example, ...
