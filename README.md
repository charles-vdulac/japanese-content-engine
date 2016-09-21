#Building a Simple Japanese Content-Based Recommender System in Python 

##Description

Online stores such as Amazon but also streaming services such as Netflix suffer 
from information overload. Customers can easily get lost in their large variety 
(millions) of products or movies. Recommendation engines help users narrow down 
the large variety by presenting possible suggestions. Also, browsing through all 
the blog posts from a website is time-consuming, especially as the number of 
posts is still increasing, a recommendation engine is useful by preventing 
information overload.

In this talk, I will show how to create a simple Japanese content-based 
recommendation system in Python for blog posts.


##Installation

* On Ubuntu

```
$ sudo aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file
$ pip install -r requirements.txt
```

* On Mac OSX

```
$ brew install mecab mecab-ipadic git curl xz
$ pip install -r requirements.txt
```

##Run it

```
$ cd /tmp/
$ wget https://dumps.wikimedia.org/jawiki/20160901/jawiki-20160901-pages-articles-multistream.xml.bz2
$ mkdir outputs
```

Then 

```
$ python make_corpus.py /tmp/jawiki-20160901-pages-articles-multistream.xml.bz2 /tmp/outputs 50
$ python content_engine.py /tmp/jawiki-20160901-pages-articles-multistream.xml.bz2 /tmp/outputs
```
