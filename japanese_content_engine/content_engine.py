
import os
import bz2
import logging
import sys

from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
from gensim.similarities import MatrixSimilarity
from gensim.corpora.wikicorpus import filter_wiki, extract_pages
from gensim.models import TfidfModel


# Add current dir to Python path
sys.path.append(os.path.dirname(os.getcwd()))

from japanese_content_engine.ja_tools import extract_jp_entities  # noqa


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
logging.root.level = logging.INFO

logger = logging.getLogger('gensim.corpora.wikicorpus')


def load_doc(dump_file, title, filter_namespaces=('0',)):
    """
    Load a wikipedia article from its title.
    """
    bz2_file = bz2.BZ2File(dump_file)
    for page_title, text, pageid in extract_pages(bz2_file, filter_namespaces):
        if page_title == title:
            text = filter_wiki(text)
            tokens = extract_jp_entities(text)
            return tokens, pageid
    return None, None


def load_titles(fname):
    logger.info("Loading titles from %s", fname)
    with open(fname, 'r') as f:
        return [l.replace('\n', '') for l in f.readlines()]


class ContentEngine(object):
    """
    Implement A Simple Content-Based Recommendation Engine
    """
    def __init__(self, dump_file, title_file, corpus_file, dict_file,
                 tfidf_model_file):

        self.dump_file = dump_file
        self.title_file = title_file
        self.corpus_file = corpus_file
        self.dict_file = dict_file
        self.tfidf_model_file = tfidf_model_file

        self.corpus = None
        self.dictionary = None
        self.tfidf_model = None
        self.index = None
        self.titles = []

        self.load()

    def load(self):
        """
        load the corpora created by `make_corpus.py`
        """
        self.corpus = MmCorpus(self.corpus_file)
        self.dictionary = Dictionary.load_from_text(self.dict_file)
        self.titles = load_titles(self.title_file)

        self.tfidf_model = TfidfModel.load(self.tfidf_model_file)
        self.index = MatrixSimilarity(self.tfidf_model[self.corpus])

    def predict(self, doc, num_best):
        """
        Project requested document to the TF-IDF space and compute similarities
        """
        vec_bow = self.dictionary.doc2bow(doc)
        vec_tfidf = self.tfidf_model[vec_bow]  # convert the doc to TFIDF space
        self.index.num_best = num_best + 1
        return self.index[vec_tfidf]

    def get_prediction(self, title, num_best=5):
        """
        Return the `num` most similar documents with `title`
        """
        doc, pageid = load_doc(self.dump_file, title)
        logger.info(
            "Requested document: Title={} (pageid={})".format(title, pageid)
        )

        best_docs = self.predict(doc, num_best)

        for doc_id, score in best_docs[1:]:
            title = self.titles[doc_id]
            logger.info(
                "https://ja.wikipedia.org/wiki/{}, score={}".format(title, score)
            )

        return best_docs


def main():

    # check and process input arguments
    if len(sys.argv) < 3:
        raise RuntimeError(globals()['__doc__'] % locals())

    dump_file, outputs_dir = sys.argv[1:3]

    if not os.path.isdir(outputs_dir):
        raise SystemExit(
            "Error: The output directory (%s) does not exist. "
            "Run first `make_corpus`", outputs_dir
        )

    engine = ContentEngine(
        dump_file=dump_file,
        title_file=os.path.join(outputs_dir, "wikipedia_titles"),
        corpus_file=os.path.join(outputs_dir, "wikipedia_bow.mm"),
        dict_file=os.path.join(outputs_dir, "wikipedia_wordids.txt.bz2"),
        tfidf_model_file=os.path.join(outputs_dir, "wikipedia.tfidf_model")
    )

    engine.get_prediction(title="地理学")


if __name__ == '__main__':
    main()
