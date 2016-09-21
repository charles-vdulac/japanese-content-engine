
import os
import bz2
import multiprocessing
import logging
import sys

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora import MmCorpus
from gensim.models import TfidfModel
from gensim.corpora.wikicorpus import (
    ARTICLE_MIN_WORDS,
    filter_wiki,
    extract_pages,
    IGNORED_NAMESPACES,
)
from gensim.utils import chunkize


# Add current dir to Python path
sys.path.append(os.path.dirname(os.getcwd()))

from japanese_content_engine.ja_tools import extract_jp_entities  # noqa


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
logging.root.level = logging.INFO

logger = logging.getLogger('gensim.corpora.wikicorpus')


def process_article(args):
    """
    Parse a wikipedia article, returning its content as a list of tokens
    (utf8-encoded strings).
    """
    text, title, pageid = args
    text = filter_wiki(text)
    tokens = extract_jp_entities(text) if text else []
    return tokens, title, pageid


class WikiCorpus(TextCorpus):
    """
    Treat a wikipedia articles dump (\*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> # create word->word_id mapping, takes almost 8h
    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2')
    >>> # another 8h, creates a file in MatrixMarket format plus file id->word
    >>> MmCorpus.serialize('wiki_en_vocab200k.mm', wiki)

    """
    def __init__(self, fname,
                 processes=None,
                 dictionary=None,
                 max_batch=None,
                 metadata=True,
                 filter_namespaces=('0',)):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.

        `max_batch` allows you to parse only a subset of the full dump. Batch
        size is 10 * `num_cpu`.
        """
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.max_batch = max_batch
        self.metadata = metadata

        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes

        self.length = None

        self.titles = []
        if dictionary is None:
            self.dictionary = self.fill_dictionary()
        else:
            self.dictionary = dictionary

    def fill_dictionary(self, prune_at=2000000):
        """
        Update dictionary from a collection of documents. Each document is a list
        of tokens = **tokenized and normalized** strings (either utf8 or unicode).

        This is a convenience wrapper for calling `doc2bow` on each document
        with `allow_update=True`, which also prunes infrequent words, keeping the
        total number of unique words <= `prune_at`. This is to save memory on very
        large inputs. To disable this pruning, set `prune_at=None`.
        """
        if self.metadata:
            dictionary = Dictionary()

            for docno, item in enumerate(self.get_texts()):
                title, document = item
                self.titles.append(title)
                # log progress & run a regular check for pruning, once
                # every 10k docs
                if docno % 10000 == 0:
                    if prune_at is not None and len(dictionary) > prune_at:
                        dictionary.filter_extremes(
                            no_below=0, no_above=1.0, keep_n=prune_at
                        )
                    logger.info("adding document #%i to %s", docno, dictionary)

                # update Dictionary with the document
                dictionary.doc2bow(document, allow_update=True)

            logger.info(
                "built %s from %i documents (total %i corpus positions)",
                dictionary, dictionary.num_docs, dictionary.num_pos)

            return dictionary
        else:
            return Dictionary(self.get_texts())

    def get_texts(self):
        """
        Iterate over the dump, returning text version of each article as a list
        of tokens.

        Only articles of sufficient length are returned (short articles &
        redirects etc are ignored).

        Note that this iterates over the **texts**; if you want vectors,
        just use the standard corpus interface instead of this function::

        >>> for vec in wiki_corpus:
        >>>     print(vec)
        """
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        texts = (
            (text, title, pageid)
            for title, text, pageid
            in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces)
        )

        batch_idx = 0
        pool = multiprocessing.Pool(self.processes)
        # Process the corpus in smaller chunks of docs,
        # because multiprocessing.Pool is dumb and would load the entire input
        # into RAM at once...
        for group in chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):
                articles_all += 1
                positions_all += len(tokens)

                # article redirects and short stubs are pruned here
                to_ignored = any(
                    title.startswith(ignore + ':')
                    for ignore in IGNORED_NAMESPACES
                )
                if len(tokens) < ARTICLE_MIN_WORDS or to_ignored:
                    continue

                articles += 1
                positions += len(tokens)

                if self.metadata:
                    yield title, tokens
                else:
                    yield tokens

            batch_idx += 1
            if self.max_batch and batch_idx == self.max_batch:
                break

        pool.terminate()

        logger.info(
            "Finished iterating over Wikipedia corpus of %i documents with "
            "%i positions (total %i articles, %i positions before pruning "
            "articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
        )

        self.length = articles

    def save_titles(self, fname):
        logger.info("saving %d titles to %s", len(self.titles), fname)
        with open(fname, 'w') as f:
            for title in self.titles:
                f.write("{}\n".format(title))


# Wiki is first scanned for all distinct word types (~7M). The types that
# appear in more than 10% of articles are removed and from the rest, the
# DEFAULT_DICT_SIZE most frequent types are kept.
DEFAULT_DICT_SIZE = 100000


def create_corpus(dump_file, outputs_dir, max_batch=None):

    # Takes about 9h on a macbook pro, for 3.5m articles (june 2011)
    wiki = WikiCorpus(dump_file, max_batch=max_batch)

    # Only keep the most frequent words (out of total ~8.2m unique tokens)
    wiki.dictionary.filter_extremes(
        no_below=3,
        no_above=0.1,
        keep_n=DEFAULT_DICT_SIZE
    )

    # Save dictionary and bag-of-words (term-document frequency matrix).
    # Another ~9h

    corpus_file = os.path.join(outputs_dir, "wikipedia_bow.mm")
    dict_file = os.path.join(outputs_dir, "wikipedia_wordids.txt.bz2")
    titles_files = os.path.join(outputs_dir, "wikipedia_titles")

    MmCorpus.serialize(corpus_file, corpus=wiki, progress_cnt=10000)
    wiki.dictionary.save_as_text(dict_file)
    wiki.save_titles(titles_files)

    return corpus_file, dict_file, titles_files


def create_tfidf_corpus(corpus_file, dict_file, outputs_dir):

    # Load back the id->word mapping directly from file
    # This seems to save more memory, compared to keeping the
    # wiki.dictionary object from above
    dictionary = Dictionary.load_from_text(dict_file)

    # initialize corpus reader and word->id mapping
    mm = MmCorpus(corpus_file)

    tfidf_model_file = os.path.join(outputs_dir, "wikipedia.tfidf_model")
    tfidf_corpus_file = os.path.join(outputs_dir, "wikipedia_tfidf.mm")

    # build TF-IDF, ~50min
    tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
    tfidf.save(tfidf_model_file)

    # save tfidf vectors in matrix market format
    # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
    MmCorpus.serialize(tfidf_corpus_file, tfidf[mm], progress_cnt=10000)

    return tfidf_model_file, tfidf_corpus_file


def main():

    # check and process input arguments
    if len(sys.argv) < 3:
        raise RuntimeError(globals()['__doc__'] % locals())

    dump_file, outputs_dir = sys.argv[1:3]

    max_batch = None
    if len(sys.argv) >= 4:
        max_batch = int(sys.argv[3])
        logger.info("Max_batch=%s", max_batch)

    if not os.path.isdir(outputs_dir):
        raise SystemExit(
            "Error: The output directory (%s) does not exist. "
            "Create the directory and try again.", outputs_dir
        )

    corpus_file, dict_file, _ = create_corpus(
        dump_file, outputs_dir, max_batch
    )

    tfidf_model_file, tfidf_corpus_file = create_tfidf_corpus(
        corpus_file, dict_file, outputs_dir
    )

    logger.info("finished running")


if __name__ == '__main__':
    main()
