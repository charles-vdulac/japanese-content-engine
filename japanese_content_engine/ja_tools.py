# -*- coding: utf8 -*-
#
# ja_tools
# Tools for Japanese
#
# Author: Romary Dupuis <romary.dupuis@altarika.com>
# Author: Charles Vallantin Dulac <charles.vdulac@altarika.com>
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import os
import re
import logging

import MeCab
from gensim.utils import to_unicode

logger = logging.getLogger(__name__)

try:
    from polyglot.text import Detector
    import pycld2

    # Disable polyglot warnings
    logging.getLogger('polyglot').setLevel(logging.ERROR)

except ImportError:
    logger.warning("Polyglot library is not available")
    Detector = None
    pycld2 = None


#
# Language Detection
#

# Note: not currently used in this project


def is_japanese(content, confidence_threshold=0.3):

    if Detector is None:
        return None

    try:
        detector = Detector(content, quiet=True)
    except pycld2.error:
        return None

    if not detector.languages:
        return None

    lang = detector.languages[0]

    if lang.code != 'ja' or lang.confidence < confidence_threshold:
        return False

    return True


kanjis = re.compile(r'[\u4E00-\u9FFF]+', re.UNICODE)  # == u'[一-龠々]+'
hiraganas = re.compile(r'[\u3040-\u309Fー]+', re.UNICODE)  # == u'[ぁ-んー]+'
katakanas = re.compile(r'[\u30A0-\u30FF]+', re.UNICODE)  # == u'[ァ-ヾ]+'


#
# Entity extraction
#


MECAB_IPADIC_PATH = os.environ.get(
    "MECAB_IPADIC_PATH",
    "/usr/local/lib/mecab/dic/mecab-ipadic-neologd/"
)

mecab = MeCab.Tagger('-d {}'.format(MECAB_IPADIC_PATH))

F1_NOUN = '名詞'
F1_PARTICLE = '助詞'
F1_CONJONCTION = '接続詞'
F1_INTERJECTION = '感動詞'
F1_VERB = '動詞'
F1_AUXILIARY_VERB = '助動詞'
F1_ADJECTIVE = '形容詞'
F1_SEPARATOR = '記号'
F2_NUMBER = '数'
F2_SUFFIX = '接尾'
F2_SUFFIX_COUNTER = '接尾'
F2_IND_VERB = '自立'
F2_NO_MEANING_NAMES = '非自立'
F2_PRONOUN = '代名詞'
F2_ADVERBS = '副詞可能'
F2_ADJECTIVE = '形容動詞語幹'
F2_PROPER_NOUN = '固有名詞'
F3_ADVERBS = '副詞可能'
F3_SUFFIX_COUNTER = '助数詞'
SENTENCE_SEPARATORS = ('句点', '')
STOPLIST = set('http https for a of the and to in co com jp'.split())


def jp_tokenizer(content, with_separators=True):
    output = []
    node = mecab.parseToNode(content)
    while node:
        features = node.feature.split(',')
        if node.surface != '':
            if not with_separators and features[0] != F1_SEPARATOR:
                output.append(node.surface)
            else:
                output.append(node.surface)
        node = node.next
    return output


def extract_jp_tokenized_sentences(content):
    output = []
    node = mecab.parseToNode(content)
    sentences = []
    sentence = []
    while node:
        features = node.feature.split(',')
        if node.surface == 'BOS/EOS' or (features[0] == F1_SEPARATOR and
                                         features[1] == '一般'):
            sentences.append(sentence)
            sentence = []
        else:
            if node.surface != '':
                sentence.append(node.surface)
        node = node.next
    if len(sentence) > 0:
        sentences.append(sentence)
    for sentence in sentences:
        tokenized_sentence = extract_jp_entities(''.join(sentence))
        if len(tokenized_sentence) > 0:
            output.append(tokenized_sentence)
    return output


def clean_entity(entity):
    keep = []
    for item in entity:
        if len(item) == 1 and kanjis.match(item):  # Regex reduces performances
            keep.append(item)
        elif 2 <= len(item) <= 15 and not item.startswith('_'):
            keep.append(item)
        else:
            continue

    return ' '.join(keep) if keep else ''


def extract_jp_entities(content):
    """
    Extract good candidates for entities in a text.
    """
    output = []
    node = mecab.parseToNode(content)
    entity = []
    while node:
        features = node.feature.split(',')
        try:
            surface = to_unicode(node.surface, errors='ignore').lower()
        except UnicodeDecodeError:
            surface = ''

        f1 = features[0]
        f2 = features[1]
        f3 = features[2]
        # Keep only nouns and proper nouns
        if f1 == F1_NOUN and \
                f2 != F2_IND_VERB and \
                f2 != F2_NO_MEANING_NAMES and \
                f2 != F2_ADVERBS and \
                f2 != F2_PRONOUN and \
                f2 != F2_ADJECTIVE and \
                f2 != F2_NUMBER and \
                f2 != F2_SUFFIX and \
                f3 != F3_ADVERBS and \
                surface not in STOPLIST:
            entity.append(surface)
        else:
            if entity:
                c_entity = clean_entity(entity)
                if c_entity:
                    output.append(c_entity)
                entity = []

        # Move to next morpheme
        node = node.next

    if entity:
        c_entity = clean_entity(entity)
        if c_entity:
            output.append(c_entity)

    return output
