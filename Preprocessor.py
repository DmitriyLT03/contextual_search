from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pymorphy2
from typing import List
import nltk
import string
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')


class Preprocessor:
    """Предобработка текста:
    приведение слов к нормальной форме
    удаление повторений
    """

    def __init__(self) -> None:
        self._stopwords_russian = stopwords.words('russian')
        self._morph = pymorphy2.MorphAnalyzer()
        self._tokenizer = TweetTokenizer(
            preserve_case=False,
            strip_handles=True,
            reduce_len=True
        )

    def _tokenize_sentences(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)

    def _preprocess_text(self, sentences: List[str]):
        clean_sentences = []
        for word in sentences:
            if (word not in self._stopwords_russian and word not in string.punctuation):
                clean_sentences.append(self._morph.parse(word)[0].normal_form)
        return clean_sentences

    def _token_preproc(self, text: str) -> List[str]:
        new_ts = self._tokenize_sentences(text)
        return self._preprocess_text(new_ts)

    def clean_text(self, text: str, remove_duplicates: bool = False) -> str:
        """
        Убирает спец символы
        Удаляет повторяющиеся слова
        Приводит слова к нормальной форме
        """
        text = text.translate(str.maketrans(
            "", "", string.punctuation)).lower()
        words = self._token_preproc(text)
        if remove_duplicates:
            # words = text.split(" ")
            return " ".join(sorted(set(words), key=words.index))
        return " ".join(words)
