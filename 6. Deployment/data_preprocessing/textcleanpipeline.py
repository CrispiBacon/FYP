

import re
import string
import contractions
import wordninja
import emoji
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from spellchecker import SpellChecker
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Ensure reproducible language detection
DetectorFactory.seed = 42

# Download required NLTK resources (run once)
def ensure_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize spellchecker and lemmatizer
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()
_word_re = re.compile(r"^[a-z]+$")

# Abbreviation dictionary (shortened sample)
abbreviation_dict = {
    "u": "you", "ur": "your", "lol": "laughing out loud", "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing", "brb": "be right back", "idk": "i don't know",
    "tbh": "to be honest", "smh": "shaking my head", "wtf": "what the fuck", "omg": "oh my god", "thx": "thank you", "ty": "thank you", "np": "no problem",
    "yw": "you're welcome", "fyi": "for your information", "b4": "before", "nvm": "never mind", "gtg": "got to go", "ttyl": "talk to you later", "gr8": "great",
    "bff": "best friends forever", "cya": "see you", "imo": "in my opinion", "imho": "in my humble opinion", "jk": "just kidding", "afk": "away from keyboard", "irl": "in real life",
    "gg": "good game", "wp": "well played", "btw": "by the way", "dm": "direct message", "rn": "right now", "afaik": "as far as i know", "asap": "as soon as possible",
    "fml": "fuck my life", "ikr": "i know right", "ily": "i love you", "lmk": "let me know", "ppl": "people", "bc": "because", "cuz": "because",
    "tho": "though", "y": "why", "r": "are", "k": "okay", "n": "and", "w/": "with", "w/o": "without",
    "stfu": "shut the fuck up", "hmu": "hit me up", "g2g": "got to go", "wyd": "what are you doing", "wym": "what you mean", "wbu": "what about you", "wb": "welcome back",
    "ofc": "of course", "pls": "please", "plz": "please", "bday": "birthday", "fav": "favorite", "msg": "message", "fb": "facebook",
    "yt": "youtube", "ig": "instagram", "snap": "snapchat", "twt": "twitter", "ftw": "for the win", "icymi": "in case you missed it", "mfw": "my face when",
    "tfw": "that feeling when", "ftl": "for the loss", "roflmao": "rolling on the floor laughing my ass off", "atk": "at the keyboard", "atm": "at the moment", "a3": "anytime anywhere anyplace", "bak": "back at keyboard",
    "bbl": "be back later", "bbs": "be back soon", "bfn": "bye for now", "b4n": "bye for now", "brt": "be right there", "cu": "see you", "cul8r": "see you later",
    "faq": "frequently asked questions", "fc": "fingers crossed", "fwiw": "for what it's worth", "gal": "get a life", "gn": "good night", "gmta": "great minds think alike", "g9": "genius",
    "ic": "i see", "ilu": "i love you", "iow": "in other words", "kiss": "keep it simple stupid", "ldr": "long distance relationship", "ltns": "long time no see", "l8r": "later",
    "mte": "my thoughts exactly", "m8": "mate", "nrn": "no reply necessary", "oic": "oh i see", "pita": "pain in the ass", "prt": "party", "prw": "parents are watching",
    "qpsa?": "que pasa", "roflol": "rolling on the floor laughing out loud", "rotflmao": "rolling on the floor laughing my ass off", "sk8": "skate", "stats": "your sex and age", "asl": "age sex location", "ttfn": "ta ta for now",
    "u2": "you too", "u4e": "yours forever", "wtg": "way to go", "wuf": "where are you from", "w8": "wait", "7k": "sick laughter", "tntl": "trying not to laugh",
    "idu": "i don't understand", "imu": "i miss you", "adih": "another day in hell", "zzz": "sleeping tired", "wywh": "wish you were here", "time": "tears in my eyes", "bae": "before anyone else",
    "fimh": "forever in my heart", "bsaaw": "big smile and a wink", "bwl": "bursting with laughter", "csl": "can't stop laughing", "std": "sexually transmitted disease", "og" : "original"
}

class TextCleaner:
    def __init__(self):
        ensure_nltk_data()

    def clean(self, text: str) -> str:
        try:
            if not isinstance(text, str) or not text.strip():
                return "invalid input"

            # Lowercase
            text = text.lower()

            # Language detection
            try:
                if detect(text) != 'en':
                    return "invalid input"
            except LangDetectException:
                return "invalid input"

            # Remove URLs and HTML
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>', '', text)

            # Remove non-alphanumeric characters (except apostrophe)
            punct_no_apost = string.punctuation.replace("'", "")
            text = re.sub(f"[{re.escape(punct_no_apost)}]", " ", text)
            text = re.sub(r"[^a-z0-9\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            # Tokenize
            tokens = nltk.word_tokenize(text)

            # Expand abbreviations
            expanded = []
            for word in tokens:
                expansion = abbreviation_dict.get(word, word)
                expanded.extend(expansion.split())
            tokens = expanded

            # Expand contractions
            expanded = []
            for token in tokens:
                expanded_token = contractions.fix(token)
                expanded.extend(expanded_token.split())
            tokens = expanded

            # Spell correction and splitting
            tokens = self._clean_tokens(tokens)

            # Lemmatization
            tokens = [lemmatizer.lemmatize(t, self._get_wordnet_pos(t)) for t in tokens]

            # Convert emojis
            tokens = self._convert_emojis(tokens)

            # Remove numbers
            tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

            if not tokens:
                return "invalid input"

            return " ".join(tokens)
        except Exception:
            return "invalid input"

    def _clean_tokens(self, tokens):
        out = []
        for t in tokens:
            out.extend(self._clean_token(t))
        return out

    def _clean_token(self, token, min_split_vocab_frac=0.7):
        if not isinstance(token, str):
            return []
        if not _word_re.match(token):
            return [token]
        if spell[token] > 0:
            return [token]
        corr = spell.correction(token)
        if corr and spell[corr] > 0:
            return [corr]
        pieces = wordninja.split(token)
        if len(pieces) > 1:
            known_frac = sum(spell[p] > 0 for p in pieces) / len(pieces)
            if known_frac >= min_split_vocab_frac:
                return [p if spell[p] > 0 else spell.correction(p) for p in pieces]
        return [token]

    def _get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def _convert_emojis(self, tokens):
        new_tokens = []
        for token in tokens:
            replaced = emoji.replace_emoji(
                token,
                replace=lambda e, data: "_" + data['en'].replace(' ', '_') + "_" if data and 'en' in data else e
            )
            new_tokens.append(replaced)
        return new_tokens

def clean_text_pipeline(text: str) -> str:
    return TextCleaner().clean(text)