import re
import numpy as np


def extract_linguistic_features(text):

    text = str(text)

    # -------------------------
    # Words
    # -------------------------
    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text)

    word_count = len(words)

    if word_count == 0:
        return [0] * 12

    word_lengths = [len(word) for word in words]

    # -------------------------
    # Sentences
    # -------------------------
    sentences = re.split(r"[.!?]+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    sentence_count = len(sentences)

    sentence_lengths = [
        len(re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", sentence))
        for sentence in sentences
    ]

    # Remove empty sentence lengths
    sentence_lengths = [
        length for length in sentence_lengths
        if length > 0
    ]

    if sentence_lengths:
        avg_sentence_length = np.mean(sentence_lengths)
        sentence_length_std = np.std(sentence_lengths)
    else:
        avg_sentence_length = 0
        sentence_length_std = 0

    # -------------------------
    # Word statistics
    # -------------------------
    avg_word_length = np.mean(word_lengths)
    word_length_std = np.std(word_lengths)

    unique_words = len(set(word.lower() for word in words))

    vocabulary_diversity = unique_words / word_count

    # -------------------------
    # Punctuation
    # -------------------------
    punctuation_count = len(re.findall(r"[^\w\s]", text))

    punctuation_ratio = punctuation_count / max(len(text), 1)

    comma_count = text.count(",")
    question_count = text.count("?")
    exclamation_count = text.count("!")

    comma_ratio = comma_count / max(len(text), 1)
    question_ratio = question_count / max(len(text), 1)
    exclamation_ratio = exclamation_count / max(len(text), 1)

    # -------------------------
    # Paragraphs
    # -------------------------
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    paragraph_count = len(paragraphs)

    # -------------------------
    # Final feature vector
    # -------------------------
    features = [
        word_count,
        sentence_count,
        avg_sentence_length,
        sentence_length_std,
        avg_word_length,
        word_length_std,
        vocabulary_diversity,
        punctuation_ratio,
        comma_ratio,
        question_ratio,
        exclamation_ratio,
        paragraph_count
    ]

    return features