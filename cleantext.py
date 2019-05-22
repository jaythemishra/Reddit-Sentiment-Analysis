#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import sys
import json

__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}


def isValidPunctuation(c):
    return (c == '.' or c == '?' or c == '!' or c == ',' or c == ';' or c == ':')


def isPunctuation(c):
    return (c == '.' or c == '?' or c == '!' or c == ',' or c == ';' or c == ':' or c == '[' or c == ']' or c == '{' or c == '}' or c == '(' or c == ')' or c == '<' or c == '>' or c == '@' or c == '#' or c == '$' or c == '%' or c == '^' or c == '&' or c == '*' or c == '-' or c == '|' or c == '\\' or c == '/' or c == '\'' or c == '"')


def replace_url_text(text):
    # Look for URLs in string
    text__removed_urls = ""
    url_regex = r'\[.*\]\(https?:\/\/[^\)]*\)'
    url_text_regex = r'\[.*\]'

    # find all instances of hyperlinks
    matched_urls = re.findall(url_regex, text)
    # split text into instances without hyperlinks
    text_without_urls = re.split(url_regex, text)
    # to store the replacement text in
    link_texts = []

    # extract hyperlink text
    for url_obj in matched_urls:
        link_text_match = re.match(url_text_regex, url_obj)
        if link_text_match:
            # get text and slice out brackets []
            link_texts.append(link_text_match.group(0)[1:-1])

    url_idx = 0
    while url_idx < len(link_texts):
        text__removed_urls += text_without_urls[url_idx] + link_texts[url_idx]
        url_idx += 1
    text__removed_urls += text_without_urls[url_idx]

    return text__removed_urls


def remove_punctuation(splitText):
    end = len(splitText)
    i = 0
    while i < end:
        word = splitText[i]
        if len(word) > 1 and isPunctuation(word[-1]):
            wordText = word[:-1]
            punc = word[-1]
            splitText[i] = wordText
            splitText.insert(i+1, punc)
            if isPunctuation(wordText[0]):
                prefix = wordText[0]
                wordText = wordText[1:]
                splitText[i] = wordText
                splitText.insert(i, prefix)
                end = end+1
            end = end+1
        elif len(word) > 1 and isPunctuation(word[0]):
            prefix = word[0]
            rest = word[1:]
            splitText[i] = rest
            splitText.insert(i, prefix)
            end = end+1
        i = i+1

    i = 0
    while i < end:
        word = splitText[i]
        if len(word) == 1 and not word.isalnum() and not isValidPunctuation(word):
            del splitText[i]
            end = end - 1
        else:
            i = i + 1

    return splitText


def get_unigrams(text):
    unigrams = ''
    for w in text:
        if not(len(w) == 1 and isValidPunctuation(w)):
            unigrams += w + ' '
    unigrams = unigrams[:-1]
    return unigrams


def get_bigrams(text):
    i = 0
    bigrams = ''
    end = len(text)
    while i < end:
        bigram = ''
        if not(len(text[i]) == 1 and isValidPunctuation(text[i])):
            bigram += text[i] + '_'
            if i+1 < end and not(len(text[i+1]) == 1 and isValidPunctuation(text[i+1])):
                bigram += text[i+1]
                bigrams += bigram + ' '
            else:
                i = i + 1
        i = i+1
    bigrams = bigrams[:-1]
    return bigrams


def get_trigrams(text):
    i = 0
    trigrams = ''
    end = len(text)
    while i < end:
        trigram = ''
        if not(len(text[i]) == 1 and isValidPunctuation(text[i])):
            trigram += text[i] + '_'
            if i+1 < end and not(len(text[i+1]) == 1 and isValidPunctuation(text[i+1])):
                trigram += text[i+1] + '_'
                if i+2 < end and not(len(text[i+2]) == 1 and isValidPunctuation(text[i+2])):
                    trigram += text[i+2]
                    trigrams += trigram + ' '
                else:
                    i = i + 2
            else:
                i = i + 1
        i = i+1
    trigrams = trigrams[:-1]
    return trigrams


def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """
    parsed_text = ""
    unigrams = ""
    bigrams = ""
    trigrams = ""

    text__replaced_link_text = replace_url_text(text)
    text__removed_extra_whitespace = ' '.join(text__replaced_link_text.split())

    text__split = text__removed_extra_whitespace.lower().split()
    text__removed_punctuation = remove_punctuation(text__split)

    parsed_text = ' '.join(text__removed_punctuation)
    unigrams = get_unigrams(text__removed_punctuation)
    bigrams = get_bigrams(text__removed_punctuation)
    trigrams = get_trigrams(text__removed_punctuation)

    # For debugging only
    # print('===== parsed =====\n\n', parsed_text, '\n\n')
    # print('===== unigrams =====\n\n', unigrams, '\n\n')
    # print('===== bigrams =====\n\n', bigrams, '\n\n')
    # print('===== trigrams =====\n\n', trigrams, '\n\n')

    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # Get filename from command line
    if len(sys.argv) < 2:
        sys.exit('No filename specified. Usage: python cleantext.py <filename>')

    filename = sys.argv[1]

    # Try opening, then reading file line by line
    # Extract value from JSON and pass to sanitize
    data = []
    sanitized_data = []

    # This is just one test case, comment this out for final submission
    print(sanitize(
        'He did BS his way in, I\'m also not the person that said they want to punch him (or anyone.) I think you\'re confusing posters.\n\nThe guy is known for not knowing what he\'s talking [about ](https://www.google.com/amp/s/amp.businessinsider.com/sebastian-gorka-trump-bio-profile-2017-2) \n\nHis credentials are well known to be [bogus.](https://www.rollingstone.com/politics/features/sebastian-gorka-the-west-wings-phony-foreign-policy-guru-w496912) \n\nAnd he\'s never served in the military or any intel agency as far as I can tell. He does however play Nazi   [dress up.](https://www.google.com/amp/s/www.nbcnews.com/news/world/amp/sebastian-gorka-made-nazi-linked-vitezi-rend-proud-wearing-its-n742851)'))

    # Uncomment below lines to read through the filename passed in
    # try:
    #     with open(filename, 'r') as json_file:
    #         for line in json_file:
    #             data.append(json.loads(line))

    #     for comment in data:
    #         sanitized_data.append(sanitize(comment['body']))

    #     print(sanitized_data)
    # except:
    #     sys.exit('There was a problem opening the file.')
    # finally:
    #     json_file.close()
