#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import json
import string
import argparse


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

# You may need to write regular expressions.


def isValidPunctuation(c):
    return (c == '.' or c == '?' or c == '!' or c == ',' or c == ';' or c == ':')


def isPunctuation(c):
    return (c == '.' or c == '?' or c == '!' or c == ',' or c == ';' or c == ':' or c == '[' or c == ']' or c == '{' or c == '}' or c == '(' or c == ')' or c == '<' or c == '>' or c == '@' or c == '#' or c == '$' or c == '%' or c == '^' or c == '&' or c == '*' or c == '-' or c == '|' or c == '\\' or c == '/' or c == '\'' or c == '"')


def linkText(matchobject):
    return matchobject.group('text')


def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    # text = re.sub('\[(?P<text>.+?)\]\(http.+\)', '\g<text>', text)
    text = re.sub('\[(?P<text>[^\[]+?)\]\(http.+\)',
                  linkText, text)  # FIGURE OUT MULTIPLE LINKS
    text = text.lower()
    splitText = text.split()
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

    parsed_text = ''
    for w in splitText:
        parsed_text += w + ' '
    parsed_text = parsed_text[:-1]

    unigrams = ''
    for w in splitText:
        if not(len(w) == 1 and isValidPunctuation(w)):
            unigrams += w + ' '
    unigrams = unigrams[:-1]

    i = 0
    bigrams = ''
    end = len(splitText)
    while i < end:
        bigram = ''
        if not(len(splitText[i]) == 1 and isValidPunctuation(splitText[i])):
            bigram += splitText[i] + '_'
            if i+1 < end and not(len(splitText[i+1]) == 1 and isValidPunctuation(splitText[i+1])):
                bigram += splitText[i+1]
                bigrams += bigram + ' '
            else:
                i = i + 1
        i = i+1
    bigrams = bigrams[:-1]

    i = 0
    trigrams = ''
    end = len(splitText)
    while i < end:
        trigram = ''
        if not(len(splitText[i]) == 1 and isValidPunctuation(splitText[i])):
            trigram += splitText[i] + '_'
            if i+1 < end and not(len(splitText[i+1]) == 1 and isValidPunctuation(splitText[i+1])):
                trigram += splitText[i+1] + '_'
                if i+2 < end and not(len(splitText[i+2]) == 1 and isValidPunctuation(splitText[i+2])):
                    trigram += splitText[i+2]
                    trigrams += trigram + ' '
                else:
                    i = i + 2
            else:
                i = i + 1
        i = i+1
    trigrams = trigrams[:-1]

    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.
    parser = argparse.ArgumentParser(description='Sanitize text.')
    parser.add_argument('file', metavar='file_name', type=str,
                        help='The json file whose contents you want to sanitize')

    args = parser.parse_args()
    fileName = args.file
    fileObject = open(fileName)
    lines = fileObject.readlines()
    for line in lines:
        # comment = json.loads(line)
        # body = comment['body']
        print(json.loads(line)['body'])
        print(sanitize(json.loads(line)['body']))
        print()
    print(fileName)
    fileObject.close()
    # print(args(args.integers))

    # We are "requiring" your write a main function so you can
    # debug your code. It will not be graded.
