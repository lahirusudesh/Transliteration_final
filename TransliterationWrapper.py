import Transliteration as tr
import GeneratePermutations as gp
import FindBestSuggestionLetterBased as bestword_letter_based
import json
import nltk
import csv
import string
import sys
from nltk.util import ngrams
from nltk.lm import NgramCounter
sys.setrecursionlimit(10**9)
special_chars = string.punctuation



class TranslterationWrapper:
    def __init__(self,data):
        self.data = data
        self.prepo_sin_words = []
        self.trigram_counter_model = []
        self.bigram_counter_model = []
        self.unigram_counter_model = []

    def transliterate(self):
        with self.data:
            tr.transliteratecsvFile(myFile)
    def loadUniquewordsInCorpus(self):
        with open('UniqueWords.txt', 'r', encoding='utf-8', errors='ignore') as file1:
            self.uniqueWordListInCorpus = json.load(file1)
            file1.close()

    def loadUniqueWordsInTest(self,text):
        UniqueWordList = []
        tokenized_sent = [list(map(str.lower, nltk.word_tokenize(sent)))
                          for sent in nltk.sent_tokenize(text)]
        for sent in tokenized_sent:
            for word in sent:
                if word not in UniqueWordList and not word.isnumeric() and word not in special_chars:
                    UniqueWordList.append(word);
        return UniqueWordList

    def getPermutationList(self,word):
        return gp.getPermutationsList(word,self.uniqueWordListInCorpus)

    def loadPrepositions(self):
        prepositions_file = open('sinhala_preposition.csv', 'r', encoding="utf-8-sig")
        with prepositions_file:
            dataReader = csv.DictReader(prepositions_file)
            for i, row in enumerate(dataReader):
                self.prepo_sin_words.append(row['sin'].lower())

    def loadThreeSyllableChunks(self):
        a_file = open('threeSyllable.txt', 'r', encoding='utf-8', errors='ignore')
        x = a_file.read()
        self.trigram_counter_model = json.loads(x)

    def loadTwoSyllableChunks(self):
        a_file = open('twoSyllable.txt', 'r', encoding='utf-8', errors='ignore')
        x = a_file.read()
        self.bigram_counter_model = json.loads(x)

    def setUpUnigramModel(self):
        newsListOne = []
        with open("combined.txt", 'r', encoding='utf-8', errors='ignore') as outfile:
            newslist = json.load(outfile)
        for news in newslist:
            newsListOne.extend(news)
        text = ' '.join([str(elem) for elem in newsListOne])
        tokenized_text = [list(map(str.lower, nltk.word_tokenize(sent))) for sent in nltk.sent_tokenize(text)]
        text_unigrams = [ngrams(sent, 1) for sent in tokenized_text]
        self.unigram_counter_model = NgramCounter(text_unigrams)

    def selectBestSuggestionUsingLetters(self,permutationList,word):
        return bestword_letter_based.selectBestSuggestionUsingLetters(self.unigram_counter_model,self.bigram_counter_model,self.trigram_counter_model,permutationList,word)

if __name__ == '__main__':
    myFile = open('Part_5_401_500.csv', 'r+', encoding="utf-8")
    obj = TranslterationWrapper(myFile)
    obj.transliterate()
    obj.loadUniquewordsInCorpus()
    obj.loadPrepositions()
    obj.loadTwoSyllableChunks()
    obj.setUpUnigramModel()
    tranliterated_l1_file = open('transliterated.csv','r',encoding='utf-8')
    tranliterated_l2_file = open('transliterated_l2.csv', 'w', encoding="utf-8", newline='')
    myFields = ['singlish_news', 'transliterated_l1', 'transliterated_l2','man_written']
    with tranliterated_l2_file:
        dataWriter = csv.DictWriter(tranliterated_l2_file, fieldnames=myFields)
        dataWriter.writeheader()
        with tranliterated_l1_file:
            permutation_list = []
            dataReader = csv.DictReader(tranliterated_l1_file)
            for row in dataReader:
                sent = row['transliterated_L1']
                man_written = row['man_written']
                uniquewordintest = obj.loadUniqueWordsInTest(sent)
                for word in uniquewordintest:
                    PermutationList = []
                    if word not in obj.uniqueWordListInCorpus and word not in obj.prepo_sin_words:
                        permutation_list_for_word = obj.getPermutationList(word)
                        bestword = obj.selectBestSuggestionUsingLetters(permutation_list_for_word,word)
                        sent.replace(word,bestword)
                dataWriter.writerow({'singlish_news': row['singlish_content'] , 'transliterated_l1': row['transliterated_L1'], 'transliterated_l2':sent, 'man_written':man_written })