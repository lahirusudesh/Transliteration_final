import Transliteration as tr
import GeneratePermutations as gp
import json
import nltk
import csv
import string
import sys
sys.setrecursionlimit(10**9)
special_chars = string.punctuation



class TranslterationWrapper:
    def __init__(self,data):
        self.data = data
        self.prepo_sin_words = []
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
        return gp.getPermutationsList(word)

    def loadPrepositions(self):
        prepositions_file = open('sinhala_preposition.csv', 'r', encoding="utf-8-sig")
        with prepositions_file:
            dataReader = csv.DictReader(prepositions_file)
            for i, row in enumerate(dataReader):
                self.prepo_sin_words.append(row['sin'].lower())

if __name__ == '__main__':
    myFile = open('../Part_5_401_500.csv', 'r+', encoding="utf-8")
    obj = TranslterationWrapper(myFile)
    obj.transliterate()
    obj.loadUniquewordsInCorpus()
    obj.loadPrepositions()
    tranliterated_l1_file = open('transliterated.csv','r',encoding='utf-8')
    permutation_file = open('permutations.csv', 'w', encoding="utf-8", newline='')
    myFields = ['sinhala_word', 'permutation_list']
    with permutation_file:
        dataWriter = csv.DictWriter(permutation_file, fieldnames=myFields)
        with tranliterated_l1_file:
            permutation_list = []
            dataReader = csv.DictReader(tranliterated_l1_file)
            for row in dataReader:
                sent = row['transliterated_L1']
                uniquewordintest = obj.loadUniqueWordsInTest(sent)
                for word in uniquewordintest:
                    PermutationList = []
                    if word not in obj.uniqueWordListInCorpus and word not in obj.prepo_sin_words:
                        permutation_list_for_word = obj.getPermutationList(word)
                        dataWriter.writerow({'sinhala_word': word, 'permutation_list': permutation_list_for_word })
                    else:
                        dataWriter.writerow({'sinhala_word': word, 'permutation_list': '' })