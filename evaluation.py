# -*- coding: utf-8 -*-
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
import nltk.translate.gleu_score as gleu
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
import csv

def evaluateUsingBLEU(transliterated_file,row_name):
    score = 0
    dataReader = csv.DictReader(transliterated_file)
    for i, row in enumerate(dataReader):
        sent_man_written = row['man_written']
        reference = [tokenizer.tokenize(sent_man_written)]
        sent_machine_gen = row[row_name]
        candidate = tokenizer.tokenize(sent_machine_gen)
        score += sentence_bleu(reference, candidate, weights=(1.0, 0.0, 0.0, 0.0))
    avg_score = score / (i + 1)
    print("Score using BLEU : ",avg_score)

def evaluateUsingGLEU(transliterated_file,row_name):
    j, score = 0,0
    dataReader = csv.DictReader(transliterated_file)
    for j, row in enumerate(dataReader):
        sent_man_written = row['man_written']
        reference = [tokenizer.tokenize(sent_man_written)]
        sent_machine_gen = row[row_name]
        candidate = tokenizer.tokenize(sent_machine_gen)
        score += gleu.sentence_gleu(reference, candidate)
    avg_score = score / (j + 1)
    print("Score using GLEU : ",avg_score)

if __name__ == '__main__':
    transliterated_file = open('transliterated_l2.csv','r',encoding='utf-8')
    tokenizer = RegexpTokenizer('[\/\-\,\(\)\.\s+]', gaps=True)
    with transliterated_file:
        evaluateUsingBLEU(transliterated_file,'transliterated_l1')
        transliterated_file.seek(0)
        evaluateUsingGLEU(transliterated_file,'transliterated_l1')
    transliterated_file.close()
    transliterated_file_2 = open('transliterated_l2.csv', 'r', encoding='utf-8')
    with transliterated_file_2:
        evaluateUsingBLEU(transliterated_file_2,'transliterated_l2')
        transliterated_file_2.seek(0)
        evaluateUsingGLEU(transliterated_file_2,'transliterated_l2')
    transliterated_file.close()