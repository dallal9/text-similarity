import operator
import re
import time
import os 
import spacy

#from Skylar.models import Flow
#afrom Skylar.utils.utils import format_message
from ai.model.keras_similarr import keras_similar
from ai.model.utils.feature_extractor import extract_features
from ai.model.utils.nltk_util import mark_negation
from ai.model.utils.qclassifier import Qclassifier
from ai.model.utils.spelling.spelling import Spelling
from ai.skysentiment import get_sentiment_values_2 as get_sentiment_values

from sematch.semantic.similarity import WordNetSimilarity
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from cucco import Cucco

normalizr = Cucco()
normalizations = ['remove_extra_white_spaces', 'replace_punctuation',
                  'replace_symbols', 'remove_accent_marks']

class fmodel(object):
    def __init__(self):
        self.out = {}
        self.keras = keras_similar()
        self.classifier = Qclassifier()
        self.spell=Spelling()
        self.wn = WordNetSimilarity()
        self.en_nlp = spacy.load("en_core_web_md")
        self.stopwords_en=[]
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            'utils', 'stopwords_en.txt')) as f:

            self.stopwords_en = f.read().splitlines()

    def ent_nltk(self, sentence):
        ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
        iob_tagged = tree2conlltags(ne_tree)
        ents = [[0, 0, 10]]
        for i in range(len(iob_tagged)):
            each = iob_tagged[i]

            if each[2] != 'O':
                if ents[-1][2] == (i - 1):
                    ents[-1][0] += " " + each[0]
                    ents[-1][2] = i
                else:
                    ents.append([each[0], each[2][2:], i])
        if len(ents) > 1:
            ents = ents[1:]
            ents = [ent[0] for ent in ents]
        else:
            ents = []

        return ents

    def mini_similar(self, q1, q2):
        self.out = {'sim': 0, 'sim_per': 0.0, 'keras': 0, 'class': ["", ""],
                    'f_class': 0, "sentiment": [0, 0, 0],
                    "keywords": [[""], [""]], "numbers": [[], []],
                    "entities": [[], []], "max_keywords": 0, "keywords_sim": 0}
        regex = re.compile('[^a-zA-Z0-9]')
        q1 = regex.sub('', q1)
        q2 = regex.sub('', q2)
        if q1 == q2:
            self.out['sim'] = 1
            self.out['sim_per'] = 100
            return self.out
        else:
            s1 = self.wn.word_similarity(q1, q2, 'lin')
            print(s1)

            if s1 > 0.9:
                self.out['sim'] = 1
                self.out['sim_per'] = 100
                return self.out

            elif s1 > 0.8:
                self.out['sim'] = 1
                self.out['sim_per'] = s1  # max([s1,s2,s3])
                return self.out
        return self.out

    def is_one_word(self, q1, q2):
        l1 = q1
        l2 = q2
        flag1 = False
        flag2 = False
        stop = True
        word1 = ""
        word2 = ""

        if len(l1)!=len(l2):

            return False
        else:
            for i in range(len(l1)):
                    if l1[i].text != l2[i].text or l1[i].lemma_ != l2[i].lemma_: 
                        if(flag2):
                            return False
                            
                        elif l1[i].text in self.stopwords_en and l2[i].text in self.stopwords_en:
                            word1 = l1[i].text
                            word2 = l2[i].text
                            flag1 = True                                                   

                        else:
                            word1 = l1[i].lemma_
                            word2 = l2[i].lemma_
                            flag1 = True
                            flag2 = True
        if flag1:
            self.out = self.mini_similar(word1,word2)
            return True
            

    def similar(self, text, challenge):
        if not isinstance(text, str) or not isinstance(challenge, str):
            q1 = text
            q2 = challenge
        else:
            q1 = normalizr.normalize(text, normalizations)
            q2 = normalizr.normalize(challenge, normalizations)

        q1 = self.spell.correct_str(q1,True)
        q2 = self.spell.correct_str(q2,True)

        if (len(q1.split()) == 1 and len(q2.split()) == 1) or (q1 == q2):
            return self.mini_similar(q1, q2)
        regex = re.compile(u'/')  # [^a-zA-Z]')
        q1 = regex.sub('', q1)
        q2 = regex.sub('', q2)

        self.out = {'sim': 0, 'sim_per': 0.0, 'keras': 0.0, 'class': ["", ""],
                    'f_class': 0, "sentiment": [0, 0, 0],
                    "keywords": [[""], [""]], "numbers": [[], []],
                    "entities": [[], []], "max_keywords": 0,
                    "keywords_sim": 0.0}
        q1_neg_list = list(set(mark_negation(q1.split())[0]))
        q2_neg_list = list(set(mark_negation(q2.split())[0]))

        if q1 == "" or q2 == "":
            return self.out


        sq1 = self.en_nlp(q1)
        sq2 = self.en_nlp(q2)

        if self.is_one_word(sq1, sq2):
            return self.out
        count = 0

        start_time = time.time()

        entsq1 = self.ent_nltk(q1)
        entsq2 = self.ent_nltk(q2)

        self.out['entities'][1] = entsq2
        self.out['entities'][0] = entsq1

        for ent in sq1.ents:
            if ent.text not in entsq1:
                # self.out['entities'][0].append([ent.label_, ent.text])
                self.out['entities'][0].append(ent.text)

        for ent in sq2.ents:
            if ent.text not in entsq2:
                # self.out['entities'][1].append((ent.label_, ent.text))
                self.out['entities'][1].append(ent.text)

        if self.out['entities'][0]:

            if self.out['entities'][1]:
                if(len(self.out['entities'][0])!= len(self.out['entities'][1])):
                    return self.out

                self.out['max_keywords'] += len(
                    set(self.out['entities'][0] + self.out['entities'][1]))


                for each in self.out['entities'][0]:
                    if(each in self.out['entities'][1]):
                        count += 1
                    else:
                        return self.out
            else:
                return self.out

        elif self.out['entities'][1]:
            return self.out
        

            

        elapsed_time = time.time() - start_time

        self.out['keras'] = self.keras.similar(q1, q2)

        self.out['sentiment'][0] = get_sentiment_values(q1)[1]['compound']
        self.out['sentiment'][1] = get_sentiment_values(q2)[1]['compound']
        self.out['sentiment'][2] = abs(
            self.out['sentiment'][0] - self.out['sentiment'][1])

        if (abs(self.out['sentiment'][0]) > 0.3 and abs(
                self.out['sentiment'][1]) > 0.3):
            if self.out['sentiment'][2] >= 0.6:
                return self.out

        start_time = time.time()
        self.out['class'][0] = self.classifier.classify_question(sq1)
        self.out['class'][1] = self.classifier.classify_question(sq2)

        self.out['f_class'] = (self.out['class'][0] == self.out['class'][1])

        self.out['keywords'][0], self.out['numbers'][0] = extract_features(sq1)
        self.out['keywords'][1], self.out['numbers'][1] = extract_features(sq2)

        self.out['max_keywords'] += len(
            set(self.out['keywords'][0] + self.out['keywords'][1]))

        if self.out['class'][0] > 0 and self.out['class'][1] > 0:
            self.out['max_keywords'] += 1

        for each in self.out['keywords'][0]:
            if each in self.out['keywords'][1]:
                if (each in q1_neg_list and each not in q2_neg_list) or (
                                each in q2_neg_list and each not in q1_neg_list):
                    self.out['max_keywords'] += 1
                else:
                    if(each in self.stopwords_en):
                        count += 0.30
                        #self.out['max_keywords'] -= 1
                    else:      
                        count+=1

        if self.out['numbers'][0]:
            self.out['max_keywords'] += 1

            if self.out['numbers'][1]:
                self.out['max_keywords'] += 1
                if self.out['numbers'][1] != self.out['numbers'][0]:
                    return self.out

        elif self.out['numbers'][1]:
            self.out['max_keywords'] += 1

        if self.out['class'][0] > 0 and self.out['class'][1] > 0:
            self.out['max_keywords'] += 1

            if self.out['f_class']:
                if self.out['max_keywords'] > 1:
                    count += 1
                else:
                    count += 0.35

        # keywords_s1= [x for x in keywords_s1 if x not in keywords_s2]
        # keywords_s3= [x for x in keywords_s2 if x not in keywords_s1]
        if self.out['max_keywords'] < 1:
            self.out['keywords_sim'] = 0
        else:
            self.out['keywords_sim'] = (count / self.out['max_keywords']) * 100
            self.out['sim_per'] = (self.out['keywords_sim']+self.out['keras'])/2.0
            #print(self.out['keywords_sim'],count,self.out['max_keywords'])

        '''
        k_value = []
        s_value = []

        k = 100.0
        s = 30.0
        k_step = 10.0
        s_step = 4.0

        self.out["sim_per"] = (self.out['keywords_sim'] + self.out['keras']) / 2

        for i in range(7):
            k -= k_step
            s += s_step
            k_value.append(k)
            s_value.append(s)
        '''
        s_value = [34.0, 40.0, 50.0, 55.0, 60.0, 60.0, 60.0]
        k_value = [90.0, 85.0, 80.0, 75.0, 70.0, 60.0, 30.0]

        if self.out['keras'] >= k_value[0]:
            if self.out['keywords_sim'] >= s_value[0]:
                self.out['sim'] = 1
                return self.out

        elif self.out['keras'] > k_value[1]:
            if self.out['keywords_sim'] >= s_value[1]:
                self.out['sim'] = 1
                return self.out

        elif self.out['keras'] > k_value[2]:
            if self.out['keywords_sim'] >= s_value[2]:
                self.out['sim'] = 1
                return self.out

        elif self.out['keras'] > k_value[3]:
            if self.out['keywords_sim'] >= s_value[3]:
                self.out['sim'] = 1
                return self.out
        elif self.out['keras'] > k_value[4]:
            if self.out['keywords_sim'] >= s_value[4]:
                self.out['sim'] = 1
                return self.out
        elif self.out['keras'] > k_value[5]:
            if self.out['keywords_sim'] >= s_value[5]:
                self.out['sim'] = 1
                return self.out
        elif self.out['keras'] > k_value[6]:
            if self.out['keywords_sim'] >= s_value[6]:
                self.out['sim'] = 1
                return self.out
        



        return self.out

    def similarr(self, text, questions=list()):

        answer, max_similarity = None, 0
        if not text or len(questions) == 0:
            return answer, max_similarity
        for question in questions:
            try:
                result = self.similar(text.lower(),
                                      question.get('question').lower())
            except:
                result = self.similar(text, question.get('question'))

            if result.get('sim') == 1:
                confidence = result.get('sim_per')
                if max_similarity <= confidence <= 100:
                    max_similarity = confidence
                    answer = question.get('id')
                    # print("round stop\n")
                if max_similarity >= 95:
                    break

        # print('[Stop]')
        return answer, max_similarity

    def get_suggestions(self, text=None, texts=list()):
        res = []
        s = []
        min_confidence = 45

        for each in texts:
            result = self.similar(text, each.get('question').lower())
            if result.get('sim') == 1:
                confidence = result.get('sim_per')
                if 100 >= confidence > min_confidence:
                    if each.get('rich_text'):
                        response = each.get('rich_text')
                    else:
                        flow = int(each.get('response').replace('flow-', ''))
                        flow = Flow.objects.filter(id=flow).values('id', 'name',
                                                                   'category__name')
                        if flow.exists():
                            response = [{'flow': flow}]
                        else:
                            response = None
                    if response:
                        res.append((confidence, each.get('id'), response,
                                    each.get('question')))
            s = sorted(res, key=operator.itemgetter(0), reverse=True)[:3]
        suggestions = []
        for e in s:
            if e[2]:
                messages = []
                for m in e[2]:
                    messages.append({'message': format_message(m)})
                suggestions.append({'confidence': e[0], 'id': e[1],
                                    'message': messages})
        return suggestions


"""
test code
"""
model = fmodel()
q1=""
q2=""
while(q1!="1"):
    q1=input(">")
    q2=input(">>")
    print(model.similar(q1,q2))
