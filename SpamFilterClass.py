import math, sys
from konlpy.tag import Okt
import pickle



class SpamMailFilter:
    
    #객체초기화
    def __init__(self):
        self.words = set() # 출현한 단어 기록(중복 없음)
        self.wordEmerge = {} # 카테고리별 단어의 출현 횟수 기록 ex) (Ham - hi:2 hello:3)
        self.categoryData = {} # 데이터셋에 카테고리별로 데이터 갯수 ex) {'ham': 4825, 'spam': 747}
    
    
    # 자연어 형태소 분석
    def split(self, text):
        results = []
        okt=Okt()
        # 단어의 기본형 사용
        malist = okt.pos(text, norm=True, stem=True)
        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외 
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                results.append(word[0])
        return results
    
    
    # 카테고리별 단어 추가 및 단어 출현 횟수 계산
    def incWord(self, word, category):
        if not category in self.wordEmerge:
            self.wordEmerge[category] = {}
        if not word in self.wordEmerge[category]:
            self.wordEmerge[category][word] = 0
        self.wordEmerge[category][word] += 1
        self.words.add(word)
        
    def incCategory(self, category):
        # 카테고리별 데이터 갯수 계산하기
        if not category in self.categoryData:
            self.categoryData[category] = 0
        self.categoryData[category] += 1
    
    # 텍스트 학습
    def fit(self, text, category):
        wordList = self.split(text)
        for word in wordList:
            self.incWord(word, category)
        self.incCategory(category)
    
    # 단어 리스트에 점수 매기기
    def score(self, words, category):
        #확률을 곱셈하여 계산. 스코어를 로그 형태로 바꾸어 계산한다
        score = math.log(self.categoryProportion(category))
        for word in words:
            score += math.log(self.naiveBayes(word, category))
        return score
    
    # 예측
    def predict(self, text):
        bestCategory = None
        maxScore = -sys.maxsize
        words = self.split(text)
        scoreList = []
        #스코어를 계산하고 더 높은 스코어를 베스트로 선정
        for category in self.categoryData.keys():
            score = self.score(words, category)
            scoreList.append((category, score))
            if score > maxScore:
                maxScore = score
                bestCategory = category
        return bestCategory
    
    
    # 카테고리 내 단어의 출현 횟수 받아오기
    def getWordCount(self, word, category):
        if word in self.wordEmerge[category]:
            return self.wordEmerge[category][word]
        else:
            return 0
    

    # 카테고리별 등장한 단어 비율 계산
    def categoryProportion(self, category):
        sum_categories = sum(self.categoryData.values()) #훈련될 데이터 총 개수
        category_v = self.categoryData[category] #훈련될 데이터 카테고리별 총 개수
        return category_v / sum_categories # 훈련될 데이터 카테고리별 비율
    
        
    # 나이브 베이즈 분류 활용(카테고리 내부의 단어 출현 비율 계산)
    def naiveBayes(self, word, category):
        n = self.getWordCount(word, category) + 1 #예측할 문장 기준 각 카테고리 별 등장한 단어별 횟수 + 1
        d = sum(self.wordEmerge[category].values()) + len(self.words) #카테고리 별 등장한 총 단어수 + 총 단어의 수
        

        return n / d
