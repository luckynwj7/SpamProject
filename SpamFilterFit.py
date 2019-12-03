from SpamFilterClass import SpamMailFilter

from sklearn.model_selection import train_test_split
import csv, codecs
import pandas as pd
import pickle



# 클래스 사용하기    
sFilter = SpamMailFilter()

# CSV 파일 열기
filePath = "./dataset/spam.csv"
fp = codecs.open(filePath, "r", "utf-8")
# 한 줄씩 읽어 들이기
reader = csv.reader(fp, delimiter=",", quotechar='"')

# pandas 데이터 프레임에 입력하기
df=pd.DataFrame({"text":[0], "category":[0]})
i=0
for cells in reader:
    df.loc[i]=(cells[1], cells[0])
    i+=1

# 데이터 나누기
train, test = train_test_split(df, test_size=0.2, random_state=123)

# 학습
for idx in train.index:
    sFilter.fit(train["text"][idx], train["category"][idx])

# 모델 생성
with open('./model/spam_mail_filter.model', 'wb') as file:
    pickle.dump(sFilter, file)
print("모델 생성 완료")

    
# 예측
ok=0
case=0
for idx in test.index:
    case+=1
    pre = sFilter.predict(test["text"][idx])
    if test["category"][idx]==pre:
        ok+=1
print("정확도 : ", ok/case)

