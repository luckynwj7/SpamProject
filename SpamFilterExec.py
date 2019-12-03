import pickle

with open('./model/spam_mail_filter.model', 'rb') as file:
    spamFilterModel = pickle.load(file)

    
print("검사할 내용을 입력해주세요")
contents=input()
pre = spamFilterModel.predict(contents)
print("결과 =", pre)
