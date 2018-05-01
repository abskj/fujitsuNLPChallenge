import json
# cmd "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3

#reading the corpus

with open('.\..\datasets\selqa-evaluater\SelQA-ass-train.json') as fp:
    s=fp.read()
arr=s.split('\n')
print(arr[0])
for i in range( len(arr)):
    try:
        obj=json.loads(arr[i])
        print(i)
        print(obj["question"]+'\n')
        print("Ans: "+obj["sentences"][obj["candidates"][0]])
        for sentence in obj["sentences"]:
            print(sentence)
        pass
    except ValueError:
        pass
   