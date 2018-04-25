import json
# cmd "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
with open('.\..\datasets\selqa-evaluater\SelQA-ass-train.json') as fp:
    s=fp.read()
arr=s.split('\n')
print(arr[0])
for i in range( len(arr)):
    try:
        obj=json.loads(arr[i])
        print(i)
        print(obj["question"]+'\n')
        print(obj["sentences"][obj["candidates"][0]])
        pass
    except ValueError:
        pass
   