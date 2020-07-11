#!/usr/bin/python
#-*- coding: utf-8
import numpy as np
import collections
import pickle
import h5py
#linelens=8763
#stunum=379
ans = np.zeros((379,101,110,2))
Max=-1
with open("/media/zy/disk1/code/code/Graph-Models-Source-Code-master/gconvRNN/datasets/ptb_char/train.csv") as f:
    data=f.readlines()
    linelens=len(data)
    stus = sorted(list(set([int(data[i-1].replace('\n','')) for i in range(1,linelens+1) if i%3 == 1])))
    skills = set()
    tmp = [set(map(int, data[i - 1].replace('\n', '').split(","))) for i in range(1, linelens + 1) if i % 3 == 2]
    for x in tmp:
        for y in x:
            skills.add(y)
    skills = sorted(list(skills))

    for i in range(2,linelens,3):
        Num = collections.Counter()
        c = collections.Counter()
        stu = stus.index(int(data[i - 2].replace('\n', '')))
        this_skills = [skills.index(int(x)) for x in data[i - 1].replace('\n', '').split(',')]
        this_judges = [int(x) for x in data[i].replace('\n', '').split(',')]

        for j in range(len(this_skills)):
            Num[this_skills[j]] +=1
            if(this_judges[j]=='1' and c[int(this_skills[j])]<=100):
                ans[stu][c[int(this_skills[j])]][int(this_skills[j])-1][0]=1
                ans[stu][c[int(this_skills[j])]][int(this_skills[j])-1][1] = 0
            elif(this_judges[j]=='0' and c[int(this_skills[j])]<=100):
                ans[stu][c[int(this_skills[j])]][int(this_skills[j])-1][0] = 0
                ans[stu][c[int(this_skills[j])]][int(this_skills[j])-1][1] = 1
            c[int(this_skills[j])]+=1
        Max=max(Max,max(Num))

def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(tmp, f)
pklSave("/media/zy/disk1/code/code/Graph-Models-Source-Code-master/gconvRNN/datasets/ptb_char/bigf.pkl", ans)
    # ssadj = np.zeros((379, 101, 110, 2))
    # ssadj = tmp.copy()
    # cur=0
    # for i in range(102):
    #     if(tmp[:,:,:]==[0,0]):
    #         cur=
    #
    #
    # for i in range(102,-1):
    #
    #     # ans[:,i,:,:],ans[:,2-i-1,:,:]=ans[:,2-i-1,:,:],ans[:,i,:,:]
    #     ssadj[:,i,:,:]=tmp[:,i-(101-cur),:,:]
    #     ans[:, 101- i - 1, :, :]= ssadj[:,i,:,:]
    # print (ans)

#!/usr/bin/python
#-*- coding: utf-8
import collections
import operator

import numpy as np
import pickle
import h5py
#skillnum=8763
#stunum=379
#单个知识点max做题记录290
#知识点共110
def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
lapalace = np.zeros((379, 110))
with open("/media/zy/disk1/code/code/Graph-Models-Source-Code-master/gconvRNN/datasets/ptb_char/train.csv") as f:
    data=f.readlines()
    linelens=len(data)
    cnt=[]
    skill=[]
    Sum={}
    renum=[]
    Max=-1

    stus = sorted(list(set([int(data[i-1].replace('\n','')) for i in range(1,linelens+1) if i%3 == 1])))
    skills = set()
    tmp = [set(map(int,data[i-1].replace('\n','').split(","))) for i in range(1,linelens+1) if i%3==2]
    for x in tmp:
        for y in x:
            skills.add(y)
    skills = sorted(list(skills))
    lapalace = np.zeros((len(stus), len(skills), 2))
    for i in range(2,linelens+1,3):
        if i%3 == 2: #第二行
            stu = stus.index(int(data[i-2].replace('\n','')))
            this_skills = [skills.index(int(x)) for x in data[i-1].replace('\n','').split(',')]
            this_judges = [int(x) for x in data[i].replace('\n','').split(',')]
            for j in range(len(this_judges)):
                lapalace[stu][this_skills[j]][this_judges[j]] += 1
            # a=data[i-1].split("\n")
            # Sum=collections.Counter(a[0].split(","))
            # b=Sum.most_common(1)
            # x=b[0]
            # c=x[1]
            # renum.append(c)
            # s=data[i-1].split(",")
            # skillset=list(set(map(int,s)))#取不同知识点
            # skill.append(skillset)
            # for j in range(len(skillset)):
            #     for k in range(j,len(skillset)):
            #         ans[skillset[j]-1][skillset[k]-1]+=1
            #         ans[skillset[k]-1][skillset[j]-1] += 1
    # stu=set(map(int,stus))
    # slen=len(stu)
    # print ("len(stu)", slen)
    # print("renum",max(renum))
    # skill=reduce(operator.add, skill)
    # print(skill)
    # print(len(set(skill)))
    # print(ans)

    #学生*知识点*2
    pklSave("/media/zy/disk1/code/code/Graph-Models-Source-Code-master/gconvRNN/datasets/ptb_char/adj.pkl", lapalace)






