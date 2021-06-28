import pandas as pd
import numpy as np

df=pd.read_csv('example.csv').fillna(value=" ")
index = df.index
num = len(index)
print(num)

#____________________________________________________#
list = []
for line in df['Symptoms']:
        list.append(line)

#__________list of diseases_________________________________________#
list1 = []
for line in df['Disease']:
       list1.append(line)
list2=','.join(list).split(',\n')


list_of_symptoms=','.join(list).split(',\n')
list_of_symptoms=sorted((set(list_of_symptoms)))
len1=len(list_of_symptoms)




ignore=[]
ignore.append(' ')
count=[]
for symp in df['Symptoms']:
        a=symp.split(',\n')
        if(a!=ignore):
                count.append(len(a))
        else:
                count.append(0)
        if len(count) == num:
                break
#____________________________________________________#

arr=[[0]* len1]*num
arr=np.array(arr)


print(count)

k=0
init=0
for j in count:
        final = j+init
        for i in range(init, final):
                if list2[i] in list_of_symptoms:
                        index = list_of_symptoms.index(list2[i])
                        arr[k][index]=1
                init=j
        k+=1

#____________________________________________________#
for i in range(0,num):
        print(list1[i],arr[i])
symptoms_df = pd.DataFrame(arr, columns=[list_of_symptoms], index=list1)
print(symptoms_df)


#symptoms_df.to_csv("symptoms_df.csv", sep='             ', encoding='utf-8',float_format='%10s')