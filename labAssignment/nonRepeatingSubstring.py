inputString = 'pwwkew'
length = inputString.__len__()
d=[]
d1=[]
for c in inputString:
 flag = 0
 flag2 = 0
 if(d.__len__()==0):
   d.append(c)
 else:
   for x in d:
       if(x!=c):
        flag = 1
       if(x==c):
        flag2 = 1
 if(flag==1 and flag2==0):
   d.append(c)
 elif((flag==1 and flag2==1)):
     d.append(c)
     d1=d
     d=[]
 elif ((flag == 0 and flag2 == 1)):
     d.append(c)
     d1 = d
     d = []
print(d1)
print(d)