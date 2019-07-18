# file = open("input.txt", "r")
# fileData = file.read()
# fileData = fileData.split()
# #print(fileData)
# dictionary = {}
# for word in fileData:
#     word = word.casefold()
#     if word not in dictionary:
#         #d[key] = value
#         dictionary[word] = 0
#     dictionary[word] += 1
# print(dictionary)

# a=['three'];
# d={"three":3,"two":2}
#
# for x in a:
#  if x in d:
#    print(d[x])

a=[0,1,2,3]
for a[-1] in a:
    print(a[-1])

d={"three":3,"two":2}
del d['three']
print(d)

d={'e':97,'a':96,'b':98}
for _ in sorted(d):
    print(d[_])
    print(d)
print('abcefd'.replace('cd','12'))