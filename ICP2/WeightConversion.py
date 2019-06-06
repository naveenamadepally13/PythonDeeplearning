stdCount = int(input("Enter number of students = "))
stdWgtInLb =[]
for x in range(stdCount):
    stdWgtInLb.append(int(input("enter weigth for student :")))
print(stdWgtInLb);
stdWgtInKg = []
i=0
while i < stdWgtInLb.__len__():
    wt = int(stdWgtInLb[i]/2.205)
    wt =  "{: .2f}".format(wt)
    print(wt)
    stdWgtInKg.append(wt)
    i += 1
print(stdWgtInKg)
