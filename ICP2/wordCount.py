file = open("input.txt", "r")
fileData = file.read()
fileData = fileData.split()
#print(fileData)
dictionary = {}
for word in fileData:
    word = word.casefold()
    if word not in dictionary:
        #d[key] = value
        dictionary[word] = 0
    dictionary[word] += 1
print(dictionary)

