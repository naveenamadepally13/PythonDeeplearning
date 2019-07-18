data =[( 'John', ('Physics', 80)),
       ('Daniel', ('Science', 90)),
       ('John', ('Science', 95)),
       ('Mark', ('Maths', 100)),
       ('Daniel', ('History', 75)),
       ('Mark', ('Social', 95))]
dictionary = {}
for word in data:
    dictionary.setdefault( word[0], []).append(word[1])
print(dictionary)
