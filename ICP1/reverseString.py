enteredString = input("Enter input string : ")
#str = str.strip('py')[2:]
enteredString = enteredString[2:]
print('string after removing first two letters : ',enteredString)
print ('reverse string : ',enteredString[::-1]);
print("Now let's add 2 numbers")
number1 = int(input('enter first number : '))
number2 = int(input('enter second number : '))
print(number1,'+',number2,' = ',number1+number2)
print('End of the programm')
print('Replace programm')
enteredString= input("enter a string: ")
print(enteredString.replace('python', 'pythons'))