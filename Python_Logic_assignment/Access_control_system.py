user_age = int(input("Hello user ! can we get your age : "))
user_ID = bool(input("Just provide your Identicard: "))

if user_age >=18 and user_ID == True or user_ID == "true":
    print("entry allowed")
else:
    print("oops! may be your age is under 18 or you don't have valid ID NOT ALLOWED")
