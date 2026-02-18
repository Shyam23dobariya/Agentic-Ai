file = open("D:\IIT\Agentic-Ai\File\Shyam.txt","r")
print(file.read())
file.close()

# open file using with statement

with  open("D:\IIT\Agentic-Ai\File\Shyam.txt","r") as fly:
    content = fly.read()
    print(content)
