account_Balanace = int(input("Enter the accoun balance: "))
withdrawal_Amount = int(input("Enter the withdrwal amount: "))


verified = input("Is the user verified True/False: ")

# Convert verification input to boolean
verified = verified == "True"



if verified  and account_Balanace >= withdrawal_Amount :
    print("Withdrawal successful")
else:
    print("transcation denied")