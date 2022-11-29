#Ask for password lenght
#Generate password

import string
import secrets

letters = string.ascii_letters
digits = string.digits
special = string.punctuation

alphabet = letters + digits + special

def generate_password():
    password_lenght=int(input('How long would you like your password to be? '))

    password = ''

    for i in range(password_lenght):
        password += ''.join(secrets.choice(alphabet))

    print(password)

generate_password()