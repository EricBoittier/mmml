from pycharmm import *


Echo = script_factory('echo hello')
my_echo = Echo()
my_echo.run()

# expected output:
# CHARMM>     echo hello
#HELLO

NewEcho = script_factory('echo', ['msg'])
my_echo = NewEcho(msg='help me')
my_echo.run()

# Expected output
# CHARMM>     echo msg help me
#MSG HELP ME

my_echo = NewEcho(msg='"help me"')
my_echo.run()

# Expected output
# CHARMM>     echo msg "help me"
#MSG "help me"
