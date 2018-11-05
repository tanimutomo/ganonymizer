import subprocess

cmd = ['python', '-m', 'ganonymizer.main']
runcmd = subprocess.check_call(cmd)
print (runcmd)
