import os,sys
tot = 0.0

pawk = os.popen("ps -u$USER -o %cpu,rss,args | awk '{ print $2 }'")

lawk = pawk.readlines()

for l in lawk:
    try:
        x = float(l.strip())
        #print(x)
        tot += x
    except:
        print("not a num: ",l.strip())
        continue

print("total is: ",tot/1e6," GB")
