import os,sys
import ROOT as rt

ploss = os.popen("cat baka2 | grep \"MAE-loss\"")
lloss = ploss.readlines()

data = []
for l in lloss:
    #print(l)
    try:
        x = float(l.strip().split("tensor(")[-1].split(",")[0])
        i = len(data)
        data.append((i,x))
    except:
        pass

smooth = []
npts = 100
for i in range(int(len(data)/npts)-1):
    x = 0.0
    for d in data[i*npts:(i+1)*npts]:
        x += d[1]/float(npts)
    smooth.append( x )


c = rt.TCanvas("c","c",2400,2000)
g = rt.TGraph(len(data))
for d in data:
    g.SetPoint( d[0], d[0], d[1] )
gs = rt.TGraph(len(smooth))
for i,s in enumerate(smooth):
    gs.SetPoint(i,i*npts,s)
gs.SetLineColor(rt.kRed)
g.Draw("ALP")
gs.Draw("LP")
c.Update()
print("[enter] to quit.")
input()
