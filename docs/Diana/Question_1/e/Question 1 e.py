from vlm import *

span_list = [5] 
Sref_list = span_list
sweep_list = [0]
chord = 1.0

for i in range(len(span_list)):
    for j in range(len(sweep_list)):
        AR = span_list[i]/chord
        print('Span = %.2lf Sref= %.2lf Sweep= %.2lf AR = %2lf' % (span_list[i], Sref_list[i], sweep_list[j], AR))
        prob = VLM(ni=5,
		       nj=50,
		       chordRoot=1.0,
		       chordTip=1.0,
		       twistRoot=0.0,
		       twistTip=0.0,
		       span=span_list[i],
		       sweep=sweep_list[j],
		       Sref=Sref_list[i],
		       referencePoint=[0.0,0.0,0.0],
		       wingType=3, #aile elliptique
		       alphaRange = [2.5])
        prob.run()
        print('\n')


