from vlm import *
prob = VLM(ni=5,
		       nj=50,
		       chordRoot=1.0,
		       chordTip=1.0,
		       twistRoot=0.0,
		       twistTip=0.0,
		       span=5.0,
		       sweep=0.0,
		       Sref =5.0,
		       referencePoint=[0.0,0.0,0.0],
		       wingType=1,
		       alphaRange = [0.0,10.0])
prob.run()
