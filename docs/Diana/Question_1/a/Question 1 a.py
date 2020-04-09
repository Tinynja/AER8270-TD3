from vlm import *
prob = VLM(ni=5,
		       nj=50,
		       chordRoot=1.0,
		       chordTip=1.0,
		       twistRoot=0.0,
		       twistTip=0.0,
		       span=1E10,
		       sweep=0.0,
		       Sref =1E10,
		       referencePoint=[0.0,0.0,0.0],
		       wingType=1,
		       alphaRange = [0.0,2.5,5.0])
prob.run()


