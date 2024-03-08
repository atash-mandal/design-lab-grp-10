from dl_model import model, extract_mfcc
import numpy as np


path_1 = "C://Users//atash//Downloads//grp_10//New folder//Atash Audios"
path_2 = "C://Users//atash//Downloads//grp_10//New folder//Sourish_Audios"
path_3 = "C://Users//atash//Downloads//grp_10//New folder//Sumajit_audios"
path_4 = "C://Users//atash//Downloads//grp_10//New folder//Kowshik_audios"
pred = []
pred.append(extract_mfcc("C://Users//atash//Downloads//grp_10//Recording (22).wav"))
pred = np.array(pred)

print(model.predict(pred))