from PIL import Image
import os
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

human_list = glob("CASIA-WebFace/*")
horse_list = glob("Horse_47/*")

#Preprocessing the human face data.
for i in tqdm(human_list):
    for j in glob(human_list[0]+"/*"):
        img = Image.open(j)
        img = img.crop([0,0,256,256*0.4])#0.4 was clearly stated in the paper
        s = j.split("\\")
        folder = s[1]
        file = s[-1]
        folder_exist = os.path.exists("upper_half_human\\"+folder)
        if folder_exist:
            img.save("upper_half_human\\"+folder+"\\"+file)
        else:
            os.makedirs("upper_half_human\\"+folder)
            img.save("upper_half_human\\"+folder+"\\"+file)
