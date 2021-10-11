'''
Descripttion: 
version: 
Author: yp
Date: 2021-06-27 22:48:49
LastEditors: yp
LastEditTime: 2021-10-09 21:50:25
'''
import os
from sphfile import SPHFile
from pydub import AudioSegment

parser = argparse.ArgumentParser(description = "DATAPRE")
parser.add_argument('--train_list',         type=str,   default=None,   help='voxceleb1 train set')

def get_all_type_file(file_dir,tp = '.txt'):

    '''
    获取指定文件夹及其子文件夹下的所有.tp音频路径
    需要导入的库：numpy,librosa
    :param file_dir: 文件夹地址（str）
    :return: 音频地址列表（list）
    '''
    _tp_paths = []
    for dir, sub_dir, file_base_name in os.walk(file_dir):
        for file in file_base_name:
            if file.endswith(tp) or file.endswith(tp):
                _tp_paths.append(os.path.join(dir, file))
    return _tp_paths

def voxcelebe_datapre(file_path):
    speak_dict = dict()
    nums = 0
    for dir, sub_dir,filename in os.walk(file_path):
        for file in filename:
            if file.endswith(".wav"):
                spk = dir.split("/")[-1]
                spk = "vox_" + spk
                if spk not in speak_dict.keys():
                    speak_dict[spk] = []
                speak_dict[spk].append(os.path.join(dir,file))
                nums += 1
            else:
                pass
    print("说话人数为： {}   总共样本数为： {}".format(len(speak_dict.keys()),nums))
    lines = list()
    for spk in speak_dict.keys():
        for wav in speak_dict[spk]:
            lines.append("{} {}\n".format(spk,wav))

    with open("./datameta/train_set.txt", "w", encoding = "utf-8") as fid:
        for line in lines:
            fid.write(line)

def Timit_datapre():
    file_path = ["/data/pineyang/ft_local/TRAIN","/data/pineyang/ft_local/TEST"]
    file_txt  = ["/data/pineyang/paper/datameta/timit_train.txt","/data/pineyang/paper/datameta/timit_test.txt"]
    result = [[],[]]
    for index,path in enumerate(file_path):
        with open(file_txt[index],"r",encoding="utf-8") as fid:
            lines = fid.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            spk,age,gender = line.split(" ")
            id_path = os.path.join(path,spk)
            if not os.path.exists(id_path):
                print(spk,id_path)
            else:
                all_wav = get_all_type_file(id_path,".wav")
                for wav in all_wav:
                    # sph = SPHFile(wav)
                    # sph.write_wav(filename = wav.replace(".WAV",".wav"))
                    result[index].append("{} {} {:.3f} {}\n".format(spk,wav,float(age),gender))
                    # os.remove(wav)
    with open("./datameta/timit_train_age.txt","w",encoding="utf-8") as fid:
        for line in result[0]:
            fid.write(line)
    with open("./datameta/timit_test_age.txt","w",encoding="utf-8") as fid:
        for line in result[1]:
            fid.write(line)
    return True
         

if __name__ == "__main__":
    args = parser.parse_args()
    Timit_datapre(args.train_list)




