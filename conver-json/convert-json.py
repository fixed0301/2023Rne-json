import csv
import glob
import json
import numpy as np
import os
import pandas as pd
from dev.create_csv import csv
from dev.calculate import preprocess_2

#json을 dataframe 형식으로 변환, 결측값 적당히 제거하고 한 값만 빈건 이전프레임 값으로 대체

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip":8, "RHip": 9,
               "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
               "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
               "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe":23, "RHeel": 24}
actions = ['backward', 'lie', 'sit', 'slide', 'stand', 'swing', 'walk']

def from_json(path): #path로 json 저장해두면 불러와서 사용
    file = open(path, 'r', encoding='utf-8')
    return json.load(file)

def extract_openpose_anns(ann_json):
    def extract_keypoints(ann_json):
        X = []
        Y = []
        C = []
        id = 0
        while id < len(ann_json):
            if ann_json[id] != 0:
                X.append(ann_json[id]) #여기에 전처리 과정을 추가
                Y.append(ann_json[id + 1])
                C.append(ann_json[id + 2]) #confidence score
                id += 3
            else:
                X.append(None)
                Y.append(None)
                C.append(None)
                id += 3

        return np.array([X, Y, C])

    kp_pose = extract_keypoints(ann_json['people'][0]['pose_keypoints_2d']) #array

    pose = {}
    i = 0
    for key in BODY_PARTS.keys():
        name_x = f'{key}' + '_x'
        name_y = f'{key}' + '_y'
        pose[f'{name_x}'] = kp_pose[0][i]
        pose[f'{name_y}'] = kp_pose[1][i]
        i += 1
    return pose

def extract_pose_annotations(path): #folder path 안 json 파일꺼내서 필요한 pose 좌표 dict으로 정리
    path = os.path.join(path, '*')
    files = glob.glob(path)
    Y_raw = []
    for file in files:
        ann_json = from_json(file)
        ann = extract_openpose_anns(ann_json)
        # ann은 dictionary
        # 정규화한 좌표 추가
        Y_raw.append(ann)

    return Y_raw


df_list = []

json_action_folder = f'../landmark-json/slide'
for action_file in os.listdir(json_action_folder):
    file_path = json_action_folder+'/'+ action_file
    if file_path == json_action_folder + '/' + 'slide_3_000000000066_keypoints.json':
        #이상하게 이 파일만 error 뜸
        continue
    with open(file_path) as json_file:
        data = json.load(json_file)
        pose = extract_openpose_anns(data) #{'Nose_x': 951.139, 'Nose_y': 480.116,...
        temp_df = pd.DataFrame(pose, index=[0])
        df_list.append(temp_df)

df1 = pd.concat(df_list, ignore_index=True) #[418 rows x 36 columns]
'''
행마다 if (RKnee and LKnee is None) or (Neck_y is None) : delete row
그리고 탐지된 개수가 총 36가지 중 18개보다 작으면 제거
남은 결측값은 이전 프레임걸로, 처음 행은 뒤의 프레임걸로
'''
df1 = df1.dropna(subset=['RKnee_y', 'LKnee_y'], how='all')
df1 = df1.dropna(subset=['Neck_y']) # [418 rows x 36 columns]

df1.fillna(method='ffill', inplace=True)
df1.iloc[:, 0].fillna(method='bfill', inplace=True)
print(df1)

#print(df1.isnull().sum(axis = 0)) #null 개수 세보자

# print(final_preprocess)
csv(df1, 'slide_test_0919')



# with open(f'../landmark-csv/slide2.csv', 'a') as f: #하나의 csv 파일에 몇개나 저장할까.. backward에 대한 csv,
#             w = csv.writer(f)
#             with open('../landmark-json/slide/slide_1_000000000067_keypoints.json') as json_file:
#                 data = json.load(json_file)
#                 pose = extract_openpose_anns(data) #{'Nose_x': 951.139, 'Nose_y': 480.116,...
#                 print(pose)
# #convert to csv
# for action in actions:
#     json_action_folder = f'../landmark-json/{action}'
#     for file in os.listdir(json_action_folder):
#         file_path = json_action_folder + "/" + file
#         Y_raw = extract_pose_annotations(file_path) #json 폴더마다 꺼내기
#         print(Y_raw)
#         for landmarkSet in Y_raw: #landmarkSet written in dictionary
#             print(landmarkSet)
#             with open(f'../landmark-csv/{action}.csv', 'w') as f: #하나의 csv 파일에 몇개나 저장할까.. backward에 대한 csv,
#                 w = csv.writer(f)
#                 w.writerow(landmarkSet.keys())
#                 w.writerow(landmarkSet.values())
#                 f.close()


