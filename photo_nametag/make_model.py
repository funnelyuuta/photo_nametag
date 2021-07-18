from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pathlib import Path
import face_recognition
import csv
import pickle
from tqdm import tqdm
import numpy as np


def make_model_main():
    '''
    モデル作り
    '''
    model_sav = Path('data','model','model.sav')
    model_csv = Path('data','model','model.csv')
    face_image = Path('data','model','make_model')
    pathlist = find_all_image_path(face_image)
    categories = list_categories(face_image)
    print(f'{categories}')
    svc, score = make_model(pathlist, categories)
    print(f'learn score: {score}%')
    save_model(svc, model_sav)
    save_categories(categories, model_csv)
    print('finish!!!')


def find_all_image_path(face_image):
    globstr = '*/*'
    allow_suffix = ['.jpg', '.jpeg', '.png', '.svg']
    _all_image_path = list(face_image.glob(globstr))
    all_image_path = []
    for path in _all_image_path:
        if not path.is_file():
            continue
        if path.suffix not in allow_suffix:
            continue
        all_image_path.append(path)
    return all_image_path


def list_categories(face_image):
    image_path_list = [p for p in face_image.iterdir() if p.is_dir()]
    categories = [p.name for p in image_path_list]
    return categories


def make_model(pathlist, categories):
    x = make_facenet_matrics(pathlist)
    y = make_answer_matrics(pathlist, categories)
    svc, score = _make_model_SVC(x, y)
    return svc, score


def save_model(svc, model_sav):
    parent = model_sav.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    with model_sav.open('wb') as fd:
        pickle.dump(svc, fd)


def save_categories(categories, model_csv):
    with model_csv.open('w') as fd:
        writer = csv.writer(fd)
        writer.writerow(categories)


def make_facenet_matrics(imagepathlist, feature_num=128):
    cnv_list = None
    for imagepath in tqdm(imagepathlist):
        cnv = load_face_image(imagepath, feature_num)
        if cnv_list is None:
            cnv_list = cnv[0]
        else:
            cnv_list = np.concatenate([cnv_list, cnv[0]])
    reshape_cnv_list = cnv_list.reshape(len(imagepathlist),
                                        feature_num)
    return reshape_cnv_list


def make_answer_matrics(pathlist, categories):
    ans_list = []
    for path in pathlist:
        dirname = path.parent.name
        index = categories.index(dirname)
        ans_list.append(index)
    return ans_list


def load_face_image(imagepath, feature_num=128):
    image = load_image(imagepath)
    face_location = (0, image.shape[1], image.shape[0], 0)
    cnv = face_recognition.face_encodings(image,
                                          known_face_locations=[face_location])
    return cnv


def load_image(imagepath):
    return face_recognition.load_image_file(imagepath)


def _make_model_SVC(x, y):
    train_test = train_test_split(x, y, random_state=0, train_size=0.8)
    xtrain, xtest, ytrain, ytest = train_test
    svc = SVC(probability=True)
    svc.fit(xtrain, ytrain)
    ypred = svc.predict(xtest)
    score = accuracy_score(ytest, ypred) * 100.0
    return svc, score
