from PIL import Image
from pathlib import Path
import face_recognition
import csv
import pickle


def predict(image_path, d, s, per, model_sav, model_csv, face_dir):
    '''
    写真に写っている顔を分析しモデルの中の一番特徴が近い名前を返す
    '''
    g_menber_list = []
    del_list = []
    return_member = []
    svc = load_model(model_sav)
    image = load_image(image_path)
    categories = load_categories(model_csv)
    face_locations = find_face_locations(image)
    result_list = predict_all_face(svc, face_locations, image)
    for index, result in enumerate(result_list):
        rate, member = max(zip(result, categories))
        print(f'{int(rate*100)}%{member}')
        g_menber_list.append(member)
        if rate > per/100:
            print(f"{per}%以上のため登録します")
            return_member.append(member)
        else:
            print(f"{per}%以下です登録しません")
    save_result_image(image, face_locations, g_menber_list, d, s, face_dir)
    return return_member


def load_model(model):
    modelpath = model
    model = None
    with modelpath.open('rb') as fd:
        model = pickle.load(fd)
    return model


def load_image(imagepath):
    return face_recognition.load_image_file(imagepath)


def load_categories(csvfile):
    csvpath = Path(csvfile)
    memberlist = []
    with csvpath.open('r') as fd:
        reader = csv.reader(fd)
        memberlist = next(reader)
    return memberlist


def find_face_locations(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations


def predict_all_face(svc, face_locations, image):
    cnv = face_recognition.face_encodings(image,
                                          known_face_locations=face_locations)
    result = svc.predict_proba(cnv)
    return result


def save_result_image(image, face_locations, g_menber_list, d, s, face_dir):
    '''
    ついでにモデル作る用顔データも取得
    '''
    for (index, location) in enumerate(face_locations):
        x1, y1, x2, y2 = location
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        y_start = min(y1, y2)
        y_end = max(y1, y2)
        facearray = image[x_start:x_end, y_start:y_end]
        pil_image = Image.fromarray(facearray)
        w, h = pil_image.size
        if w >90 and h >90:
            pil_image_name = f'{d}_face{index:0>2}.{s}'
            eng_name = g_menber_list[index]
            save_path = Path(face_dir, eng_name)
            if save_path.exists() is False:
                save_path.mkdir(parents=True)
            pil_image_path = Path(save_path, pil_image_name)
            pil_image.save(pil_image_path)
            print(g_menber_list[index])
