import face_recognition
from pathlib import Path
from PIL import Image,UnidentifiedImageError

def face_cut_main():
    '''
    写真から人の顔を切り抜く
    '''
    imgdirpath = Path('data','face_cut','in_image')
    savedir = Path('data','face_cut','out_image')
    dirnames = [d for d in imgdirpath.iterdir() if d.is_dir()]
    print(dirnames)
    for dirname in dirnames:
        dirpath = Path(dirname)
        name = dirpath.name
        filepaths = [p for p in dirpath.iterdir()]
        for filepath in filepaths:
            save_face(filepath, savedir, name)

def save_face(filepath, savedir, name):
    imagefilename = filepath.stem
    savedirpath = Path(savedir, name)
    if not savedirpath.exists():
        savedirpath.mkdir(parents=True)
    try:
        image = face_recognition.load_image_file(filepath)
    except UnidentifiedImageError as error:
        print(error)
        print('顔が見つかりません')
        return None
    locations = face_recognition.face_locations(image)
    #顔の範囲を取得
    for (index, location) in enumerate(locations):
        x1, y1, x2, y2 = location
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        y_start = min(y1, y2)
        y_end = max(y1, y2)
        facearray = image[x_start:x_end, y_start:y_end]
        pil_image = Image.fromarray(facearray)
        pil_image_name = f'{imagefilename}_face{index:0>2}{filepath.suffix}'
        pil_image_path = Path(savedir, name, pil_image_name)
        print(pil_image_path)
        pil_image.save(pil_image_path)

