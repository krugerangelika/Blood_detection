import os
import shutil
import numpy as np

execution_path = os.getcwd()
# podanie ścieżki bazowej do katalogu, który zawiera obrazy do klasyfikacji
base_dir = os.path.join(execution_path, 'dataset')

# podanie nazwy klas, stałe:
BLOOD_CLASS_1 = 'basophil'
BLOOD_CLASS_2 = 'eosinophil'
BLOOD_CLASS_3 = 'erythroblast'
BLOOD_CLASS_4 = 'lymphocyte'
BLOOD_CLASS_5 = 'neutrophil'
BLOOD_CLASS_6 = 'monocyte'



# wskaźnik mówiący nam ile chcemy zostawić danych w zbiorze treningowym
TRAIN_BLOOD_RATIO = 0.6
# wskaźnik mówiący nam ile chcemy zostawić danych w zbiorze walidacyjnym
VALID_BLOOD_RATIO = 0.3
# wskazanie katalogu z naszymi danymi
DATA_BLOOD_DIR = r'images'

# wskazanie liczby obrazów w danej klasie
raw_no_of_files = {}
classes = [BLOOD_CLASS_1, BLOOD_CLASS_2] #BLOOD_CLASS_3, BLOOD_CLASS_4, BLOOD_CLASS_5, BLOOD_CLASS_6, BLOOD_CLASS_7, BLOOD_CLASS_8

number_of_samples = [(dir, len(os.listdir(os.path.join(base_dir)))) for dir in classes]

# jeśli nie istnieje katalog images to go stworzymy
if not os.path.exists(DATA_BLOOD_DIR): os.mkdir(DATA_BLOOD_DIR)

# Katalogi do zbiorów: train, valid, test
train_blood_dir = os.path.join(DATA_BLOOD_DIR, 'train_blood')
valid_blood_dir = os.path.join(DATA_BLOOD_DIR, 'valid_blood')
test_blood_dir = os.path.join(DATA_BLOOD_DIR, 'test_blood')

# CLASS_1
train_class_1_dir = os.path.join(train_blood_dir, BLOOD_CLASS_1)
valid_class_1_dir = os.path.join(valid_blood_dir, BLOOD_CLASS_1)
test_class_1_dir = os.path.join(test_blood_dir, BLOOD_CLASS_1)

# CLASS_2
train_class_2_dir = os.path.join(train_blood_dir, BLOOD_CLASS_2)
valid_class_2_dir = os.path.join(valid_blood_dir, BLOOD_CLASS_2)
test_class_2_dir = os.path.join(test_blood_dir, BLOOD_CLASS_2)

#CLASS_3
train_class_3_dir = os.path.join(train_blood_dir, BLOOD_CLASS_3)
valid_class_3_dir = os.path.join(valid_blood_dir, BLOOD_CLASS_3)
test_class_3_dir = os.path.join(test_blood_dir, BLOOD_CLASS_3)

# CLASS_4
train_class_4_dir = os.path.join(train_blood_dir, BLOOD_CLASS_4)
valid_class_4_dir = os.path.join(valid_blood_dir, BLOOD_CLASS_4)
test_class_4_dir = os.path.join(test_blood_dir, BLOOD_CLASS_4)

# CLASS_5
train_class_5_dir = os.path.join(train_blood_dir, BLOOD_CLASS_5)
valid_class_5_dir = os.path.join(valid_blood_dir, BLOOD_CLASS_5)
test_class_5_dir = os.path.join(test_blood_dir, BLOOD_CLASS_5)

# CLASS_6
train_class_6_dir = os.path.join(train_blood_dir, BLOOD_CLASS_6)
valid_class_6_dir = os.path.join(valid_blood_dir, BLOOD_CLASS_6)
test_class_6_dir = os.path.join(test_blood_dir, BLOOD_CLASS_6)



for dir in (train_blood_dir, valid_blood_dir, test_blood_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_1_dir, valid_class_1_dir, test_class_1_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_2_dir, valid_class_2_dir, test_class_2_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_3_dir, valid_class_3_dir, test_class_3_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_4_dir, valid_class_4_dir, test_class_4_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_5_dir, valid_class_5_dir, test_class_5_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_6_dir, valid_class_6_dir, test_class_6_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)



print('[INFO] Load file names ..')
class_1_names = os.listdir(os.path.join(base_dir, BLOOD_CLASS_1))
class_2_names = os.listdir(os.path.join(base_dir, BLOOD_CLASS_2))
class_3_names = os.listdir(os.path.join(base_dir, BLOOD_CLASS_3))
class_4_names = os.listdir(os.path.join(base_dir, BLOOD_CLASS_4))
class_5_names = os.listdir(os.path.join(base_dir, BLOOD_CLASS_5))
class_6_names = os.listdir(os.path.join(base_dir, BLOOD_CLASS_6))


print('[INFO] Validation of the correctness of names .. ')

class_1_names = [fname for fname in class_1_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
class_2_names = [fname for fname in class_2_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
class_3_names = [fname for fname in class_3_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
class_4_names = [fname for fname in class_4_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
class_5_names = [fname for fname in class_5_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
class_6_names = [fname for fname in class_6_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]

# przetasowanie nazw plików
np.random.shuffle(class_1_names)
np.random.shuffle(class_2_names)
np.random.shuffle(class_3_names)
np.random.shuffle(class_4_names)
np.random.shuffle(class_5_names)
np.random.shuffle(class_6_names)


print(f'[INFO] Number of images in the dataset {BLOOD_CLASS_1}: {len(class_1_names)}')
print(f'[INFO] Number of images in the dataset {BLOOD_CLASS_2}: {len(class_2_names)}')
print(f'[INFO] Number of images in the dataset {BLOOD_CLASS_3}: {len(class_3_names)}')
print(f'[INFO] Number of images in the dataset {BLOOD_CLASS_4}: {len(class_4_names)}')
print(f'[INFO] Number of images in the dataset {BLOOD_CLASS_5}: {len(class_5_names)}')
print(f'[INFO] Number of images in the dataset {BLOOD_CLASS_6}: {len(class_6_names)}')


train_index_class_1 = int(TRAIN_BLOOD_RATIO * len(class_1_names))
valid_index_class_1 = train_index_class_1 + int(VALID_BLOOD_RATIO * len(class_1_names))

train_index_class_2 = int(TRAIN_BLOOD_RATIO * len(class_2_names))
valid_index_class_2 = train_index_class_2 + int(VALID_BLOOD_RATIO * len(class_2_names))

train_index_class_3 = int(TRAIN_BLOOD_RATIO * len(class_3_names))
valid_index_class_3 = train_index_class_3 + int(VALID_BLOOD_RATIO * len(class_3_names))

train_index_class_4 = int(TRAIN_BLOOD_RATIO * len(class_4_names))
valid_index_class_4 = train_index_class_4 + int(VALID_BLOOD_RATIO * len(class_4_names))

train_index_class_5 = int(TRAIN_BLOOD_RATIO * len(class_5_names))
valid_index_class_5 = train_index_class_5 + int(VALID_BLOOD_RATIO * len(class_5_names))

train_index_class_6 = int(TRAIN_BLOOD_RATIO * len(class_6_names))
valid_index_class_6 = train_index_class_6 + int(VALID_BLOOD_RATIO * len(class_6_names))



print('[INFO] Copying files to target directories..')
for i, fname in enumerate(class_1_names):
    if i <= train_index_class_1:
        src = os.path.join(base_dir, BLOOD_CLASS_1, fname)
        dst = os.path.join(train_class_1_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_1 < i <= valid_index_class_1:
        src = os.path.join(base_dir, BLOOD_CLASS_1, fname)
        dst = os.path.join(valid_class_1_dir, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_1 < i <= len(class_1_names):
        src = os.path.join(base_dir, BLOOD_CLASS_1, fname)
        dst = os.path.join(test_class_1_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(class_2_names):
    if i <= train_index_class_2:
        src = os.path.join(base_dir, BLOOD_CLASS_2, fname)
        dst = os.path.join(train_class_2_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_2 < i <= valid_index_class_2:
        src = os.path.join(base_dir, BLOOD_CLASS_2, fname)
        dst = os.path.join(valid_class_2_dir, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_2 < i <= len(class_2_names):
        src = os.path.join(base_dir, BLOOD_CLASS_2, fname)
        dst = os.path.join(test_class_2_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(class_3_names):
    if i <= train_index_class_3:
        src = os.path.join(base_dir, BLOOD_CLASS_3, fname)
        dst = os.path.join(train_class_3_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_3 < i <= valid_index_class_3:
        src = os.path.join(base_dir, BLOOD_CLASS_3, fname)
        dst = os.path.join(valid_class_3_dir, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_3 < i <= len(class_3_names):
        src = os.path.join(base_dir, BLOOD_CLASS_3, fname)
        dst = os.path.join(test_class_3_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(class_4_names):
    if i <= train_index_class_4:
        src = os.path.join(base_dir, BLOOD_CLASS_4, fname)
        dst = os.path.join(train_class_4_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_4 < i <= valid_index_class_4:
        src = os.path.join(base_dir, BLOOD_CLASS_4, fname)
        dst = os.path.join(valid_class_4_dir, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_4 < i <= len(class_4_names):
        src = os.path.join(base_dir, BLOOD_CLASS_4, fname)
        dst = os.path.join(test_class_4_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(class_5_names):
    if i <= train_index_class_5:
        src = os.path.join(base_dir, BLOOD_CLASS_5, fname)
        dst = os.path.join(train_class_5_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_5 < i <= valid_index_class_5:
        src = os.path.join(base_dir, BLOOD_CLASS_5, fname)
        dst = os.path.join(valid_class_5_dir, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_5 < i <= len(class_5_names):
        src = os.path.join(base_dir, BLOOD_CLASS_5, fname)
        dst = os.path.join(test_class_5_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(class_6_names):
    if i <= train_index_class_6:
        src = os.path.join(base_dir, BLOOD_CLASS_6, fname)
        dst = os.path.join(train_class_6_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_6 < i <= valid_index_class_6:
        src = os.path.join(base_dir, BLOOD_CLASS_6, fname)
        dst = os.path.join(valid_class_6_dir, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_6 < i <= len(class_6_names):
        src = os.path.join(base_dir, BLOOD_CLASS_6, fname)
        dst = os.path.join(test_class_6_dir, fname)
        shutil.copyfile(src, dst)



print(f'[INFO] Number of images of the {BLOOD_CLASS_1} class in the training dataset: {len(os.listdir(train_class_1_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_1} class the validation dataset: {len(os.listdir(valid_class_1_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_1} class the test dataset: {len(os.listdir(test_class_1_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_2} class in the training dataset: {len(os.listdir(train_class_2_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_2} class the validation dataset: {len(os.listdir(valid_class_2_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_2} class the test dataset: {len(os.listdir(test_class_2_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_3} class in the training dataset: {len(os.listdir(train_class_3_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_3} class the validation dataset: {len(os.listdir(valid_class_3_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_3} class the test dataset: {len(os.listdir(test_class_3_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_4} class in the training dataset: {len(os.listdir(train_class_4_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_4} class the validation dataset: {len(os.listdir(valid_class_4_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_4} class the test dataset: {len(os.listdir(test_class_4_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_5} class in the training dataset: {len(os.listdir(train_class_5_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_5} class the validation dataset: {len(os.listdir(valid_class_5_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_5} class the test dataset: {len(os.listdir(test_class_5_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_6} class in the training dataset: {len(os.listdir(train_class_6_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_6} class the validation dataset: {len(os.listdir(valid_class_6_dir))}')
print(f'[INFO] Number of images of the {BLOOD_CLASS_6} class the test dataset: {len(os.listdir(test_class_6_dir))}')
