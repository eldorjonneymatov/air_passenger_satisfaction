# Kerakli modullarni import qilish va ba`zi
# vizualizatsiya parametrlarini o`rnatish
import io
import os
import gc
import pandas as pd
from PIL import Image
from shutil import rmtree
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('png')
plt.rcParams['figure.figsize'] = [12, 12]

# Baholi, kategoriyali ustun nomlarini belgilash
rat_cols = ['Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink',
            'Online boarding', 'Seat comfort', 'Inflight entertainment',
           'On-board service', 'Leg room service', 'Baggage handling',
           'Checkin service', 'Inflight service', 'Cleanliness']

cat_cols = ['Customer Type', 'Type of Travel', 'Class']

# Baholi ustunlarni tasvirlash uchun parametrlarni o`rnatish
r_colors = [
    plt.cm.get_cmap('Set2_r', 6), plt.cm.get_cmap('gist_heat', 6), plt.cm.get_cmap('tab20b_r', 6),
    plt.cm.get_cmap('Purples', 6), plt.cm.get_cmap('BuPu', 6), plt.cm.get_cmap('Wistia', 6),
    plt.cm.get_cmap('hot', 6), plt.cm.get_cmap('cool', 6), plt.cm.get_cmap('autumn', 6),
    plt.cm.get_cmap('summer', 6), plt.cm.get_cmap('spring', 6), plt.cm.get_cmap('bone', 6),
    plt.cm.get_cmap('winter', 6), plt.cm.get_cmap('PuRd')
]

# Kategoriyali ustunlarni tasvirlash uchun parametrlarni o`rnatish
marker_colors = {
    'Customer Type': '#00FFFF',
    'Class': '#8B8878',
    'Type of Travel': '#228B22'
}

marker_symbols = {
    'Customer Type': ['1', 'p'],
    'Type of Travel': ['P', '*'],
    'Class': ['H', 'd', 6]
}

marker_positions = {
    'Customer Type': [10, 60],
    'Type of Travel': [25, 60],
    'Class': [40, 60]
}

# Tasvirlarni saqlash uchun foydalaniladigan papka yaratuvchi funksiya
def make_folder(folder_path):
    path = folder_path
    if os.path.exists(path):
        rmtree(path, ignore_errors=True)
    os.mkdir(path)

# Baholi ustunlarni chizish
def rat_cols_draw(ob, ax):
    x = y = val = None
    for i in range(len(rat_cols)):
        x = 4 * i
        val = int(ob[rat_cols[i]])
        for j in range(val):
            y = 8 * j
            ax.add_patch(Rectangle((x, y), 3.5, 7, facecolor=r_colors[i](j)))
    del (ob, ax, x, y, val)

# Kategoriyali ustunlarni chizish
def cat_cols_draw(ob, ax):
    val = m = mc = x = y = None
    for c in cat_cols:
        val = int(ob[c])
        m = marker_symbols[c][val - 1]
        mc = marker_colors[c]
        x, y = marker_positions[c]
        ax.scatter(x, y, marker=m, c=mc, s=5000)
    del (ob, ax)

# Yoshni tasvirlash
def age_draw(ob, ax):
    age = int(ob['Age'])
    cmap = plt.cm.get_cmap('Set3', age)
    ccount = 0
    rows = age // 30
    last_row = age - 30 * rows
    for r in range(rows):
        for i in range(30):
            ax.add_patch(Rectangle((64 + 2 * i, 7 * (r + 1)), 7, 0.4, angle=240, facecolor=cmap(ccount)))
            ccount += 1
    for i in range(last_row):
        ax.add_patch(Rectangle((64 + 2 * i, 7 * (rows + 1)), 7, 0.4, angle=240, facecolor=cmap(ccount)))
        ccount += 1
    del (age, cmap, ccount, rows, last_row, ob, ax)

# Parvoz masofasini tasvirlash
def distance_draw(ob, ax):
    cmap = plt.cm.get_cmap('Dark2', 300)
    ccount = 1
    distance = int(ob['Flight Distance'] / 2)
    rows = distance // 63
    last_row = distance - 63 * rows
    x = y = colors = None
    for r in range(rows):
        x = range(1, 127, 2)
        y = [127 - r * 1.2] * len(x)
        colors = range(ccount, ccount + 63)
        ax.scatter(x, y, c=cmap(colors))
        ccount += 63
    x = range(1, 1 + 2 * last_row, 2)
    y = [127 - rows * 1.2] * len(x)
    colors = range(ccount, ccount + last_row)
    ax.scatter(x, y, c=cmap(colors))
    del (ccount, ob, ax, x, y, colors)

# Barcha ustunlarni chizish uchun funksiya
def draw(ob, ax):
    rat_cols_draw(ob, ax)
    cat_cols_draw(ob, ax)
    age_draw(ob, ax)
    distance_draw(ob, ax)
    del (ob, ax)

# Umumlashtiruvchi funksiya
def train_image_generator(df):
    name0 = name1 = 1
    make_folder(f'{root}//images')
    make_folder(f'{root}//images//satisfaction')
    make_folder(f'{root}//images//dissatisfaction')
    ob = file_name = img_buf = fig = ax = im = None
    for idx in range(len(df)):
        if idx % 200 == 0:
            gc.collect()
        print(f"{idx + 1}/{len(df)}|{(idx + 1) / len(df) * 100:.0f}%", end='\r')
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        ob = df.iloc[idx]
        draw(ob, ax)
        plt.axis('off')
        img_buf = io.BytesIO()
        if df['satisfaction'].iloc[idx] == 0:
            file_name = f"{root}//images/dissatisfaction/dissatis_{name0}.png"
            name0 += 1
        else:
            file_name = f"{root}//images/satisfaction/satis_{name1}.png"
            name1 += 1
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        im.save(file_name)
        plt.close(fig)
        img_buf.close()
        im.close()
    del (df, img_buf, im, ob, fig, ax, name0, name1)

# Sinov to`plami uchun tasvirlar yaratuvchi funksiya
def test_image_generator(df):
    name = 1
    make_folder(f'{root}//test_images')
    ob = file_name = img_buf = fig = ax = im = None
    for idx in range(len(df)):
        if idx % 200 == 0:
            gc.collect()
        print(f"{idx+1}/{len(df)}|{(idx+1)/len(df)*100:.0f}%", end='\r')
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        ob = df.iloc[idx]
        draw(ob, ax)
        plt.axis('off')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        file_name = f"{root}//test_images//test_{name}.png"
        im.save(file_name)
        name += 1
        plt.close(fig)
        img_buf.close()
        im.close()
    del(df, ob, fig, ax, img_buf, im, name)

if __name__ == '__main__':
    root = "..."
    df = pd.read_csv(f"{root}\\prepared_to_img.csv")
    test_df = pd.read_csv(f"{root}\\prepared_test_to_img.csv")
    train_image_generator(df)
    test_image_generator(test_df)