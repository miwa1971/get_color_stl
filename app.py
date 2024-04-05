import streamlit as st
import pandas as pd
import numpy as np
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from streamlit.components.v1 import html
import io
import base64

def create_image_link():
    """ 
    【出典】
        https://blog.shikoan.com/streamlit-download/
    """
    img_array = cv2.imread('./107FTASK-orthophoto_zoom_hyst.jpg')
    # img = np.array(image)
    # img_array = np.zeros((256, 256, 3), np.uint8)
    # img_array[:, :, 0] = 255
    # img_array[:, :, 1] = np.arange(256, dtype=np.uint8)[None, :]
    # img_array[:, :, 2] = np.arange(256, dtype=np.uint8)[:, None]
    with io.BytesIO() as buf:
        with Image.fromarray(img_array) as img:
            img.save(buf, format="png")
        image_str = base64.b64encode(buf.getvalue()).decode()
        js_code = f"""<a href="data:png;base64,{image_str}" download="sample.png">download</a>"""
    return js_code

def Cnv_Gamma(img,gamma):
    """
    **********************************************************
      【ガンマ補正の公式】
       Y = 255(X/255)**(1/γ)
      【γの設定方法】
       ・γ>1の場合：画像が明るくなる
       ・γ<1の場合：画像が暗くなる
      【出典」
         https://di-acc2.com/programming/python/19009/
    **********************************************************
    """
    
    # ガンマ変換用の数値準備 
    # gamma     = 3.0                               # γ値を指定
    img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値
    
    # 公式適用
    for i in range(256):
        img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)
    
    # 読込画像をガンマ変換
    gamma_img = cv2.LUT(img,img2gamma)
    
    return gamma_img


def sidebar(img):
    #サイドバー内に、ヒストグラムを作成する
    # img = cv2.imread(path)
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
    hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
    
    hist_rgb = np.hstack([hist_r,hist_g,hist_b])
    df_rgb = pd.DataFrame(hist_rgb,columns=['r','g','b'])
    
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
    hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    
    hist_hsv = np.hstack([hist_h,hist_s,hist_v])
    df_hsv = pd.DataFrame(hist_hsv,columns=['h','s','v'])
    
    fig_type = st.sidebar.selectbox(
        "グラフ種別",
        ("RGB", "HSV", "Both")
    )
    
    if fig_type=="RGB":
        st.sidebar.line_chart(df_rgb)
    elif fig_type=="HSV":
        st.sidebar.line_chart(df_hsv)
    else:
        st.sidebar.line_chart(df_hsv)   
        st.sidebar.line_chart(df_rgb)
    

def make_pdf(img,hmin,hmax,smin,smax,vmin,vmax,gamma,max_rgb,max_hsv):
    #pdfファイルを作成する
    # 作図準備
    
    fig=plt.figure(figsize=(8.27,11.69),dpi=100)
    plt.subplots_adjust(hspace=0.4,left=0.5,bottom=0.07,top=0.92)   

    ax1= fig.add_axes([0.3,0.8,0.5,0.15])   
    # img = cv2.imread(path)
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
    hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
    ax1.plot(hist_r, color='r', label="r")
    ax1.plot(hist_g, color='g', label="g")
    ax1.plot(hist_b, color='b', label="b")
    # ax1.set_yscale("log", base=3)
    ax1.legend()
    ax1.set_title('RGB histogram')
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("No of pixcels")
    ax1.grid(axis="both",which="both")
    ax1.set_xlim(0,255)
    
    if max_rgb.isnumeric()==True:
        ax1.set_ylim(0,float(max_rgb))
    
    
    # ax1.set_ylim(0,1000)
    ax2= fig.add_axes([0.3,0.55,0.5,0.15])   
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
    hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    ax2.plot(hist_h, color='r', label="h")
    ax2.plot(hist_s, color='g', label="s")
    ax2.plot(hist_v, color='b', label="v")
    # ax2.set_yscale("log", base=3)
    ax2.legend()
    ax2.set_title('HSV histogram')
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("No of pixcels")
    ax2.grid(axis="both",which="both")
    ax2.set_xlim(0,255)
    
    if max_hsv.isnumeric()==True:
        ax2.set_ylim(0,float(max_hsv))
    
    #画像を入れる
    ax_img = fig.add_axes([0.25, 0.02, 0.5, 0.5])  
    # img = mpimg.imread(path)   #画像を読み込む
    # ax.text(1, 0.5, "hue_min" + str(hmin), fontsize="xx-large")
    ax_img.imshow(img) #画像をグラフに貼り付ける
    ax_img.axis('off') # 軸を非表示にする
    ax_img.set_frame_on(False) # 枠線を非表示に
    
    plt.savefig('./hyst.jpg')
    
    img_array = cv2.imread('./hyst.jpg')

    with io.BytesIO() as buf:
        with Image.fromarray(img_array) as img2:
            img2.save(buf, format="png")
        image_str = base64.b64encode(buf.getvalue()).decode()
        js_code = f"""<a href="data:png;base64,{image_str}" download="sample.png">download</a>"""
    return js_code

def main():
    st.title("画像編集テスト")
    
    import io 
    uploaded_file = st.file_uploader('Choose a image file',type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
    else:
        exit()
        
    #画像の表示領域の指定
    back_fig = st.radio(label='画像の表示',
                     options=('orig', 'h','s','v','hsv'),
                     index=0,
                     horizontal=True,
    )
    
    frm_wid = st.radio(label='画像の表示',
                     options=('all', 'part'),
                     index=0,
                     horizontal=True,
    )
    
    #抽出領域の表示領域の指定
    area_exp = st.radio(label='対象領域の表示',
                     options=('Black', 'line'),
                     index=1,
                     horizontal=True,
    )
    
    #HSVによる領域
    hmin = st.sidebar.slider('H_min', 0, 255, 1)
    hmax = st.sidebar.slider('H_max', 0, 255, 255)
    smin = st.sidebar.slider('s_min', 0, 255, 1)
    smax = st.sidebar.slider('s_max', 0, 255, 255)
    vmin = st.sidebar.slider('v_min', 0, 255, 1)
    vmax = st.sidebar.slider('v_max', 0, 255, 255)

    #画質調整
    #ガンマ変換 → スライドバーで調整
    gamma = st.sidebar.slider('ガンマ変換',min_value=0.1, max_value=5.0, step=0.1,value=1.0)
    img = Cnv_Gamma(img, gamma)
    
    # #コントラスト項目
    # alpha = st.sidebar.slider('α（コントラスト）',min_value=0.1, max_value=1.6, step=0.1,value=1.0)
    # #明るさ項目
    # beta = st.sidebar.slider('β（明るさ）',min_value=0, max_value=60, step=1,value=0)    

    # # 明るさ・コントラスト計算
    # img = cv2.convertScaleAbs(img,alpha = alpha,beta = beta)
    
    
    # HSV変換
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    
    #背景画像の選択
    if back_fig=='orig':
        img_back=img
    elif back_fig=='h':
        img_back = cv2.merge((img_hsv[:,:,0],img_hsv[:,:,0],img_hsv[:,:,0]))
    elif back_fig=='s':
        img_back = cv2.merge((img_hsv[:,:,1],img_hsv[:,:,1],img_hsv[:,:,1]))
    elif back_fig=='v':
        img_back = cv2.merge((img_hsv[:,:,2],img_hsv[:,:,2],img_hsv[:,:,2]))
    elif back_fig=='hsv':
        img_back = cv2.merge((img_hsv[:,:,0],img_hsv[:,:,1],img_hsv[:,:,2]))

    # マスク用array作成
    img_mask = cv2.inRange(img_hsv, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    # img_mask2 = cv2.inRange(img_hsv, np.array([10, 135, 90]), np.array([20, 165, 115]))
    
    # 輪郭抽出
    contours, hierarchy = \
                            cv2.findContours(
                            img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 小さい輪郭は誤検出として削除
    contours = list(filter(lambda x: cv2.contourArea(x) > 10, contours))
    
    # 輪郭の描画
    cv2.drawContours(img_back, contours, -1, color=(0, 255, 0), thickness=3)
    
    #マスクのインバースを作成
    img_mask_inv = cv2.bitwise_not(img_mask)

    # cv2.imshow('image', img_res)
    if frm_wid=='all':
        is_all=True
    else:
        is_all=False
    
    #画像の出力
    if area_exp == 'Black':       
        img_res = cv2.bitwise_and(img_back, img_back, mask=img_mask_inv)
    else:
        img_res = img_back
   
    st.image(
            img_res, caption='',
            use_column_width=is_all
            )   
        
    #SideBar表示    
    #この部分に、RGB/HSV図と、スライダーを配置する。
    sidebar(img)
    
    max_rgb = st.text_input('RGV_Y軸 max', '100')
    max_hsv = st.text_input('HSV_Y軸_max', '100')
    
    #ボタン操作でpdf出力
    # if st.button('pdfファイルに出力'):
        # st.write('ボタンがクリックされました！')    
        # make_pdf(img_res,hmin,hmax,smin,smax,vmin,vmax,gamma,max_rgb,max_hsv)
    
    download_button = st.button("Click to download")
    container = st.empty()
    if download_button:
        with container:
            html(make_pdf(img_res,hmin,hmax,smin,smax,vmin,vmax,gamma,max_rgb,max_hsv), height=50) 
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




  



