# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:24:07 2019

@author: fancheyu
"""

#导入模块
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image,ImageDraw
import face_recognition as fr
from face_recognition.face_recognition_cli import image_files_in_folder

#函数定义
#训练
def train(train_dir,model_save_path='trained_knn_madel.clf',n_neighbors=3,knn_algo='ball_tree'):
    """
    训练一个KNN分类器
    :param train_dir
    :param model_save_path
    :param n_neighbors 
    :param knn_algo
    return：KNN分类器
    """
    
    #初始化训练集
    X = []
    y = []
    
    #遍历训练集中的每个人
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue 
    #加载图片
        for img_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            image = fr.load_image_file(img_path)
            boxes = fr.face_locations(image)
            print("{}".format(img_path))
            #对每张照片编码
            X.append(fr.face_encodings(image,known_face_locations=boxes)[0])
            y.append(class_dir)
        
    #确定k
    if n_neighbors is None:
        n_neighbors = 3 
        
    #训练出分类器
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X,y)
    
    #保存分类器
    if model_save_path is not None:
        with open(model_save_path,'wb') as f:
            pickle.dump(knn_clf,f)
    #返回分类器
    return  knn_clf
        
#预测
def predict(X_img_path,knn_clf=None,model_path=None,distance_threshold = 0.45):
    """
    利用KNN分离器识别
    :return :[人名，边界，...]
    """
    
    if knn_clf is None and model_path is None:
        raise Exception("请选择KNN分类器方式：knn_clf 或 model_path")
        
    #加载KNN模型
    #rb读入二进制数据
    if knn_clf is None:
        with open(model_path,'rb') as f:
            knn_clf = pickle.load(f)
            
    #加载图片，发现人脸位置
    X_img = fr.load_image_file(X_img_path)
    X_face_locations = fr.face_locations(X_img)
    
    #编码
    encodings = fr.face_encodings(X_img,known_face_locations=X_face_locations)
    
    #利用KNN找出匹配图片
    closest_distances = knn_clf.kneighbors(encodings,n_neighbors = 3)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    
    #判断类别
    return[(pred,loc) if rec else ("unknown",loc) for pred,loc,rec in zip(knn_clf.predict(encodings),X_face_locations,are_matches)]
            
#识别结果可视化
def show_names_on_image(img_path,predictions)    :
    """
    :param img_path:待识别图片位置
    :Param prediction:预测结果
    """
    
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    
    #画出人脸边界盒子
    for name,(top,right,bottom,left) in predictions:
        draw.rectangle(((left,top),(right,bottom)),outline = (225,0,255))

    #生成utf-8格式
    name = name.encode("UTF-8")
    name = name.decode("ascii")
    
    #写下名字作为标签
    text_width,text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10),(right,bottom)),fill = (225,0,255),outline = (225,0,255))
    draw.text((left+6,bottom - text_height - 5),name,fill = (255,255,255))
    
    #名字列表
    li_names.append(name)
    
    #从内存删除
    del draw
    
    #显示结果图 
    pil_image.show()
    
#统计分析
    
li_names = []
    
#计算总人数
def count(train_dir):
    path = train_dir
    count = 0
    for fn in os.listdir(path):#fn代表文件夹
        count = count + 1
    return count
            
#获取所有名字的列表
def list_all(train_dir):
    path = train_dir
    result = []
    for fn in os.listdir(path):
        result.append(fn)
    return result
            
#输出结果
def stat_output():
    s_list = set(li_names)
    s_list_all = set(list_all("examples/train"))
    if "unknown" in s_list:
        s_list.remove("unknown")
        
    tot_num = count("examples/train")
    s_absent = set(s_list_all - s_list)
    print("\n")
    print("==================================================================")
    print("全体名单",s_list_all)
    print("已到名单",s_list)
    print("应到人数",tot_num)
    print("已到人数",len(s_list))
    print("出勤率:{:.2f}".format(float(len(s_list))/float(tot_num)))
    print("未到",s_absent)
    print("==================================================================")
            
#运行
if __name__ ==  "__main__":
    #1.训练分类器
    print("正在训练KNN分类器请稍后...")
    print("训练详情：")
    train("examples/train",model_save_path="trained_knn_model.clf",n_neighbors=3)
    print("训练完成！")
    #2.利用训练好的模型预测新照片
    for image_file in os.listdir("examples/test"):
        full_file_path = os.path.join("examples/test",image_file)
        
        #利用分类器找出人脸
        predictions = predict(full_file_path,model_path = "trained_knn_model.clf")
    #3.输出结果
        #打印
        print("识别结果如下：")
        for name,(top,right,bottom,left) in predictions:
            print("发现{},位置:{},{}.".format(name,top,right))
        #显示名字
        show_names_on_image(os.path.join("examples/test",image_file),predictions)
        
        #统计数据
        stat_output()
        