#!/usr/bin/python3 
import glob
import cv2
import os
import numpy as np

def main():
    while(True):
        path     = input("Entre com caminho a partir de onde você está:\n")
        pat_w    = input("Entre com a quantidade de pontos na horizontal:\n") 
        pat_h    = input("Entre com a quantidade de quinas na vertical:\n") 
        sqr_size = input("Entre com o tamanho do quadrado em milímetros:\n") 

        images_list = readImages(path)

        if (images_list != []):
            if ((pat_w != '') and (pat_h != '')) and (sqr_size != ''):
                calibrate(images_list,pattern_size=(int(pat_w), int(pat_h)),square_size=int(sqr_size)) 
                break   
            elif ((pat_w == '') or (pat_h == '')) and (sqr_size != '') :
                calibrate(images_list,square_size=int(sqr_size))
                break    
            elif ((pat_w != '') and (pat_h != '')) and (sqr_size == '') :
                calibrate(images_list,pattern_size=(int(pat_w), int(pat_h)))
                break    
            else:
                calibrate(images_list)
                break

def readImages(path):

    images_name_list = []

    script_path = os.getcwd()
    if not (script_path[-1] == '/'):
        script_path += '/'
    if not (path[-1] == '/'):
        path += '/'
    full_path = script_path + path
     
    #if not os._exists(full_path):
    #    print(f'O caminho {full_path}, não existe!')
    #    return []

    types = ['*.bmp',  '*.dib', '*.jpeg', '*.jpg', '*.jpe',
             '*.jp2',  '*.png', '*.webp', '*.pbm', '*.pgm',
             '*.ppm',  '*.pxm', '*.pnm',  '*.sr',  '*.ras',
             '*.tiff', '*.tif', '*.exr',  '*.hdr', '*.pic']
    
    for _ ,type in enumerate(types):
        found = glob.glob(full_path + type)
        if not (found == []):
            images_name_list.extend(found) 

    if images_name_list == []:
        print(f"\nErro ao carregar as imagens!")
        return []
    else:
        print(f"\nCarregamento finalizado! Foram carregadas {len(images_name_list)} imagens")

    return images_name_list

def calibrate(images_list, pattern_size=(8, 6), square_size=39, scale=(1,1)):
# O square_size está em milímetros (os parametros intrínsecos estarão na mesma unidade utilizada no square_size) 

    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2) * square_size 
    objpoints = [] 
    imgpoints = [] 

    nx, ny = pattern_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

    for image in images_list:
        img_rgb = cv2.imread(image)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        patternWasFound, corners = cv2.findChessboardCorners(img_gray, (nx,ny), None)

        if patternWasFound == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img_gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
#            cv2.drawChessboardCorners(img_rgb, (pattern_size[0],pattern_size[1]), corners2, patternWasFound)
#            cv2.imshow('image', img_rgb)
#            cv2.waitKey(1000)

    [w,h] = img_gray.shape[:2]
    _ , mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
 
    print(f'\nMatriz K, parâmetros intrínsecos da câmera:')
    print(f'\n{newcameramtx}')
    print(f'\nErro de reprojeção: {tot_error/len(objpoints)} pixels')  
    print(f"\nFx: {newcameramtx[0,0]:.2f}, Fy: {newcameramtx[1,1]:.2f}")

    if len(rvecs) > 0:
        individual_errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            individual_errors.append(error)
        
        print(f"\nErros das imagens individuais: {np.array(individual_errors)}")
        print(f"Desvio padrão dos erros individuais: {np.std(individual_errors):.3f} pixels")
 
main()    