# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: 

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):
    # Calculate centroid
    cent = np.mean(points[0:2,:], axis = 1)
    
    # Calculate the average distance of the points having the centroid as origin
    dist = np.linalg.norm(points[0:2,:] - cent.reshape(2,1), axis = 0)
    dist_m = np.mean(dist)

    # Define the scale to have the average distance as sqrt(2)
    esc = np.sqrt(2)/dist_m

    # Define the normalization matrix (similar transformation)
    T = np.array([[esc, 0, -esc*cent[0]], [0, esc, -esc*cent[1]], [0,0,1]])

    # Normalize points
    norm_points = T@points
    
    return norm_points, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):
    npoints = pts1.shape[1]
    A = np.zeros((3*npoints,9))

    for k in range(npoints):
        A[3*k,3:6] = -pts2[2,k]*pts1[:,k]
        A[3*k,6:9] =  pts2[1,k]*pts1[:,k]

        A[3*k+1,0:3] =  pts2[2,k]*pts1[:,k]
        A[3*k+1,6:9] = -pts2[0,k]*pts1[:,k]

        A[3*k+2,0:3] = -pts2[1,k]*pts1[:,k]
        A[3*k+2,3:6] =  pts2[0,k]*pts1[:,k]

    return A

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):
    # Add homogeneous coordinates
    pts1 = np.hstack(( pts1, np.ones((pts1.shape[0],1)) )).T
    pts2 = np.hstack(( pts2, np.ones((pts2.shape[0],1)) )).T

    # Normalize points
    pts1n, T1 = normalize_points(pts1)
    pts2n, T2 = normalize_points(pts2)

    # Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
    A = compute_A(pts1n, pts2n)

    # Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada 
    U,S,Vt = np.linalg.svd(A)

    h = Vt[-1,:]
    H_nomalizada = np.reshape(h,(3,3))

    # Denormaliza H_normalizada e obtém H
    H = np.linalg.inv(T2)@H_nomalizada@T1

    return H


# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem 
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado 
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens


def RANSAC(pts1, pts2, dis_threshold, N, Ninl):
    
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc 
    

    # Processo Iterativo
        # Enquanto não atende a critério de parada
        
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado

        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado

        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        # Atualiza também o número N de iterações necessárias

    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.

    return H, pts1_in, pts2_in


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('comicsStarWars02.jpg', 0)   # queryImage - box
img2 = cv.imread('comicsStarWars01.jpg', 0)        # trainImage - photo01a

# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])#.reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])#.reshape(-1, 1, 2)
    
    #################################################
    M = compute_normalized_dlt(src_pts, dst_pts)
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
