# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: 

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
import random
# Set a seed for get the same results
random.seed(100)

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

    # Avoid errors of division by zero
    if dist_m==0:
        dist_m = 1e-16
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
    
    max_inliers_past = []
    best_H = None
    num_pts = pts1.shape[0]

    # Processo Iterativo
    for _ in range(N):
        # Enquanto não atende ao primeiro critério de parada (n de iterações)
        
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2
        # Semente foi setada acima para reproduzir os mesmos resultados
        indices = random.sample(range(num_pts), 4)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        H = compute_normalized_dlt(sample_pts1, sample_pts2)
        
        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado
        projected_pts = H@np.vstack((pts1.T, np.ones(num_pts)))
        # Evita erro de divisão por zero
        projected_pts[projected_pts == 0] = 1e-16
        # Normaliza pelo ulltimo elemento fazendo com que posição 2 == 1 
        projected_pts /= projected_pts[2]
        # Afere-se a distância entre o ponto da imagem 2 e os pontos projetados referentes a DLT normalizada
        # Da amostra aleatória de 4 pontos do SIFT
        distances = np.linalg.norm(pts2.T - projected_pts[:2], axis=0)
        # Cria uma mascara para verificar em quais indexes tem-se distancia menor que o limiar
        inliers_mask_now = np.where(distances < dis_threshold)[0]

        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        # Atualiza também o número N de iterações necessárias
        # Compara os inliers atuais com os inliers passados
        if len(inliers_mask_now) > len(max_inliers_past):
            max_inliers = inliers_mask_now
            best_H = H
            # proportion of outliers (%)
            e = 0.01
            N = int(np.log(1 - 0.99) / np.log(1 - (1-e) ** 4))

    # Ao sair do loop a variavel best_H conterá a homografia com o melhor número de inliers
    if best_H is not None:
        best_H = compute_normalized_dlt(pts1[max_inliers], pts2[max_inliers])

    return best_H, pts1[max_inliers], pts2[max_inliers]


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT
im1s = ['box.jpg','batman.jpg','outdoors01.jpg','comicsStarWars01.jpg','mesa_revista03.jpg']
im2s = ['photo01a.jpg','outdoor_batman.jpg','outdoors02.jpg','comicsStarWars02.jpg','mesa02_1.jpg']
for im1,im2 in zip(im1s,im2s):
    print("Processing",im1,"and",im2)
    MIN_MATCH_COUNT = 10
    img1 = cv.imread(im1, 0)             # queryImage - box
    img2 = cv.imread(im2, 0)        # trainImage - photo01a

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
        # Define os parâmetros do RANSAC
        dis_threshold = 5  # Limiar de distância
        N = 10000           # Número máximo de iterações
        Ninl = 10          # Limiar mínimo de inliers

        # Chama a função RANSAC
        M, src_inliers, dst_inliers = RANSAC(src_pts, dst_pts, dis_threshold, N, Ninl)
        # M = compute_normalized_dlt(src_pts, dst_pts)
        #################################################

        img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

        # Criar uma nova lista de keypoints a partir dos inliers do RANSAC
        kp1_inliers = []
        for x in src_inliers:
            kp1_inliers.append(cv.KeyPoint(x[0], x[1], 1))
        kp2_inliers = []
        for x in dst_inliers:
            kp2_inliers.append(cv.KeyPoint(x[0], x[1], 1))
        
        # Criar novos matches para os inliers do RANSAC
        good_inliers = []
        for i in range(0,len(kp1_inliers)):
            good_inliers.append(cv.DMatch(i,i,0))

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    flags = 2)

    #img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    img3 = cv.drawMatches(img1, kp1_inliers, img2, kp2_inliers, good_inliers, None, **draw_params)

    # Cria subplots para impressao dos resultados
    fig, axs = plt.subplots(2, 2, figsize=(30, 15))

    axs[0,0].set_title('Resultado RANSAC')
    axs[0,0].imshow(img3, 'gray')

    axs[0,1].set_title('Primeira imagem')
    axs[0,1].imshow(img1, 'gray')

    axs[1,0].set_title('Segunda imagem')
    axs[1,0].imshow(img2, 'gray')

    axs[1,1].set_title('Primeira imagem após transformação')
    axs[1,1].imshow(img4, 'gray')

    plt.savefig("resultado"+ im1.replace(".jpg","") + "_" + im2.replace(".jpg","") + "_.png")
    plt.show()

    ########################################################################################################################
