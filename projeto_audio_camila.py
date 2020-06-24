#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from IPython.display import Audio

import sklearn.preprocessing

import os

import warnings
warnings.filterwarnings("ignore")


# ### ESPECIFICAÇÕES DO PROJETO
# 
# Projeto
# 
# 1 - Baixar o mini dataset http://opihi.cs.uvic.ca/sound/mini-genres.tar.bz2 que contém uma amostra de diferentes gêneros de música
# 
# 2 - Carregar os áudio com um limite de 10s por arquivo
# 
# 3 - Montar um script para extrair as seguintes features (hop_len e n_fft, podem manter padrão):
# 
# 4 - Espectrograma em escala logaritmica de todos os áudios (figura)
# 
# 5 - Espectrograma da escala de mel (MFCC) para n_mcc=13 (figura)
# 
# 6 - Chroma feature (figura)
# 
# 7 - Spectral Centroid, Spectral Rolloff, Spectral Bandwidth, zero crossing

# In[2]:


#Exploração de quantos arquivos de áudios possuímos


# In[3]:


get_ipython().system('ls mini-genres/classical')


# In[4]:


get_ipython().system('ls mini-genres/jazz')


# In[5]:


get_ipython().system('ls mini-genres/metal')


# In[6]:


get_ipython().system('ls mini-genres/pop')


# In[7]:


get_ipython().system('ls mini-genres/rock')


# In[12]:


def transforma_audio():
    
    '''
    Recebe os arquivos de audio a partir de um caminho,
    e devolve as features e os gráficos de espectograma, chrome feature e mfcc
    '''
#    
#definição do caminho para se buscar os áudios
#
    cwd = os.getcwd()
    rootdir = cwd+'/mini-genres'
    caminhos_audio = []

#para cada caminho de audio identificado os adiciona a uma lista para servirem ao load.librosa
#
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if ".au" in file:
                caminhos_audio.append(os.path.join(root, file))
                
#separando o gênero musical e o número da faixa para salvar tudo que for imagem no dir local
#
    for item in caminhos_audio:
        genero = item.split('/')[10]
        faixa = item.split('.')[-2]
        print("gerando os gráficos e salvando em seu diretório atual. Aguarde...")
        print("Gênero musical: ", genero)
        print("Faixa: ", faixa)
#gerando os arrays   
        arqs_audio = librosa.load(item, duration=10)
        arqs_audio = list(arqs_audio)
        #print(arqs_audio)
    
#para cada arquivo de audio gera o gráfico em escala logaritimica para Espectograma - Domínio da Frequência
#
        x, sr = librosa.load(item, duration=10)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequencia (Hz)')
        plt.colorbar()
        plt.set_cmap('inferno')
        filename = f"{genero}_{faixa}_espectograma_log"
        os.mkdir(cwd+f'/{genero}_{faixa}/')
        path = cwd+f'/{genero}_{faixa}/'
        plt.savefig(os.path.join(path, filename))

    #mfcc
    #
        mfccs = librosa.feature.mfcc(x, sr=sr, n_mfcc=13)
        plt.figure(figsize=(9, 3))
        librosa.display.specshow(mfccs, sr=sr)
        plt.xlabel('time (s)')
        plt.ylabel('MFCC coefficients')
        plt.title('Espectrograma (MFCC) para n = 13')
        plt.set_cmap('inferno')
        plt.colorbar()
        filename = f"{genero}_{faixa}_mfcc13"
        #os.mkdir(cwd+f'/{genero}_{faixa}_mfcc/')
        path = cwd+f'/{genero}_{faixa}/'
        plt.savefig(os.path.join(path, filename))


    #chroma feature
    #
        plt.figure(figsize=(9,3))
        chroma_feature = librosa.feature.chroma_stft(x, sr=sr)
        librosa.display.specshow(chroma_feature)
        plt.xlabel('time (s)')
        plt.ylabel('Pitch Class')
        plt.title('Chroma Feature')
        plt.set_cmap('inferno')
        plt.colorbar()
        filename = f"{genero}_{faixa}_chroma_feature"
        #os.mkdir(cwd+f'/{genero}_{faixa}_chroma_feature/')
        path = cwd+f'/{genero}_{faixa}/'
        plt.savefig(os.path.join(path, filename))


    #spectral centroids
    #
        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        df = pd.DataFrame(data = {'spectral_centroids':spectral_centroids})
        csv_name = 'spectral_centroids' + '_' + genero + '_' + faixa
        df.to_csv(cwd+f'/{genero}_{faixa}/' + csv_name + '.csv')


    #spectral rolloff
    #
        rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
        df = pd.DataFrame(data = {'spectral_rollof':rolloff})
        csv_name = 'spectral_rollof' + '_' + genero + '_' + faixa
        df.to_csv(cwd+f'/{genero}_{faixa}/' + csv_name + '.csv')


    #spectral bandwidth
    #
        bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)[0]
        df = pd.DataFrame(data = {'spectral_bandwidth':bandwidth})
        csv_name = 'spectral_bandwidth' + '_' + genero + '_' + faixa
        df.to_csv(cwd+f'/{genero}_{faixa}/' + csv_name + '.csv')

    #zero crossings
    #
        zero_crossing = librosa.zero_crossings(x)
        soma = [sum(zero_crossing)]
        df = pd.DataFrame(data = {'zero_crossings':zero_crossing})
        csv_name = 'zero_crossings' + '_' + genero + '_' + faixa
        df.to_csv(cwd+f'/{genero}_{faixa}/' + csv_name + '.csv')
        
        
    print("---ok---")


# In[13]:


transforma_audio()


# In[ ]:




