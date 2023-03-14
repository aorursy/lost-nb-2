#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


a simple SVD with 10 axis used as similarity (actually i couldn't see much difference with 3 axis already')
trick: you transpose the data train.csv and copy 5 times the 192 properties matrix resulting in an matrix 1152 row properties x 990 column species

    paste this matrix in this javascript online
    [http://www.gaverapotheek.be/SVD/SVD10d.htm][1]
    copy the first leaf from the test.csv transposed, also 5 times making it equally long in the Query field
    
due to memory problems i have to 'castrate' abit the SVD.htm javascript, and prevent to paste the U,S,V matrix in the placeholders of the html.

    the result
    leaf 4 
    
    resembles most with

Column938	99.9%	1508	Quercus_Castaneifolia
Column100	99.6%	178	Ginkgo_Biloba >>>>>  very  strange....
Column294	99.5%	487	Quercus_Kewensis
Column623	99.5%	989	Quercus_Kewensis
Column469	99%	765	Callicarpa_Bodinieri
Column594	98%	942	Tilia_Tomentosa
Column720	98.9%	1159	Acer_Opalus
Column8	98.8%	14	Quercus_Brantii
Column626	98.8%	993	Zelkova_Serrata
Column977	98.7%	1563	Quercus_Pubescens
Column298	98.3%	491	Quercus_Trojana
Column527	98.3%	852	Quercus_Kewensis
Column583	98.2%	926	Castanea_Sativa
Column798	98.2%	1281	Quercus_Ellipsoidalis
Column10	98.1%	17	Zelkova_Serrata
Column438	97.6%	718	Zelkova_Serrata
Column822	97.5%	1322	Prunus_Avium
Column880	97.2%	1410	Zelkova_Serrata
Column358	97.1%	592	Castanea_Sativa
Column697	97.1%	1119	Callicarpa_Bodinieri


    
    lets check a manual visual lookup...
    just visually without seeing the colours.. lets compare those results leaf 4 compares
    
Column0	90.7%	1	Acer_Opalus	
Column1	83.3%	2	Pterocarya_Stenoptera	
Column2	61.7%	3	Quercus_Hartwissiana	
Column3	79.5%	5	Tilia_Tomentosa	
Column4	41.6%	6	Quercus_Variabilis	>> this compares
Column5	20%	8	Magnolia_Salicifolia	
Column6	45.5%	10	Quercus_Canariensis	 >> compares
Column7	48%	11	Quercus_Rubra	
Column8	98.8%	14	Quercus_Brantii	>> compares
Column9	73.7%	15	Salix_Fragilis	
Column10	98.1%	17	Zelkova_Serrata	>> compares
Column11	59.3%	18	Betula_Austrosinensis	
Column12	49%	20	Quercus_Pontica	
Column13	75.7%	21	Quercus_Afares	
Column14	83%	22	Quercus_Coccifera	
Column15	68.6%	25	Fagus_Sylvatica	
Column16	75.1%	26	Phildelphus	
Column17	27.6%	27	Acer_Palmatum	
Column18	89.6%	29	Quercus_Pubescens	
Column19	80.3%	30	Populus_Adenopoda	
Column20	84.3%	31	Quercus_Trojana	
Column21	51.6%	32	Quercus_Variabilis	
Column22	76.8%	34	Alnus_Sieboldiana	
Column23	35.6%	35	Quercus_Ilex	
Column24	54.1%	37	Arundinaria_Simonii	
Column25	87.1%	38	Acer_Platanoids	
Column26	62.7%	40	Quercus_Phillyraeoides	
Column27	26.2%	42	Cornus_Chinensis	
Column28	74.1%	43	Quercus_Phillyraeoides	
Column29	43.4%	45	Fagus_Sylvatica	
Column30	16.5%	48	Liriodendron_Tulipifera	
Column31	51.4%	49	Cytisus_Battandieri	
Column32	85.5%	50	Tilia_Tomentosa	
Column33	21.6%	54	Rhododendron_x_Russellianum	
Column34	72.1%	55	Alnus_Rubra	
Column35	53.1%	56	Eucalyptus_Glaucescens	
Column36	20.1%	58	Cercis_Siliquastrum	
Column37	33.8%	60	Cotinus_Coggygria	
Column38	43%	61	Celtis_Koraiensis	
Column39	80.3%	63	Quercus_Crassifolia	
Column40	28.3%	64	Quercus_Variabilis	


  [1]: http://www.gaverapotheek.be/SVD/SVD10d.htm


# In[3]:


Try a second leaf nr 9
its a more typical acer leaf, this should become more clearcut

Column434	98.2%	713	Acer_Rufinerve
Column658	98.5%	1040	Tilia_Platyphyllos
Column534	98.6%	860	Quercus_Pontica
Column947	98.6%	1520	Acer_Rufinerve
Column631	98.7%	1000	Acer_Circinatum  >>>>>>>>>>>>> this compares 
Column964	98.7%	1547	Prunus_X_Shmittii
Column91	98.9%	164	Acer_Circinatum  >>>>>>>>>>>> this compares
Column400	99.1%	657	Quercus_Pontica
Column34	99.3%	55	Alnus_Rubra
Column793	99.3%	1273	Castanea_Sativa
Column940	99.3%	1511	Castanea_Sativa
Column396	99.4%	651	Acer_Rufinerve
Column89	99.5%	160	Acer_Circinatum  >>>>>>>>>>>>>>>>manually i would choose this leaf.
Column99	99.5%	175	Betula_Austrosinensis
Column163	99.5%	286	Betula_Austrosinensis
Column715	99.5%	1152	Tilia_Platyphyllos
Column764	99.6%	1228	Morus_Nigra
Column172	99.7%	304	Morus_Nigra
Column432	99.7%	711	Morus_Nigra
Column133	99.8%	239	Tilia_Platyphyllos
Column289	99.8%	475	Alnus_Rubra
Column802	99.8%	1286	Prunus_Avium
Column84	99%	148	Acer_Rufinerve

