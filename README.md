# Project-3.EDA+ Feature Engineering

### Описание проекта    
Представьте, что вы работаете датасаентистом в компании Booking. Одна из проблем компании — это нечестные отели, которые накручивают себе рейтинг. Одним из способов нахождения таких отелей является построение модели, которая предсказывает рейтинг отеля. Если предсказания модели сильно отличаются от фактического результата, то, возможно, отель играет нечестно, и его стоит проверить.

## Задача проекта 
Построить модель, которая предсказывает рейтинг отеля.

## Проект будет состоять из пяти частей:

1. Постановка задачи
2. Загрузка данных
3. Обработка признаков
4. Отбор признаков
5. Обучение модели 

**Метрика качества**     
Результаты оцениваются по метрике MAPE.

## Библиотеки
* Базовые библиотеки: Pandas, NumPy
* Визуализация: Matplotlib, Seaborn, Plotly

### [Cсылка на Проект 3](https://github.com/Amina313/-3.-EDA/blob/main/project_3_eda.ipynb)



import numpy as np 
import pandas as pd 
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_selection import chi2 # хи-квадрат
from sklearn.preprocessing import MinMaxScaler
# используем тест ANOVA для непрерывных признаков
from sklearn.feature_selection import f_classif # anova
# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели
