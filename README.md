# MEGA_project
Курсовой проект от ведущего мобильного оператора в России

Стек:

ML: dask, sklearn, pandas, numpy, scipy, catboost, xgboost, lightgbm, hyperopt, tensorflow,

tensorflow-addons, keras 

API: Luigi

   Здравствуйте!
   
   Знакомство с проектом прошу начинать с файла Presentation.pdf
   
   В папке  notebooks находятся:
- ноутбук eda_before_merge с разведочным анализом исходных датасетов
- ноутбук eda_after_merge с разведочным анализом после объединения исходных датасетов
- ноутбук Feature_Endineering с предобработкой данных и генерированием новых признаков
- ноутбук ML_MEGAFON_END с проработкой методов предсказания

   В файле Luigi_pipeline_Catb.py находится обернутый в Luigi пайплайн модели на основе Катбуста
   
   В файле Luigi_pipeline_NN.py находится обернутый в Luigi пайплайн модели на основе нейронной сети
   
   Файлы my_pca и my_functions используются в файлах Luigi_pipeline_Catb и Luigi_pipeline_NN