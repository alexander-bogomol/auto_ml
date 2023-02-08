# Бинарный классификатор.
Данная библиотека представляет собой базовую реализацию автоматического бинарного классификатора.
Для инициализации автоклассификатора необходимо передать список моделей, среди которых будет выбрана лучшая
(на основании указанной при тренировке метрики).

На данный момент доступны для оценки следующие модели:
- SVM (Support Vector Machines)
- LOGREG (Logistic Regression)
- KNN
- DECISIONTREE
- NAIVEBAYES 
- RANDOMFOREST

И метрики:
- F1
- ACCURACY
- PRECISION
- RECALL

Так же в автоклассификаторе **после тренировки** реализованы следующие функции:
- Если во время инициализации среди переданного автоклассификатору списка моделей были модели логистической регрессии или хотя бы одна из "деревянных" - то будет доступна возможность оценить важность признаков через метод `.feature_importance()`. В качестве аргумента метод принимает одно из значений: `forest`, `tree`, `logreg`.
- Метод `.models_ranking()` позволяет вывести список обученных n-моделей, отсортированных по их метрикам (метрика указывается при тренировке). Доступный аргумент - `top`
- Лучшую модель можно получить методом `.get_best_model`
- С помощью `.best_predict()` можно получить предсказания лучшей модели. За лейблы предсказаний отвечает булевый аргумент `inverse_transform`


Для использования библиотеки необходимо установить зависимости:

```bash
conda env create -f environment.yml
# для pip
python3 -m pip install -r requirements.txt
```

В качестве данных для обучения афтоклассификатор ожидает объект типа `pd.Dataframe`. Кроме того, ожидается:
1. В данных нет пропусков
2. Все столбцы приведены к типам int|float - для количественных, object - для категориальных признаков
3. Для категориальных признаков используется OneHotEncoder. Пожалуйста, учитывайте это, если в датафрейме используется большое количество категорий.
4. Невозможна тренировка одинаковых видов классификаторов.

[Примеры использования](https://github.com/alexander-bogomol/auto_ml/blob/main/examples.ipynb)