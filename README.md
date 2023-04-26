# Как использовать этот репозиторий?

```
-src
    -models [скрипты и модули для обучения моделей]
    -data [скрипты и модули для предобработки данных]
-config [файлы конфигах, которые, как правило, используются в скриптах src. Там можно изменять гиперпараметры модели для обучения. Конфигурационный файл обучающий модель называется также, как и скрипт для обучения этой модели в ./src/model/]
-data [содержит как данные датасетов в стандартных форматах, так и сериализованные данные. Как правило, ссылки на сохранение данных, которые лежат в конфиге ссылаются на эту папку]
-models [папка для сериализации моделей]
-artefacts [папка содержит картинки графиков лоссов]
```

Как пользоваться:
1. `pd_milk_data.csv` - это просто файл с данными по молочным таварам, которые извлечены из изначальных партиций даска;
2. В папке репозитория, если нет сериализованных `data/test_dataset.pt`, `data/train_dataset.pt`, введи `python3 src/data/prepare_data.py`;
3. После того, как датасеты будут, то чтобы начать обучать бейзлайн введии следующее `python3 src/model/training_CatEmbLSTM.py`;
4. После того, как модель обучится (пока что стейджи не сохраняются), она сериализуется в папку (имя берётся из поля `name` конфигурационного файла, пока что тут без mlflow).

Чтобы поменять количество эпох и размер батча у бейзлайна, то открой `config/CatEmbLSTM.yaml`.

Если же хочется поменять оптимизатор с `SGD` на `Adam`, то в файле `config/CatEmbLSTM.yaml` напротив поля `optimizer` напиши `Adam`.